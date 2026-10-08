# Copyright 2025-2026 EUMETSAT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MTG FCI L1C DirectZarr ingestion plugin.

A scene arrives either as one ZIP or as a bundle of loose BODY/TRAIL chunk
files. Each batch materialises its inputs through Firecube, extracts ZIPs,
optionally builds the pixel-time lookup table, and emits Firecube
``WriteIntent`` objects. Core owns the actual Zarr writes and time-axis growth.
"""

# mypy: disable-error-code=import-untyped

from __future__ import annotations

import dataclasses
import logging
import posixpath
import threading
from collections.abc import Callable, Iterable, Sequence
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np  # pyright: ignore[reportMissingImports]

from firecube.ingestor.api import (  # pyright: ignore[reportMissingImports]  # type: ignore[import-untyped]
    ConfigurationError,
    DirectZarrIngestor,
    PipelineBatch,
    PipelineRunState,
    PluginContext,
    RuntimeIngestContext,
    WriteIntent,
    ZarrGroupSpec,
    ZarrTemplateConfig,
    register_ingestor,
)
from firecube.core.api import (  # pyright: ignore[reportMissingImports]  # type: ignore[import-untyped]
    BatchResourceRegistry,
    IndexSpec,
    IndexedWrite,
    ItemInfo,
    TimeAxis,
    normalize_epoch_iso,
)

from ._constants import (
    CONSTANTS,
    PRODUCT_TYPE_FDHSI,
    PRODUCT_TYPE_HRFI,
    REPEAT_CYCLE_MINUTES,
    REPEAT_CYCLES_PER_DAY,
    dimsize_for,
)
from ._data import (
    SceneBundle,
    classify_items,
    detect_product_type,
    extract_slot_time_from_path,
    group_chunks_into_bundles,
    is_valid_fci_zip,
)
from .geolocation import LatLonProvider
from ._ephemeris import sun_earth_distance_au
from ._group_plan import (
    GroupPlan,
    group_name,
    validate_effective_resolutions,
    resolve_group_plans,
)
from ._decode import (
    AssemblyPreconditionError,
    ChannelCalibration,
    ChannelSlicePayload,
    ChunkOwnedAssembler,
    SharedNcPartReader,
    SlotGeometryAccumulator,
    TimeMapAccumulator,
    list_fci_nc_parts,
)
from .config import MtgFciL1cConfig
from ._variables import TIME_COORD_NAME, build_all_specs

log = logging.getLogger("firecube.ingestor.mtg_fci_l1c")

if TYPE_CHECKING:
    from ._scratch import BatchScratch
    from ._variables import Variable, VariableContext


@dataclasses.dataclass
class BatchResources:
    """Per-batch resources registered for lifecycle-managed cleanup."""

    shared_reader: SharedNcPartReader | None = None
    batch_scratch: BatchScratch | None = None
    chunk_owned_cache: Any | None = None


def _assemble_and_extract(
    batch_id: str,
    batch_resources_ref: dict[str, BatchResources],
    batch_resources_lock: threading.Lock,
    nc_parts: list[Path],
    nc_channel: str,
    index2time: dict[int, float] | None,
    pixel_time_dtype: np.dtype,
    group: str,
    timestamp: Any,
    y_range: tuple[int, int],
    variable_set: frozenset[str],
    part_row_ranges: dict[Path, tuple[int, int]],
    variable_source: Any,
    ctx: Any,
) -> np.ndarray:
    """Assemble a spatial chunk at dispatch time and project one variable.

    Raises ``ValueError`` if the projected source returns ``None``. Callers
    (``_emit_spatial_intents``) probe every source with a dummy payload before
    emitting an intent, so a ``None`` here means the probe missed a case and
    Firecube core would otherwise be handed ``data=None`` and corrupt the
    region write.
    """
    with batch_resources_lock:
        resources = batch_resources_ref.get(batch_id)
    if resources is None or resources.chunk_owned_cache is None:
        raise RuntimeError(f"No assembler for batch {batch_id!r}")

    payload = resources.chunk_owned_cache.assemble(
        nc_parts,
        nc_channel,
        index2time,
        pixel_time_dtype,
        group,
        timestamp,
        y_range,
        variable_set,
        part_row_ranges,
    )
    ctx_with_payload = dataclasses.replace(ctx, channel_payload=payload)
    result = variable_source(ctx_with_payload)
    if result is None:
        raise ValueError(
            f"variable source {variable_source!r} returned None at dispatch "
            "time; the intent should not have been emitted."
        )
    return result


def _item_name(item: Any) -> str:
    """Return the file name of a local path or remote URI item.

    Plain string handling: ``Path`` would collapse ``s3://b/x`` into
    ``s3:/b/x``.
    """
    return posixpath.basename(str(item))


def _item_product_type(item: Any) -> str | None:
    """Return the product type named in an item's file name, if any."""
    try:
        return detect_product_type(_item_name(item))
    except ValueError:
        return None


def _output_chunk_ranges(dimsize: int, chunk_y: int) -> list[tuple[int, int]]:
    """Return half-open output chunk y-ranges for one resolution group."""
    return [
        (start, min(start + chunk_y, dimsize)) for start in range(0, dimsize, chunk_y)
    ]


def _intersecting_part_ranges(
    nc_part_ranges: list[tuple[Path, tuple[int, int]]],
    y_range: tuple[int, int],
) -> list[tuple[Path, tuple[int, int]]]:
    """Return nc_parts whose row ranges intersect an output chunk."""
    y_start, y_end = y_range
    intersecting = [
        (part_path, part_range)
        for part_path, part_range in nc_part_ranges
        if y_start < part_range[1] and y_end > part_range[0]
    ]
    return sorted(intersecting, key=lambda item: item[1][0])


def _validate_contiguous_part_ranges(
    intersecting: list[tuple[Path, tuple[int, int]]],
    y_range: tuple[int, int],
) -> None:
    """Raise when intersecting nc_parts do not cover the output chunk exactly."""
    y_start, y_end = y_range
    cursor = y_start
    for _part_path, (part_start, part_end) in intersecting:
        clipped_start = max(part_start, y_start)
        clipped_end = min(part_end, y_end)
        if clipped_start != cursor:
            raise AssemblyPreconditionError(
                f"Output chunk y={y_range} has a gap before row {clipped_start}; "
                "nc_part row ranges must be contiguous."
            )
        cursor = clipped_end
    if cursor != y_end:
        raise AssemblyPreconditionError(
            f"Output chunk y={y_range} is only covered through row {cursor}; "
            "nc_part row ranges must be contiguous."
        )


@register_ingestor("mtg_fci_l1c")
class MtgFciL1cIngestor(DirectZarrIngestor):
    """DirectZarr ingestor for MTG FCI L1C ZIPs and loose chunk files."""

    PRODUCT_NAME: ClassVar[str] = "mtg_fci_l1c"
    name = "mtg_fci_l1c"
    plugin_config_class = MtgFciL1cConfig

    # Keep schema, ingestor, and write-intent time coordinates on the same name.
    time_dim_name: ClassVar[str] = TIME_COORD_NAME

    # Bump this if the declared time-axis definition (epoch, cadence, mode) changes.
    INDEX_MODEL: ClassVar[str] = "eumetsat_repeat_cycle_v1"

    def index_spec(self, ctx: PluginContext) -> IndexSpec | None:
        """Declare the repeat-cycle index model for each resolution group.

        Always returns the declared axis; ``slot_count`` is ``None`` when no
        fixed extent is configured (serial mode). The engine's parallel gate
        refuses slot-range flags loudly for unbounded axes. Invalid
        ``time_slots`` / ``time_end`` values raise instead of being
        silently swallowed.
        """
        del ctx
        config: MtgFciL1cConfig = self.plugin_config  # type: ignore[assignment]
        product_type = self._require_product_type()
        validate_effective_resolutions(config, product_type)
        return self._build_index_spec(product_type)

    def _build_index_spec(self, product_type: str) -> IndexSpec:
        """Build the declared time-axis spec; ``slot_count`` may be ``None``.

        Single source of truth for the axis definition; the engine resolves
        it for both serial and parallel ingestion, so slot positions always
        come from one axis definition.
        """
        config: MtgFciL1cConfig = self.plugin_config  # type: ignore[assignment]
        # Same resolution set as zarr_schema(), so every IndexSpec group exists.
        resolutions = config.effective_resolutions(product_type)
        cadence_s = REPEAT_CYCLE_MINUTES * 60
        epoch_iso = normalize_epoch_iso(f"{config.time_epoch}T00:00:00Z")
        axis = TimeAxis.observed(
            coordinate=TIME_COORD_NAME,
            epoch=epoch_iso,
            cadence_s=cadence_s,
            slot_count=self._configured_total_slots(),
        )
        name = self.INDEX_MODEL
        if config.flat_store:
            # Flat stores of different grids share the root group and axis, so the
            # grid goes into the index identity to keep them from being appended to
            # each other.
            name = "_".join([name, product_type.lower(), *resolutions])
        return IndexSpec(
            name=name,
            groups={group_name(res, config.flat_store): axis for res in resolutions},
        )

    def inspect_item(self, item: Any, ctx: PluginContext) -> ItemInfo | None:
        """Return the time coordinate label of a ZIP item or a chunk bundle.

        A ZIP is labelled with its sensing start floored to the minute; a
        chunk bundle with the nominal start of its repeat cycle.
        """
        del ctx
        if isinstance(item, SceneBundle):
            return ItemInfo(coordinate=item.coordinate)
        timestamp = extract_slot_time_from_path(Path(_item_name(item)))
        if timestamp is None:
            return None
        return ItemInfo(coordinate=timestamp)

    def _epoch_date(self) -> Any:
        """Return the configured slot-index anchor date (UTC, midnight)."""
        import datetime

        config: MtgFciL1cConfig = self.plugin_config  # type: ignore[assignment]
        return datetime.date.fromisoformat(config.time_epoch)

    def _configured_total_slots(self) -> int | None:
        """Total time-axis length for preallocation / parallel schema sizing.

        Returns ``None`` when neither ``time_slots`` nor ``time_end`` is
        configured (serial mode without a fixed horizon). Invalid values
        raise ``ValueError`` so misconfiguration is never silently ignored.
        """
        import datetime

        config: MtgFciL1cConfig = self.plugin_config  # type: ignore[assignment]
        if config.time_slots is not None:
            if int(config.time_slots) <= 0:
                raise ValueError("time_slots must be a positive integer.")
            return int(config.time_slots)
        if config.time_end is not None:
            end = datetime.date.fromisoformat(config.time_end)
            total = (end - self._epoch_date()).days * REPEAT_CYCLES_PER_DAY
            if total <= 0:
                raise ValueError(
                    f"time_end {config.time_end} is not after time_epoch "
                    f"{config.time_epoch}; nothing to pre-allocate."
                )
            return total
        return None

    def zarr_schema(self, ctx: PluginContext) -> list[ZarrGroupSpec]:
        """Declare the Zarr store layout per resolution.

        The layout itself lives in :mod:`firecube_mtg_fci_l1c._variables` (the
        single place to add variables or attributes). This hook resolves the
        product type and delegates the declarative build there.
        """
        config: MtgFciL1cConfig = self.plugin_config  # type: ignore[assignment]
        # Firecube's tier configurator already parsed and validated this value
        # into DirectZarrIngestor.template_config. Route that typed value into
        # the plugin config used by schema construction; do not parse option
        # strings here.
        template_config = cast(ZarrTemplateConfig | None, self.template_config)
        if "zarr_sharding" in ctx.options and template_config is not None:
            config.template_config.zarr_sharding = template_config.zarr_sharding
        product_type = self._require_product_type()
        if product_type not in CONSTANTS:
            raise ValueError(
                f"Unsupported product type: {product_type!r}. "
                f"Expected one of {sorted(CONSTANTS)}"
            )
        validate_effective_resolutions(config, product_type)
        return build_all_specs(config, product_type)

    def _require_product_type(self) -> str:
        """Return the configured product type; raise when it is not set.

        Core resolves the index and schema from configuration alone, before
        any input is listed, so the product type cannot come from the input.
        """
        config: MtgFciL1cConfig = self.plugin_config  # type: ignore[assignment]
        if config.product_type is None:
            raise ConfigurationError(
                "product_type is required. Pass "
                f"--option product_type={PRODUCT_TYPE_FDHSI} or "
                f"--option product_type={PRODUCT_TYPE_HRFI}."
            )
        return config.product_type

    def discover_source_files(self, ctx: PluginContext) -> Iterable[Any]:
        """Return ZIP items and per-cycle bundles of loose chunk files.

        Core discovery lists the source (local or remote) and applies
        ``--input-filters``. ZIP items are returned unchanged and in core's
        order; chunk files are grouped into one :class:`SceneBundle` per
        repeat cycle. Other files are dropped. ZIPs and chunks in one source,
        or items of the other product type, are configuration errors.
        """
        product_type = self._require_product_type()
        items = list(super().discover_source_files(ctx))
        try:
            zips, chunks, _ignored = classify_items(items)
        except ValueError as exc:
            raise ConfigurationError(
                f"{exc} Keep the chunk files with --input-filters '[\"!*.zip\"]' "
                "or the ZIP files with --input-filters '[\"!*.nc\"]'."
            ) from exc

        foreign = [
            _item_name(item)
            for item in (*zips, *chunks)
            if _item_product_type(item) != product_type
        ]
        if foreign:
            other = (
                PRODUCT_TYPE_HRFI
                if product_type == PRODUCT_TYPE_FDHSI
                else PRODUCT_TYPE_FDHSI
            )
            shown = ", ".join(repr(name) for name in foreign[:3])
            more = f" and {len(foreign) - 3} more" if len(foreign) > 3 else ""
            raise ConfigurationError(
                f"{len(foreign)} input file(s) are not {product_type} products "
                f"(product_type={product_type}): {shown}{more}. Exclude them "
                f"with --input-filters '[\"!*{other}*\"]', or set "
                f"--option product_type to match the input."
            )

        try:
            bundles = group_chunks_into_bundles(chunks)
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from exc
        return [*zips, *bundles]

    def _emit_static_intents(
        self,
        config: MtgFciL1cConfig,
        product_type: str,
        plans: list[GroupPlan],
    ) -> list[WriteIntent]:
        """Iterate VARIABLES with non-time dims; emit static WriteIntents."""
        from ._variables import VARIABLES, VariableContext, variable_enabled

        del product_type
        intents: list[WriteIntent] = []
        for plan in plans:
            ctx = VariableContext(
                group=plan.group,
                resolution=plan.resolution,
                product_type=plan.product_type,
                config=config,
                dimsize=plan.dimsize,
                n_channels=len(plan.logical_channels),
                logical_channels=plan.logical_channels,
                geo_provider=self._geo_provider if config.include_geolocation else None,
            )

            pending: list[tuple[Any, Any]] = []
            with self._static_lock:
                for variable in VARIABLES:
                    if TIME_COORD_NAME in variable.dims:
                        continue
                    if not variable_enabled(variable, config, plan.logical_channels):
                        continue
                    if variable.source is None:
                        continue
                    static_key = f"{plan.group}/{variable.name}"
                    if static_key in self._static_coords_written:
                        continue
                    self._static_coords_written.add(static_key)
                    pending.append((variable, ctx))

            for variable, ctx_ in pending:
                intents.append(
                    WriteIntent(
                        group=ctx_.group,
                        array=variable.name,
                        ts_index=0,
                        data=self._static_payload_loader(variable, ctx_),
                        kind="static",
                    )
                )
        return intents

    def _static_payload_loader(
        self, variable: Variable, ctx: VariableContext
    ) -> Callable[[], np.ndarray]:
        """Return a zero-arg loader for a static variable's payload.

        Static intents carry callables so the engine materialises a payload
        only for intents it actually dispatches; an intent suppressed before
        dispatch never resolves its data.
        """
        static_key = f"{ctx.group}/{variable.name}"
        source = variable.source
        if source is None:  # pragma: no cover - emission filters source=None
            raise ValueError(f"static variable {static_key} declares no source")

        def _load() -> np.ndarray:
            try:
                data = source(ctx)
            except Exception:
                with self._static_lock:
                    self._static_coords_written.discard(static_key)
                raise
            if data is None:
                # Variables without a data payload declare source=None and are
                # filtered at emission; a source returning None here would
                # silently drop a declared array.
                raise RuntimeError(
                    f"static source for {static_key} returned None at dispatch"
                )
            return data

        return _load

    def _emit_time_channel_intents(
        self,
        config: MtgFciL1cConfig,
        product_type: str,
        res: str,
        logical_channels: list[str],
        timestamp: Any,
        calibration_table: dict[str, ChannelCalibration],
        nc_channels: list[str] | None = None,
        slot_geometry: dict[str, float] | None = None,
    ) -> list[IndexedWrite]:
        """Emit slot writes for VARIABLES with dims ('time', 'channel') or ('time',).

        The ``time`` coordinate itself has no data source and is written by
        Firecube from the slot coordinate.
        """
        from ._variables import VARIABLES, VariableContext, variable_enabled

        group = group_name(res, config.flat_store)
        dimsize = dimsize_for(product_type, res)
        ctx = VariableContext(
            group=group,
            resolution=res,
            product_type=product_type,
            config=config,
            dimsize=dimsize,
            n_channels=len(logical_channels),
            logical_channels=tuple(logical_channels),
            calibration_table=calibration_table,
            nc_channels=tuple(nc_channels or ()),
            slot_geometry=slot_geometry,
        )

        intents: list[IndexedWrite] = []
        for variable in VARIABLES:
            if variable.dims not in ((TIME_COORD_NAME, "channel"), (TIME_COORD_NAME,)):
                continue
            if not variable_enabled(variable, config, ctx.logical_channels):
                continue
            if variable.source is None:
                continue
            data = variable.source(ctx)
            if data is None:
                continue
            intents.append(
                IndexedWrite.slot(
                    group=group,
                    array=variable.name,
                    coordinate=timestamp,
                    data=data,
                )
            )
        return intents

    def _emit_spatial_intents(
        self,
        batch_id: str,
        plan: GroupPlan,
        timestamp: Any,
        nc_part_ranges: list[tuple[Path, tuple[int, int]]],
        index2time: dict[int, float] | None,
        pixel_time_dtype: np.dtype,
        config: MtgFciL1cConfig,
    ) -> list[IndexedWrite]:
        """Emit one lazy spatial projection per variable, channel, and chunk."""
        from ._variables import VARIABLES, VariableContext, variable_enabled

        intents: list[IndexedWrite] = []
        chunk_y = config.get_group_chunk_shape(plan.resolution)[1]
        chunk_ranges = _output_chunk_ranges(plan.dimsize, chunk_y)
        for ch_idx, nc_channel in enumerate(plan.nc_channels):
            base_ctx = VariableContext(
                group=plan.group,
                resolution=plan.resolution,
                product_type=plan.product_type,
                config=config,
                dimsize=plan.dimsize,
                n_channels=len(plan.logical_channels),
                logical_channels=plan.logical_channels,
                nc_channels=plan.nc_channels,
            )
            # Probe mirrors `load_channel_slice` semantics: pixel_time is
            # non-None at dispatch iff index2time is truthy. Sources whose
            # probe returns None (e.g. `_pixel_time_source` with no time map)
            # are skipped so Firecube core never receives `data=None`, which
            # would corrupt the region write. This restores the old
            # eager-path behavior of `if data is None: continue`.
            probe_pixel_time: np.ndarray | None = (
                np.empty(0, dtype=pixel_time_dtype) if index2time else None
            )
            probe_ctx = dataclasses.replace(
                base_ctx,
                channel_payload=ChannelSlicePayload(
                    counts=np.empty(0, dtype=np.uint16),
                    pixel_quality=np.empty(0, dtype=np.uint8),
                    pixel_time=probe_pixel_time,
                ),
            )

            variable_sources: list[
                tuple[str, Callable[[VariableContext], np.ndarray | None]]
            ] = []
            for variable in VARIABLES:
                if variable.dims != (TIME_COORD_NAME, "y", "x", "channel"):
                    continue
                if not variable_enabled(variable, config, plan.logical_channels):
                    continue
                if variable.source is None:
                    continue
                if variable.source(probe_ctx) is None:
                    continue
                variable_sources.append((variable.name, variable.source))

            variable_set = frozenset(name for name, _source in variable_sources)
            for y_range in chunk_ranges:
                intersecting = _intersecting_part_ranges(nc_part_ranges, y_range)
                if not intersecting:
                    continue
                _validate_contiguous_part_ranges(intersecting, y_range)
                chunk_parts = [part_path for part_path, _part_range in intersecting]
                part_row_ranges = dict(intersecting)
                y_slice = slice(*y_range)
                ctx = dataclasses.replace(base_ctx, y_slice=y_slice)
                for variable_name, variable_source in variable_sources:
                    intents.append(
                        IndexedWrite.region(
                            group=plan.group,
                            array=variable_name,
                            coordinate=timestamp,
                            data=partial(
                                _assemble_and_extract,
                                batch_id,
                                self._batch_resources,
                                self._batch_resources_lock,
                                chunk_parts,
                                nc_channel,
                                index2time,
                                pixel_time_dtype,
                                plan.group,
                                timestamp,
                                y_range,
                                variable_set,
                                part_row_ranges,
                                variable_source,
                                ctx,
                            ),
                            y_slice=y_slice,
                            channel_index=ch_idx,
                        )
                    )
        return intents

    def build_write_intents(
        self, batch: PipelineBatch, ctx: PluginContext
    ) -> list[WriteIntent | IndexedWrite]:
        """Materialise the batch's scenes and emit write intents per scene.

        A ZIP item is materialised and extracted into per-batch scratch; a
        chunk bundle has each member materialised and is read in place.
        """
        from ._scratch import BatchScratch

        config: MtgFciL1cConfig = self.plugin_config  # type: ignore[assignment]
        product_type = self._require_product_type()
        batch.metadata["product_type"] = product_type

        plans = resolve_group_plans(config, product_type)

        intents: list[WriteIntent | IndexedWrite] = []
        files_processed = 0
        files_failed = 0
        zip_errors: list[str] = []

        run_id = str(ctx.run_id or ctx.option("run_id", "mtg_fci_l1c_run"))
        scratch_dir = config.scratch_dir

        # Per-batch scratch roots keep concurrent batch cleanup isolated.
        scratch_id = f"{run_id}-{batch.batch_id}"

        # Core grows time-indexed arrays from emitted slots; the plugin never
        # opens the store to resize arrays.

        intents.extend(self._emit_static_intents(config, product_type, plans))

        has_zips = any(not isinstance(item, SceneBundle) for item in batch.items)

        # Reader and scratch are registered (not `with`-scoped) so cached
        # file handles stay open until core dispatches the deferred callable
        # payloads; teardown happens in cleanup_batch_data, or at the next
        # on_pipeline_start for batches whose cleanup never ran. Chunk
        # bundles are read in place, so only a batch with ZIPs gets scratch.
        core_scratch = BatchScratch(scratch_dir, scratch_id) if has_zips else None
        shared_reader = SharedNcPartReader()
        chunk_owned_cache = ChunkOwnedAssembler(shared_reader)
        with self._batch_resources_lock:
            resources = self._batch_resources.setdefault(
                batch.batch_id, BatchResources()
            )
            resources.shared_reader = shared_reader
            resources.batch_scratch = core_scratch
            resources.chunk_owned_cache = chunk_owned_cache
            # Registration order is teardown close order: the assembler
            # before the reader it wraps, the scratch last because its
            # close() hands removal off to a daemon thread.
            self._batch_registry.register(batch.batch_id, chunk_owned_cache)
            self._batch_registry.register(batch.batch_id, shared_reader)
            if core_scratch is not None:
                self._batch_registry.register(batch.batch_id, core_scratch)

        # Materialise and extract every ZIP of the batch up front, extraction
        # in parallel; deferred payload dispatch reads nc_parts until batch
        # cleanup, so peak scratch usage is unchanged, only the extraction
        # wall time shrinks. Keys are batch positions: the materialised path
        # of a local item is resolved and can differ from the item's name.
        zip_sources: dict[int, Path] = {}
        zip_failures: dict[int, str] = {}
        for position, item in enumerate(batch.items):
            if isinstance(item, SceneBundle):
                continue
            try:
                zip_sources[position] = Path(ctx.materialize(item))
            except Exception as exc:  # noqa: BLE001 - continue-on-error per scene
                zip_failures[position] = f"cannot materialise: {exc}"
        extracted_dirs: dict[Path, Path] = {}
        extract_failures: dict[Path, str] = {}
        if core_scratch is not None and zip_sources:
            engine_config = getattr(self, "engine_config", None)
            extract_workers = int(getattr(engine_config, "extract_workers", 4) or 4)
            extracted_dirs, extract_failures = core_scratch.extract_zips_parallel(
                list(dict.fromkeys(zip_sources.values())),
                max_workers=extract_workers,
            )

        for position, item in enumerate(batch.items):
            label = item.uri if isinstance(item, SceneBundle) else _item_name(item)
            scene_error: str | None = None
            try:
                if isinstance(item, SceneBundle):
                    # Members are in BODY-then-TRAIL chunk order, the order
                    # list_fci_nc_parts gives the same files from a ZIP;
                    # later parts win on shared root-table rows.
                    self._intents_for_scene(
                        timestamp=item.coordinate,
                        nc_parts=[Path(ctx.materialize(m)) for m in item.members],
                        plans=plans,
                        shared_reader=shared_reader,
                        config=config,
                        product_type=product_type,
                        batch_id=batch.batch_id,
                        intents=intents,
                    )
                else:
                    failure = zip_failures.get(position)
                    if failure is None:
                        failure = extract_failures.get(zip_sources[position])
                    if failure is not None:
                        self._log.warning("Failed to extract %s: %s", item, failure)
                        scene_error = f"{label}: {failure}"
                    else:
                        scene_error = self._intents_for_zip(
                            zip_name=label,
                            zip_dir=extracted_dirs[zip_sources[position]],
                            plans=plans,
                            shared_reader=shared_reader,
                            config=config,
                            product_type=product_type,
                            batch_id=batch.batch_id,
                            intents=intents,
                        )
            except Exception as exc:  # noqa: BLE001 - preserve legacy continue-on-error behavior
                self._log.warning("Failed to process %s: %s", label, exc)
                self._log.exception("nc_part processing failed for %s", label)
                scene_error = f"{label}: {exc}"
            if scene_error is None:
                files_processed += 1
            else:
                zip_errors.append(scene_error)
                files_failed += 1

        batch.metadata["plugin_failure_counters"] = {
            "files_processed": files_processed,
            "files_failed": files_failed,
            "zip_errors": zip_errors,
        }

        return intents

    def _intents_for_zip(
        self,
        *,
        zip_name: str,
        zip_dir: Path,
        plans: list[GroupPlan],
        shared_reader: SharedNcPartReader,
        config: MtgFciL1cConfig,
        product_type: str,
        batch_id: str,
        intents: list[WriteIntent | IndexedWrite],
    ) -> str | None:
        """Decode one extracted ZIP and extend ``intents`` per resolution group.

        The time label is the ZIP's sensing start floored to the minute.
        Returns ``None`` on success, or an error string when the ZIP cannot
        be decoded at all.
        """
        nc_parts = list_fci_nc_parts(zip_dir)
        if not nc_parts:
            return f"No nc_parts found in {zip_name}"

        timestamp = cast(Any, extract_slot_time_from_path(Path(zip_name)))
        if timestamp is None:
            return f"Could not extract timestamp from {zip_name}"

        self._intents_for_scene(
            timestamp=timestamp,
            nc_parts=nc_parts,
            plans=plans,
            shared_reader=shared_reader,
            config=config,
            product_type=product_type,
            batch_id=batch_id,
            intents=intents,
        )
        return None

    def _intents_for_scene(
        self,
        *,
        timestamp: Any,
        nc_parts: list[Path],
        plans: list[GroupPlan],
        shared_reader: SharedNcPartReader,
        config: MtgFciL1cConfig,
        product_type: str,
        batch_id: str,
        intents: list[WriteIntent | IndexedWrite],
    ) -> None:
        """Decode one scene's nc_parts and extend ``intents`` per resolution group.

        ``nc_parts`` are local files in BODY-then-TRAIL chunk order.
        Intents are appended incrementally so a mid-scene failure preserves
        what was already emitted (legacy continue-on-error behavior).
        """
        index2time: dict[int, float] | None = None
        if config.include_pixel_time:
            time_accum = TimeMapAccumulator()
            for part_path in nc_parts:
                if shared_reader.has_time_map(part_path):
                    time_accum.accumulate(shared_reader.reader_for(part_path))
            index2time = time_accum.build_index2time()

        geometry_accum = SlotGeometryAccumulator()
        for part_path in nc_parts:
            geometry_accum.accumulate(shared_reader.reader_for(part_path))
        slot_geometry = geometry_accum.build()
        slot_geometry["sun_earth_distance"] = sun_earth_distance_au(timestamp)

        for plan in plans:
            self._intents_for_plan(
                plan=plan,
                nc_parts=nc_parts,
                timestamp=timestamp,
                index2time=index2time,
                slot_geometry=slot_geometry,
                shared_reader=shared_reader,
                config=config,
                product_type=product_type,
                batch_id=batch_id,
                intents=intents,
            )

    def _intents_for_plan(
        self,
        *,
        plan: GroupPlan,
        nc_parts: list[Path],
        timestamp: Any,
        index2time: dict[int, float] | None,
        shared_reader: SharedNcPartReader,
        config: MtgFciL1cConfig,
        product_type: str,
        batch_id: str,
        intents: list[WriteIntent | IndexedWrite],
        slot_geometry: dict[str, float] | None = None,
    ) -> None:
        """Emit per-channel and spatial intents for one group."""
        res = plan.resolution
        logical_channels = list(plan.logical_channels)
        nc_channels = list(plan.nc_channels)

        row_ranges: dict[int, tuple[int, int]] = {}
        for part_idx, part_path in enumerate(nc_parts):
            try:
                row_ranges[part_idx] = shared_reader.read_row_range(part_path, res)
            except KeyError:
                continue

        if not row_ranges:
            return

        calibration_table: dict[str, ChannelCalibration] = {}
        if config.include_calibration:
            for part_idx, part_path in enumerate(nc_parts):
                if part_idx not in row_ranges:
                    continue
                for ch in plan.nc_channels:
                    if ch in calibration_table:
                        continue
                    cal = shared_reader.decode_channel(part_path, ch)
                    if cal is not None:
                        calibration_table[ch] = cal

        pixel_time_dtype = np.dtype(
            np.float32 if config.pixel_time_dtype == "float32" else np.float64
        )

        intents.extend(
            self._emit_time_channel_intents(
                config,
                product_type,
                res,
                logical_channels,
                timestamp,
                calibration_table,
                nc_channels,
                slot_geometry,
            )
        )
        nc_part_ranges = [
            (nc_parts[part_idx], row_ranges[part_idx])
            for part_idx in sorted(row_ranges)
        ]
        intents.extend(
            self._emit_spatial_intents(
                batch_id,
                plan,
                timestamp,
                nc_part_ranges,
                index2time,
                pixel_time_dtype,
                config,
            )
        )

    def prepare_batch_data(
        self, batch: PipelineBatch, ctx: PluginContext
    ) -> dict[str, Any] | None:
        """Register per-batch resources for lifecycle-managed cleanup."""
        del ctx
        with self._batch_resources_lock:
            self._batch_resources[batch.batch_id] = BatchResources()
        return {"batch_resources_registered": 1}

    def cleanup_batch_data(self, batch: PipelineBatch, ctx: PluginContext) -> None:
        """Clean up per-batch caches, reader handles, and scratch directories."""
        del ctx
        with self._batch_resources_lock:
            # Pop first: dispatch-time payload lookups must fail loudly once
            # teardown starts.
            self._batch_resources.pop(batch.batch_id, None)

        # Close outside the lock. Teardown closes ~40 NetCDF handles per batch;
        # holding `_batch_resources_lock` across that would block every
        # concurrent `prepare_batch_data` and dispatch-time payload lookup for
        # the duration. The registry does its own bookkeeping.
        try:
            self._batch_registry.teardown(batch.batch_id)
        except Exception:  # noqa: BLE001 - cleanup must be best-effort
            self._log.warning(
                "Batch resource teardown raised; continuing", exc_info=True
            )

    def on_pipeline_start(self, ctx: PluginContext, state: PipelineRunState) -> None:
        """Reset run-local guards for a new pipeline invocation."""
        super().on_pipeline_start(ctx, state)

        with self._static_lock:
            self._static_coords_written.clear()
        self._teardown_orphaned_batch_resources()

    def __init__(self, *, name: str | None = None, chunk_manager=None):
        """Initialize geolocation and run-local static-coordinate guards."""
        super().__init__(name=name or self.name, chunk_manager=chunk_manager)
        self._geo_provider = LatLonProvider(self._log)
        # Reset per run; locked because build_write_intents can run concurrently.
        self._static_coords_written: set[str] = set()
        self._static_lock = threading.Lock()
        self._batch_resources: dict[str, BatchResources] = {}
        self._batch_resources_lock = threading.Lock()
        self._batch_registry = BatchResourceRegistry()

    def _teardown_orphaned_batch_resources(self) -> None:
        """Tear down resources of batches whose cleanup never ran (crash paths)."""
        with self._batch_resources_lock:
            self._batch_resources.clear()

        # Closed outside the lock, for the same reason as `cleanup_batch_data`.
        try:
            self._batch_registry.teardown_all()
        except Exception:  # noqa: BLE001 - startup cleanup must not abort the run
            self._log.warning(
                "Orphaned batch resource teardown raised; continuing",
                exc_info=True,
            )

    def slice_meta_keys(self) -> list[str]:
        """Keys that define a logical ingest slice for resume safety."""
        return [
            "resolutions",
            "channels",
            "product_type",
            "include_pixel_quality",
            "include_pixel_time",
            "include_calibration",
            "include_geolocation",
            "pixel_time_dtype",
            "scratch_dir",
            "zarr_chunk_y",
            "time_epoch",
        ]

    def slice_meta(self, ctx: PluginContext) -> dict[str, Any]:
        """Return normalized config for manifest metadata."""
        config: MtgFciL1cConfig = self.plugin_config  # type: ignore[assignment]
        return {
            "resolutions": sorted(config.get_resolutions(config.product_type)),
            "channels": config.channels,
            "product_type": config.product_type,
            "include_pixel_quality": config.include_pixel_quality,
            "include_pixel_time": config.include_pixel_time,
            "include_calibration": config.include_calibration,
            "include_geolocation": config.include_geolocation,
            "pixel_time_dtype": config.pixel_time_dtype,
            "scratch_dir": config.scratch_dir,
            "zarr_chunk_y": config.zarr_chunk_y,
            "time_epoch": config.time_epoch,
        }

    def filter_item(self, item: Any, ctx: PluginContext) -> bool:
        """Keep chunk bundles and file items named like FCI L1C ZIPs."""
        del ctx
        if isinstance(item, SceneBundle):
            return True
        return is_valid_fci_zip(Path(_item_name(item)))

    def item_size_bytes(self, item: Any) -> int | None:
        """Return a ZIP's size, or a bundle's summed member sizes.

        ``None`` when any size is unknown, for example for remote items.
        """
        file_size = super().item_size_bytes
        if not isinstance(item, SceneBundle):
            return file_size(item)
        total = 0
        for member in item.members:
            size = file_size(member)
            if size is None:
                return None
            total += size
        return total

    def get_batch_groups(self, items: Sequence[Any], ctx: PluginContext) -> list[str]:
        """Return resolution write-groups for a batch of FCI items.

        Canonical batch-group hook (``get_batch_groups(items, ctx)``); invoked
        at batch-planning time with the batch's item list. The groups follow
        the configured product type.
        """
        del items, ctx
        config: MtgFciL1cConfig = self.plugin_config  # type: ignore[assignment]
        product_type = self._require_product_type()
        return [p.group for p in resolve_group_plans(config, product_type)]

    def _aggregate_metrics(
        self, ctx: RuntimeIngestContext, state: PipelineRunState
    ) -> dict[str, Any]:
        """Aggregate default metrics plus per-scene success/failure counters.

        A scene is one ZIP or one chunk bundle; the ``zip_errors`` key keeps
        its name for both.
        """
        merged = dict(self.default_aggregate_metrics(ctx, state))

        files_processed = 0
        files_failed = 0
        zip_errors: list[str] = []
        total_zips = 0

        for batch in state.batches:
            metadata = batch.metadata or {}
            counters = metadata.get("plugin_failure_counters")
            if counters:
                files_processed += int(counters.get("files_processed", 0))
                files_failed += int(counters.get("files_failed", 0))
                zip_errors.extend(counters.get("zip_errors", []))
            total_zips += int(metadata.get("plugin_total_zips", len(batch.items)))

        merged["files_processed"] = files_processed
        merged["files_failed"] = files_failed
        merged["zip_errors"] = zip_errors
        merged["count"] = total_zips
        return merged
