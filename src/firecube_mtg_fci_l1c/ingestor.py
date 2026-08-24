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

Each batch extracts ZIPs, discovers BODY/TRAIL nc_parts, optionally builds the
pixel-time lookup table, and emits Firecube ``WriteIntent`` objects. Core owns
the actual Zarr writes and time-axis growth.
"""

# mypy: disable-error-code=import-untyped

from __future__ import annotations

import dataclasses
import logging
import threading
from collections.abc import Callable, Iterable, Sequence
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np  # pyright: ignore[reportMissingImports]

from firecube.ingestor.api import (  # pyright: ignore[reportMissingImports]  # type: ignore[import-untyped]
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
    IndexSpec,
    ItemInfo,
    RegularTimeAxis,
    discover_input_files,
    normalize_epoch_iso,
    resolve_index_spec,
)

from ._constants import (
    CONSTANTS,
    PRODUCT_TYPE_FDHSI,
    REPEAT_CYCLE_MINUTES,
    REPEAT_CYCLES_PER_DAY,
    dimsize_for,
)
from ._data import (
    detect_product_type,
    extract_timestamp_from_path,
    is_valid_fci_zip,
    validate_no_mixed_products,
)
from .geolocation import LatLonProvider
from ._group_plan import GroupPlan, resolve_group_plans
from ._decode import (
    AssemblyPreconditionError,
    ChannelSlicePayload,
    ChunkOwnedAssembler,
    NCPartReader,
    SharedNcPartReader,
    TimeMapAccumulator,
    list_fci_nc_parts,
)
from .config import MtgFciL1cConfig
from ._variables import TIME_COORD_NAME, build_all_specs
from ._scratch import register_cleanup_thread

log = logging.getLogger("firecube.ingestor.mtg_fci_l1c")

if TYPE_CHECKING:
    from ._scratch import BatchScratch


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
    ts_index: int,
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
        ts_index,
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
    """DirectZarr ingestor for MTG FCI L1C ZIP inputs."""

    PRODUCT_NAME: ClassVar[str] = "mtg_fci_l1c"
    name = "mtg_fci_l1c"
    plugin_config_class = MtgFciL1cConfig

    # Keep schema, ingestor, and write-intent time coordinates on the same name.
    time_dim_name: ClassVar[str] = TIME_COORD_NAME

    # Bump this if the declared time-axis definition (epoch, cadence, mode) changes.
    INDEX_MODEL: ClassVar[str] = "eumetsat_repeat_cycle_v1"
    _retained_batch_scratches: list[Any]
    _retained_batch_scratches_lock: threading.Lock

    def index_spec(self, ctx: PluginContext) -> IndexSpec | None:
        """Declare the repeat-cycle index model for each resolution group.

        Returns ``None`` when no fixed extent is configured (serial mode);
        the engine's parallel gate then refuses slot-range flags loudly.
        Invalid ``time_slots`` / ``time_end`` values raise instead of being
        silently swallowed.
        """
        if self._configured_total_slots() is None:
            return None
        return self._build_index_spec(ctx)

    def _build_index_spec(self, ctx: PluginContext) -> IndexSpec:
        """Build the declared time-axis spec; ``slot_count`` may be ``None``.

        Single source of truth for the axis definition. ``index_spec()``
        gates this on a configured extent for the engine's parallel path;
        ``_resolve_declared_index()`` resolves it as-is so serial ingestion
        maps timestamps to slots through the same engine arithmetic.
        """
        config: MtgFciL1cConfig = self.plugin_config  # type: ignore[assignment]
        product_type = self._detect_product_type(ctx)
        # Use get_resolutions() so the IndexSpec matches zarr_schema() exactly.
        resolutions = config.get_resolutions(product_type)
        cadence_s = REPEAT_CYCLE_MINUTES * 60
        epoch_iso = normalize_epoch_iso(f"{config.time_epoch}T00:00:00Z")
        axis = RegularTimeAxis(
            coordinate=TIME_COORD_NAME,
            epoch=epoch_iso,
            cadence_s=cadence_s,
            mode="floor",
            slot_count=self._configured_total_slots(),
        )
        return IndexSpec(
            name=self.INDEX_MODEL,
            groups={f"data_{res}": axis for res in resolutions},
        )

    def _resolve_declared_index(self, ctx: PluginContext):
        """Resolve the declared axis locally, with or without a fixed extent.

        The engine-owned ``resolved_index(ctx)`` requires ``index_spec()`` to
        be non-None, which excludes serial runs without a horizon. This
        helper resolves the same declaration unconditionally so slot
        positions always come from one axis definition.
        """
        return resolve_index_spec(
            self._build_index_spec(ctx), time_dim_name=self.time_dim_name
        )

    def inspect_item(self, item: Any, ctx: PluginContext) -> ItemInfo | None:
        """Extract the timestamp coordinate from an input item path."""
        del ctx
        timestamp = extract_timestamp_from_path(Path(str(item)))
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

        The layout itself lives in :mod:`firecube_mtg_fci_l1c.schema` (the
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
        product_type = self._detect_product_type(ctx)
        if product_type not in CONSTANTS:
            raise ValueError(
                f"Unsupported product type: {product_type!r}. "
                f"Expected one of {sorted(CONSTANTS)}"
            )
        return build_all_specs(config, product_type)

    def _detect_product_type(self, ctx: PluginContext) -> str:
        """Resolve configured or source-inferred FCI product type."""
        config: MtgFciL1cConfig = self.plugin_config  # type: ignore[assignment]
        if config.product_type is not None:
            return config.product_type

        source = Path(ctx.source) if hasattr(ctx, "source") else None
        if source is not None and source.exists():
            try:
                files = sorted(source.glob("*.zip")) if source.is_dir() else [source]
                if files:
                    return validate_no_mixed_products(files)
            except Exception as exc:
                self._log.warning(
                    "Unable to infer product_type from source %s: %s", source, exc
                )

        return PRODUCT_TYPE_FDHSI

    def _emit_static_intents(
        self,
        config: MtgFciL1cConfig,
        product_type: str,
        plans: list[GroupPlan],
    ) -> list[WriteIntent]:
        """Iterate VARIABLES with non-time dims; emit static WriteIntents."""
        from ._variables import VARIABLES, VariableContext, variable_enabled

        if not config.emit_static_variables:
            return []

        del product_type
        intents: list[WriteIntent] = []
        for plan in plans:
            ctx = VariableContext(
                group=plan.group,
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
                    if not variable_enabled(variable, config):
                        continue
                    if variable.source is None:
                        continue
                    static_key = f"{plan.group}/{variable.name}"
                    if static_key in self._static_coords_written:
                        continue
                    self._static_coords_written.add(static_key)
                    pending.append((variable, ctx))

            for variable, ctx_ in pending:
                static_key = f"{ctx_.group}/{variable.name}"
                try:
                    data = variable.source(ctx_)
                except Exception:
                    with self._static_lock:
                        self._static_coords_written.discard(static_key)
                    raise
                if data is None:
                    continue
                intents.append(
                    WriteIntent(
                        group=ctx_.group,
                        array=variable.name,
                        ts_index=0,
                        data=data,
                        kind="static",
                    )
                )
        return intents

    def _emit_timestamp_intents(
        self,
        config: MtgFciL1cConfig,
        product_type: str,
        res: str,
        logical_channels: list[str],
        ts_index: int,
        timestamp: Any,
    ) -> list[WriteIntent]:
        """Iterate VARIABLES with dims==('time',); emit timestamp WriteIntents."""
        from ._variables import VARIABLES, VariableContext, variable_enabled

        group = f"data_{res}"
        dimsize = dimsize_for(product_type, res)
        ctx = VariableContext(
            group=group,
            product_type=product_type,
            config=config,
            dimsize=dimsize,
            n_channels=len(logical_channels),
            logical_channels=tuple(logical_channels),
            timestamp=timestamp,
        )

        intents: list[WriteIntent] = []
        for variable in VARIABLES:
            if variable.dims != (TIME_COORD_NAME,):
                continue
            if not variable_enabled(variable, config):
                continue
            intents.append(
                WriteIntent(
                    group=ctx.group,
                    array=variable.name,
                    ts_index=ts_index,
                    data=None,
                    kind="timestamp",
                    timestamp_val=timestamp,
                )
            )
        return intents

    def _emit_time_channel_intents(
        self,
        config: MtgFciL1cConfig,
        product_type: str,
        res: str,
        logical_channels: list[str],
        ts_index: int,
        calibration_table: dict[str, tuple[float, float]],
        nc_channels: list[str] | None = None,
    ) -> list[WriteIntent]:
        """Iterate VARIABLES with dims==('time','channel'); emit 1d WriteIntents."""
        from ._variables import VARIABLES, VariableContext, variable_enabled

        group = f"data_{res}"
        dimsize = dimsize_for(product_type, res)
        ctx = VariableContext(
            group=group,
            product_type=product_type,
            config=config,
            dimsize=dimsize,
            n_channels=len(logical_channels),
            logical_channels=tuple(logical_channels),
            calibration_table=calibration_table,
            nc_channels=tuple(nc_channels or ()),
        )

        intents: list[WriteIntent] = []
        for variable in VARIABLES:
            if variable.dims != (TIME_COORD_NAME, "channel"):
                continue
            if not variable_enabled(variable, config):
                continue
            if variable.source is None:
                continue
            data = variable.source(ctx)
            if data is None:
                continue
            intents.append(
                WriteIntent(
                    group=group,
                    array=variable.name,
                    ts_index=ts_index,
                    data=data,
                    kind="1d",
                )
            )
        return intents

    def _emit_spatial_intents(
        self,
        batch_id: str,
        plan: GroupPlan,
        ts_index: int,
        nc_part_ranges: list[tuple[Path, tuple[int, int]]],
        index2time: dict[int, float] | None,
        pixel_time_dtype: np.dtype,
        config: MtgFciL1cConfig,
    ) -> list[WriteIntent]:
        """Emit one lazy spatial projection per variable, channel, and chunk."""
        from ._variables import VARIABLES, VariableContext, variable_enabled

        intents: list[WriteIntent] = []
        chunk_y = config.get_group_chunk_shape(plan.group)[1]
        chunk_ranges = _output_chunk_ranges(plan.dimsize, chunk_y)
        for ch_idx, nc_channel in enumerate(plan.nc_channels):
            base_ctx = VariableContext(
                group=plan.group,
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
                if not variable_enabled(variable, config):
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
                        WriteIntent(
                            group=plan.group,
                            array=variable_name,
                            ts_index=ts_index,
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
                                ts_index,
                                y_range,
                                variable_set,
                                part_row_ranges,
                                variable_source,
                                ctx,
                            ),
                            kind="region",
                            y_slice=y_slice,
                            channel_index=ch_idx,
                        )
                    )
        return intents

    def build_write_intents(
        self, batch: PipelineBatch, ctx: PluginContext
    ) -> list[WriteIntent]:
        """Extract ZIP contents and emit write intents via phase emitters."""
        from ._scratch import BatchScratch

        config: MtgFciL1cConfig = self.plugin_config  # type: ignore[assignment]
        product_type = batch.metadata.get("product_type")
        if product_type is None:
            product_type = self._detect_product_type(ctx)
        batch.metadata["product_type"] = product_type

        plans = resolve_group_plans(config, product_type)
        declared_index = self._resolve_declared_index(ctx)

        intents: list[WriteIntent] = []
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

        core_scratch = BatchScratch(scratch_dir, scratch_id)
        with self._retained_batch_scratches_lock:
            self._retained_batch_scratches.append(core_scratch)

        # Reader is retained (not `with`-scoped) so its cached file handles
        # stay open until core dispatches the deferred callable payloads.
        # `_cleanup_retained_batch_scratches` closes it at the next
        # `on_pipeline_start`, symmetric with `BatchScratch` lifetime.
        shared_reader = SharedNcPartReader()
        chunk_owned_cache = ChunkOwnedAssembler(shared_reader)
        with self._retained_batch_scratches_lock:
            self._retained_batch_scratches.append(shared_reader)
        with self._batch_resources_lock:
            resources = self._batch_resources.setdefault(
                batch.batch_id, BatchResources()
            )
            resources.shared_reader = shared_reader
            resources.batch_scratch = core_scratch
            resources.chunk_owned_cache = chunk_owned_cache

        for zip_path in batch.items:
            zip_path = Path(zip_path) if not isinstance(zip_path, Path) else zip_path
            try:
                extract_dir = core_scratch.extract_zip(zip_path)
                nc_parts = list_fci_nc_parts(extract_dir)

                if not nc_parts:
                    zip_errors.append(f"No nc_parts found in {zip_path.name}")
                    files_failed += 1
                    continue

                timestamp = cast(Any, extract_timestamp_from_path(zip_path))
                if timestamp is None:
                    zip_errors.append(
                        f"Could not extract timestamp from {zip_path.name}"
                    )
                    files_failed += 1
                    continue

                index2time: dict[int, float] | None = None
                if config.include_pixel_time:
                    time_accum = TimeMapAccumulator()
                    for part_path in nc_parts:
                        with NCPartReader(part_path) as reader:
                            if reader.has_time_map():
                                time_accum.accumulate(reader)
                    index2time = time_accum.build_index2time()

                for plan in plans:
                    res = plan.resolution
                    group = plan.group
                    logical_channels = list(plan.logical_channels)
                    nc_channels = list(plan.nc_channels)
                    ts_index = declared_index.position(group, timestamp)

                    row_ranges: dict[int, tuple[int, int]] = {}
                    for part_idx, part_path in enumerate(nc_parts):
                        with NCPartReader(part_path) as reader:
                            try:
                                row_ranges[part_idx] = reader.read_row_range(res)
                            except KeyError:
                                continue

                    if not row_ranges:
                        continue

                    calibration_table: dict[str, tuple[float, float]] = {}
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
                        np.float32
                        if config.pixel_time_dtype == "float32"
                        else np.float64
                    )

                    intents.extend(
                        self._emit_timestamp_intents(
                            config,
                            product_type,
                            res,
                            logical_channels,
                            ts_index,
                            timestamp,
                        )
                    )
                    intents.extend(
                        self._emit_time_channel_intents(
                            config,
                            product_type,
                            res,
                            logical_channels,
                            ts_index,
                            calibration_table,
                            nc_channels,
                        )
                    )
                    nc_part_ranges = [
                        (nc_parts[part_idx], row_ranges[part_idx])
                        for part_idx in sorted(row_ranges)
                    ]
                    intents.extend(
                        self._emit_spatial_intents(
                            batch.batch_id,
                            plan,
                            ts_index,
                            nc_part_ranges,
                            index2time,
                            pixel_time_dtype,
                            config,
                        )
                    )

                files_processed += 1
            except Exception as exc:  # noqa: BLE001 - preserve legacy continue-on-error behavior
                self._log.warning("Failed to process %s: %s", zip_path, exc)
                self._log.exception("nc_part processing failed for %s", zip_path)
                zip_errors.append(f"{zip_path.name}: {exc}")
                files_failed += 1

        batch.metadata["plugin_failure_counters"] = {
            "files_processed": files_processed,
            "files_failed": files_failed,
            "zip_errors": zip_errors,
        }

        return intents

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
            resources = self._batch_resources.pop(batch.batch_id, None)

        if resources is None:
            return

        retained_ids = {
            id(item)
            for item in (resources.shared_reader, resources.batch_scratch)
            if item is not None
        }
        if retained_ids:
            with self._retained_batch_scratches_lock:
                self._retained_batch_scratches[:] = [
                    item
                    for item in self._retained_batch_scratches
                    if id(item) not in retained_ids
                ]

        if resources.chunk_owned_cache is not None:
            try:
                resources.chunk_owned_cache.close()
            except Exception:  # noqa: BLE001 - cleanup must be best-effort
                pass

        if resources.shared_reader is not None:
            try:
                resources.shared_reader.close()
            except Exception:  # noqa: BLE001 - cleanup must continue
                self._log.warning(
                    "SharedNcPartReader.close() raised during cleanup; continuing"
                )

        if resources.batch_scratch is not None:
            scratch = resources.batch_scratch
            cleanup_thread = threading.Thread(
                target=scratch.cleanup,
                daemon=True,
                name="batch-scratch-cleanup",
            )
            cleanup_thread.start()
            register_cleanup_thread(cleanup_thread)

    def on_pipeline_start(self, ctx: PluginContext, state: PipelineRunState) -> None:
        """Reset run-local guards for a new pipeline invocation."""
        super().on_pipeline_start(ctx, state)

        with self._static_lock:
            self._static_coords_written.clear()
        self._cleanup_retained_batch_scratches()

    def __init__(self, *, name: str | None = None, chunk_manager=None):
        """Initialize geolocation and run-local static-coordinate guards."""
        super().__init__(name=name or self.name, chunk_manager=chunk_manager)
        self._geo_provider = LatLonProvider(self._log)
        # Reset per run; locked because build_write_intents can run concurrently.
        self._static_coords_written: set[str] = set()
        self._static_lock = threading.Lock()
        self._batch_resources: dict[str, BatchResources] = {}
        self._batch_resources_lock = threading.Lock()
        self._retained_batch_scratches: list[Any] = []
        self._retained_batch_scratches_lock = threading.Lock()

    def _cleanup_retained_batch_scratches(self) -> None:
        """Release scratch dirs and reader handles retained for deferred payloads."""
        while True:
            with self._retained_batch_scratches_lock:
                if not self._retained_batch_scratches:
                    return
                item = self._retained_batch_scratches.pop()
            if isinstance(item, SharedNcPartReader):
                item.close()
            else:
                item.cleanup()

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
            "emit_static_variables",
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
            "emit_static_variables": config.emit_static_variables,
            "pixel_time_dtype": config.pixel_time_dtype,
            "scratch_dir": config.scratch_dir,
            "zarr_chunk_y": config.zarr_chunk_y,
            "time_epoch": config.time_epoch,
        }

    def filter_item(self, item: Any, ctx: PluginContext) -> bool:
        """Filter files to ensure they are valid FCI L1C ZIPs."""
        path = Path(item)
        return is_valid_fci_zip(path)

    def get_batch_groups(self, items: Sequence[Any], ctx: PluginContext) -> list[str]:
        """Return resolution write-groups for a batch of FCI items.

        Canonical batch-group hook (``get_batch_groups(items, ctx)``); invoked
        at batch-planning time with the batch's item list. The product type is
        resolved from config, then inferred from the batch items or source.
        """
        config: MtgFciL1cConfig = self.plugin_config  # type: ignore[assignment]
        product_type = config.product_type
        if product_type is None and items:
            try:
                product_type = detect_product_type(str(items[0]))
            except (ValueError, IndexError):
                product_type = None
        if product_type is None:
            product_type = self._detect_product_type(ctx)
        return [p.group for p in resolve_group_plans(config, product_type)]

    def discover_source_files(self, ctx: PluginContext) -> Iterable[Any]:
        """Discover ZIP source files from the source directory."""
        source_dir = Path(ctx.source) if hasattr(ctx, "source") else None
        if source_dir and source_dir.exists():
            includes = None
            if getattr(self, "engine_config", None) is not None:
                includes = getattr(self.engine_config, "include_patterns", None)
            files = discover_input_files(
                source_dir, preferred_globs=includes, recursive=True
            )
            return files
        return iter(())

    def _aggregate_metrics(
        self, ctx: RuntimeIngestContext, state: PipelineRunState
    ) -> dict[str, Any]:
        """Aggregate default metrics plus per-ZIP success/failure counters."""
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
