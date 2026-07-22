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

import logging
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np  # pyright: ignore[reportMissingImports]

from firecube.ingestor.api import (  # pyright: ignore[reportMissingImports]  # type: ignore[import-untyped]
    DirectZarrIngestor,
    PipelineBatch,
    PipelineRunState,
    PluginContext,
    RuntimeIngestContext,
    WriteIntent,
    ZarrGroupSpec,
    register_ingestor,
)
from firecube.core.api import (  # pyright: ignore[reportMissingImports]  # type: ignore[import-untyped]
    SlotAxis,
    SlotIndexModel,
    discover_input_files,
    normalize_epoch_iso,
)

from ._constants import (
    CONSTANTS,
    PRODUCT_TYPE_FDHSI,
    REPEAT_CYCLE_MINUTES,
    REPEAT_CYCLES_PER_DAY,
    VALID_RESOLUTIONS,
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
from ._streaming import (
    NCPartReader,
    TimeMapAccumulator,
    list_fci_nc_parts,
)
from .config import MtgFciL1cConfig
from .schema import TIME_COORD_NAME, build_all_specs

log = logging.getLogger("firecube.ingestor.mtg_fci_l1c")


@register_ingestor("mtg_fci_l1c")
class MtgFciL1cIngestor(DirectZarrIngestor):
    """DirectZarr ingestor for MTG FCI L1C ZIP inputs."""

    PRODUCT_NAME: ClassVar[str] = "mtg_fci_l1c"
    name = "mtg_fci_l1c"
    plugin_config_class = MtgFciL1cConfig

    # Keep schema, ingestor, and write-intent time coordinates on the same name.
    time_dim_name: ClassVar[str] = TIME_COORD_NAME

    # Slot indices are deterministic, so independent pods can own disjoint ranges.
    SUPPORTS_SLOT_RANGE_PARALLELISM: ClassVar[bool] = True

    # Bump this if ``timestamp_to_ts_index`` changes.
    INDEX_MODEL: ClassVar[str] = "eumetsat_repeat_cycle_v1"

    def slot_index_model(self, ctx: PluginContext) -> SlotIndexModel:
        """Declare the 10-minute repeat-cycle slot model for every product group."""
        config: MtgFciL1cConfig = self.plugin_config  # type: ignore[assignment]
        product_type = self._detect_product_type(ctx)
        valid = VALID_RESOLUTIONS.get(product_type, VALID_RESOLUTIONS[PRODUCT_TYPE_FDHSI])
        cadence_s = REPEAT_CYCLE_MINUTES * 60
        epoch_iso = normalize_epoch_iso(f"{config.time_epoch}T00:00:00Z")
        return SlotIndexModel(
            name=self.INDEX_MODEL,
            epoch=epoch_iso,
            groups={
                f"data_{res}": SlotAxis(cadence_s=cadence_s, mode="floor") for res in valid
            },
        )

    def _epoch_date(self) -> Any:
        """Return the configured slot-index anchor date (UTC, midnight)."""
        import datetime

        config: MtgFciL1cConfig = self.plugin_config  # type: ignore[assignment]
        return datetime.date.fromisoformat(config.time_epoch)

    @staticmethod
    def _as_datetime(value: Any) -> Any:
        """Coerce a timestamp-like value to a ``datetime.datetime``."""
        import datetime

        if isinstance(value, datetime.datetime):
            return value
        # numpy datetime64 / ISO string / other scalar -> naive datetime
        return np.datetime64(value, "s").tolist()

    def timestamp_to_ts_index(self, group: str, timestamp_val: Any) -> int:
        """Map acquisition time to ``(date - epoch).days * 144 + hour*6 + minute//10``."""
        del group  # all data groups share the same time axis
        t = self._as_datetime(timestamp_val)
        day = (t.date() - self._epoch_date()).days
        if day < 0:
            raise ValueError(
                f"Acquisition {t.isoformat()} precedes time_epoch "
                f"{self._epoch_date().isoformat()}; lower time_epoch to cover it."
            )
        cycle = t.hour * (60 // REPEAT_CYCLE_MINUTES) + t.minute // REPEAT_CYCLE_MINUTES
        return day * REPEAT_CYCLES_PER_DAY + cycle

    def _configured_total_slots(self) -> int:
        """Total time-axis length for preallocation / parallel schema sizing."""
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
        raise ValueError(
            "Slot-range parallelism / preallocate requires the planned time-axis "
            "length: set --option time_end=YYYY-MM-DD or --option time_slots=N."
        )

    def global_expected_time_count(self, ctx: PluginContext) -> dict[str, int]:
        """Declare the full time-axis length per data group for preallocation."""
        config: MtgFciL1cConfig = self.plugin_config  # type: ignore[assignment]
        product_type = self._detect_product_type(ctx)
        total = self._configured_total_slots()
        return {p.group: total for p in resolve_group_plans(config, product_type)}

    def filter_items_to_slot_range(
        self,
        items: Sequence[Any],
        slot_start: int,
        slot_end: int,
        ctx: PluginContext,
    ) -> Sequence[Any]:
        """Keep only ZIPs whose deterministic slot falls in ``[slot_start, slot_end)``."""
        del ctx
        kept: list[Any] = []
        for item in items:
            timestamp = extract_timestamp_from_path(Path(str(item)))
            if timestamp is None:
                continue
            ts_index = self.timestamp_to_ts_index("", timestamp)
            if slot_start <= ts_index < slot_end:
                kept.append(item)
        return kept

    def zarr_schema(self, ctx: PluginContext) -> list[ZarrGroupSpec]:
        """Declare the Zarr store layout per resolution.

        The layout itself lives in :mod:`firecube_mtg_fci_l1c.schema` (the
        single place to add variables or attributes). This hook resolves the
        product type and delegates the declarative build there.
        """
        config: MtgFciL1cConfig = self.plugin_config  # type: ignore[assignment]
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
        from .schema import VARIABLES, VariableContext, variable_enabled

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
        from .schema import VARIABLES, VariableContext, variable_enabled

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
        from .schema import VARIABLES, VariableContext, variable_enabled

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
        plan: GroupPlan,
        ts_index: int,
        reader: Any,
        y_slice: slice,
        index2time: dict[int, float] | None,
        pixel_time_dtype: np.dtype,
        config: MtgFciL1cConfig,
    ) -> list[WriteIntent]:
        """Load each channel payload once, then project all spatial variables."""
        from ._streaming import load_channel_slice
        from .schema import VARIABLES, VariableContext, variable_enabled

        intents: list[WriteIntent] = []
        for ch_idx, nc_channel in enumerate(plan.nc_channels):
            payload = load_channel_slice(reader, nc_channel, index2time, pixel_time_dtype)
            ctx = VariableContext(
                group=plan.group,
                product_type=plan.product_type,
                config=config,
                dimsize=plan.dimsize,
                n_channels=len(plan.logical_channels),
                logical_channels=plan.logical_channels,
                nc_channels=plan.nc_channels,
                y_slice=y_slice,
                channel_payload=payload,
            )
            for variable in VARIABLES:
                if variable.dims != (TIME_COORD_NAME, "y", "x", "channel"):
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
                        group=plan.group,
                        array=variable.name,
                        ts_index=ts_index,
                        data=data,
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

        with BatchScratch(scratch_dir, scratch_id) as core_scratch:
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
                        ts_index = self.timestamp_to_ts_index(group, timestamp)

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
                                with NCPartReader(part_path) as reader:
                                    for ch in plan.nc_channels:
                                        if ch in calibration_table:
                                            continue
                                        cal = reader.read_calibration(ch)
                                        if cal is not None:
                                            calibration_table[ch] = cal

                        pixel_time_dtype = np.dtype(
                            np.float32 if config.pixel_time_dtype == "float32" else np.float64
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
                        for part_idx, part_path in enumerate(nc_parts):
                            if part_idx not in row_ranges:
                                continue

                            start_row, end_row = row_ranges[part_idx]
                            y_slice = slice(start_row, end_row)

                            with NCPartReader(part_path) as reader:
                                intents.extend(
                                    self._emit_spatial_intents(
                                        plan,
                                        ts_index,
                                        reader,
                                        y_slice,
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

    def on_pipeline_start(self, ctx: PluginContext, state: PipelineRunState) -> None:
        """Reset run-local guards for a new pipeline invocation."""
        super().on_pipeline_start(ctx, state)

        with self._static_lock:
            self._static_coords_written.clear()

    def __init__(self, *, name: str | None = None, chunk_manager=None):
        """Initialize geolocation and run-local static-coordinate guards."""
        super().__init__(name=name or self.name, chunk_manager=chunk_manager)
        self._geo_provider = LatLonProvider(self._log)
        # Reset per run; locked because build_write_intents can run concurrently.
        self._static_coords_written: set[str] = set()
        self._static_lock = threading.Lock()

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
