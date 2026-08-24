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

"""Operator-facing configuration for the MTG FCI L1C ingestor."""

from __future__ import annotations

from dataclasses import dataclass, field

from firecube.ingestor.api import (  # pyright: ignore[reportMissingImports]
    PluginConfig as BasePluginConfig,
    ZarrTemplateConfig,
)

from ._constants import (
    CHUNK_DEFAULTS_BY_RESOLUTION,
    CONSTANTS,
    FCI_DATA_EPOCH,
    PRODUCT_TYPE_FDHSI,
    VALID_RESOLUTIONS,
    logical_channel_resolution_map,
)

# Maximum zarr_chunk_y per resolution for chunk-owned assembly (2 × nc_part row count).
# Values exceeding this would require assembling from > 2 nc_parts per output chunk,
# which violates the bounded-cache invariant. Config parse rejects them loudly.
MAX_CHUNK_Y_PER_RESOLUTION: dict[str, int] = {
    "500m": 1112,  # 2 × 556
    "1km": 556,  # 2 × 278
    "2km": 278,  # 2 × 139
}

# Every ``data_<res>`` group that could be sharded, across all product types.
_VALID_SHARD_GROUPS: frozenset[str] = frozenset(
    f"data_{res}" for resolutions in VALID_RESOLUTIONS.values() for res in resolutions
)

_VALID_PIXEL_TIME_DTYPES: frozenset[str] = frozenset(
    {"float64", "float32", "int32", "int64"}
)

_VALID_PROJECTION_UNITS: frozenset[str] = frozenset({"meter", "metre", "radian"})


def _default_template_config() -> ZarrTemplateConfig:
    return ZarrTemplateConfig(zarr_sharding=True)


@dataclass
class MtgFciL1cConfig(BasePluginConfig):
    """Fields parsed from Firecube ``--option`` values."""

    # Resolution selection (process both by default)
    resolutions: str | None = None  # e.g. "500m", "1km", "2km", "1km,2km"
    channels: str | None = None
    product_type: str | None = None

    # Output control
    include_pixel_quality: bool = True
    include_pixel_time: bool = True
    include_calibration: bool = True
    include_geolocation: bool = True
    emit_static_variables: bool = True
    """Emit static coordinate arrays (latitude, longitude, x, y, spatial_ref).

    Set ``False`` on non-owner pods in a parallel fan-out to avoid redundant
    static writes. The production helper (``scripts/fci-ingest.sh``) sets this
    automatically; operators should not need to set it manually.
    """
    fci_grids_file: str | None = None
    """Path to pre-generated .npz grids file.

    If set, lat/lon are loaded from this file instead of being computed.
    Use `firecube plugins mtg_fci_l1c geo generate` to create it.
    """

    # Streaming mode configuration
    pixel_time_dtype: str = "float64"
    """Pixel time output dtype.

    Valid values are ``float64``, ``float32``, ``int32``, and ``int64``.
    ``int32`` overflows around year 2068 for seconds since 2000-01-01
    (``2147483647`` seconds ≈ 68 years), while ``int64`` is effectively
    overflow-free for any practical timeframe.
    """

    projection_units: str = "meter"
    """Projection coordinate units for the x/y axes.

    ``meter`` (default) uses metres with ``standard_name=projection_x_coordinate``
    and ``units=m``. ``metre`` is an alias for ``meter`` with identical
    behaviour. ``radian`` uses angular coordinates with
    ``standard_name=projection_x_angular_coordinate`` and ``units=radian``.
    Changing this option between ingests to the same store raises
    ``SchemaDriftError``.
    """

    scratch_dir: str | None = None
    """Base directory for scratch extraction. Uses system temp if None."""

    zarr_chunk_y: int | None = None
    """Y-dimension chunk size for Zarr arrays. If None, uses nc_part-aligned defaults."""

    zarr_shard_target_bytes: int = 128 * 1024 * 1024
    """Target uncompressed bytes per shard for the default shard policy.

    The default groups whole chunks along the y axis up to this budget. It never
    splits a chunk; if one chunk already exceeds the target, the shard is one chunk.
    """

    zarr_shard_overrides: dict[str, tuple[int, int, int, int]] | None = None
    """Explicit per-group shard shapes for the 4-D data arrays.

    Keyed by ``data_<res>`` group name; each value is a rank-4
    ``(time, y, x, channel)`` shard. Applies only to time-indexed 4-D data
    arrays in that group (not the static lat/lon coords). Validated at schema
    build time to be a whole multiple of each array's chunk shape.
    """

    zarr_chunk_overrides: dict[str, tuple[int, int, int, int]] | None = None
    """Explicit per-group chunk shapes for the 4-D data arrays.

    Keyed by ``data_<res>`` group name; each value is a rank-4
    ``(time, y, x, channel)`` chunk shape. Applies only to time-indexed
    4-D data arrays in that group (``counts``, ``pixel_quality``,
    ``pixel_time``). Takes precedence over ``zarr_chunk_y`` and the
    nc_part-aligned defaults.

    Validation rules (two phases):
    - Phase 1 (here, ``__post_init__``): rank-4, positive dims, time==1, channel==1,
      group in ``_VALID_SHARD_GROUPS``. Run at config construction time.
    - Phase 2 (``get_group_chunk_shape``): y ≤ dimsize, x ≤ dimsize, and
      cross-check against ``zarr_shard_overrides`` for divisibility. Run at schema
      build time when dimsize is known.

    Example (full-disk shard with 4 chunks per shard along Y; X stays full row):
        zarr_chunk_overrides={"data_1km": (1, 2784, 11136, 1)}
        zarr_shard_overrides={"data_1km": (1, 11136, 11136, 1)}
        # 11136 / 2784 = 4 chunks along Y, 11136 / 11136 = 1 along X.

    Trade-off: chunks larger than the nc_part row count (default 278 for
    1km) cause read-modify-write during streaming ingest. Cheap in
    ``--write-mode staged`` (local disk staging), expensive in
    ``--write-mode direct`` to S3. See docs/performance-tuning.md.
    """

    # Deterministic time-slot indexing (EUMETSAT repeat-cycle model).
    time_epoch: str = FCI_DATA_EPOCH
    """Anchor date (ISO ``YYYY-MM-DD``, UTC) for slot index 0.

    Slot index is ``(date - time_epoch).days * 144 + hour*6 + minute//10``: a
    pure function of the acquisition time (EUMETSAT's repeat-cycle scheme),
    identical for FDHSI and HRFI. This makes ingestion idempotent and
    parallel-safe: any worker, in any process, derives the same slot for the
    same acquisition without reading the store. The default anchors day 0 at the
    dataset's first available date; set it to your data's start for compact
    indices on a one-off conversion.
    """

    time_end: str | None = None
    """Exclusive upper-bound date (ISO ``YYYY-MM-DD``) for the planned time axis.

    Together with ``time_epoch`` this fixes the total slot count
    (``(time_end - time_epoch).days * 144``) used by ``firecube zarr preallocate``
    and slot-range parallel ingestion. Required for those paths (and overridden
    by ``time_slots`` when set); ignored for single-process ingest, which sizes
    the time axis to the data it actually writes.
    """

    time_slots: int | None = None
    """Explicit total slot count for the planned time axis.

    Takes precedence over ``time_end``. Use when you want to declare the axis
    length directly rather than derive it from an end date.
    """

    _template_config: ZarrTemplateConfig = field(
        default_factory=_default_template_config,
        init=False,
        repr=False,
        compare=False,
    )

    @property
    def template_config(self) -> ZarrTemplateConfig:
        return self._template_config

    @template_config.setter
    def template_config(self, value: ZarrTemplateConfig) -> None:
        self._template_config = value

    def __post_init__(self) -> None:
        """Validate sharding config: positive target, and override shape sanity.

        The shard target must be positive (a zero/negative budget produces
        nonsensical chunk/shard math). Override divisibility against the per-array
        chunk shape is checked later, at schema build time, where the chunk shape
        for each group is known.
        """
        if self.zarr_shard_target_bytes <= 0:
            raise ValueError(
                "zarr_shard_target_bytes must be a positive integer, got "
                f"{self.zarr_shard_target_bytes!r}"
            )

        if getattr(self, "batch_workers", None) is not None:
            raise ValueError(
                "The `batch_workers` plugin option was removed; DirectZarr streaming uses "
                "pipeline-level parallelism via firecube's --batch-workers / --pipeline-workers. "
                "Please remove this option from your config."
            )

        if self.pixel_time_dtype not in _VALID_PIXEL_TIME_DTYPES:
            raise ValueError(
                f"pixel_time_dtype must be one of {sorted(_VALID_PIXEL_TIME_DTYPES)!r}, "
                f"got {self.pixel_time_dtype!r}"
            )

        if self.projection_units not in _VALID_PROJECTION_UNITS:
            raise ValueError(
                f"projection_units must be one of {sorted(_VALID_PROJECTION_UNITS)!r}, "
                f"got {self.projection_units!r}"
            )

        if self.zarr_chunk_y is not None:
            if self.product_type is not None or self.resolutions is not None:
                selected = frozenset(self.get_resolutions(self.product_type))
                resolution_maxima = [
                    (res, max_val)
                    for res, max_val in MAX_CHUNK_Y_PER_RESOLUTION.items()
                    if res in selected
                ]
            else:
                resolution_maxima = [("1km", MAX_CHUNK_Y_PER_RESOLUTION["1km"])]
            for res, max_val in resolution_maxima:
                if (
                    self.zarr_chunk_overrides is not None
                    and f"data_{res}" in self.zarr_chunk_overrides
                ):
                    continue
                if self.zarr_chunk_y > max_val:
                    raise ValueError(
                        f"zarr_chunk_y={self.zarr_chunk_y} exceeds max supported for "
                        f"chunk-owned assembly at resolution {res}; use zarr_chunk_y <= {max_val} "
                        "or file an issue."
                    )

        if self.zarr_shard_overrides is not None:
            for group, shard_shape in self.zarr_shard_overrides.items():
                if group not in _VALID_SHARD_GROUPS:
                    raise ValueError(
                        f"Invalid zarr_shard_overrides group: {group!r}. "
                        f"Expected one of {sorted(_VALID_SHARD_GROUPS)}"
                    )
                if len(shard_shape) != 4:
                    raise ValueError(
                        f"zarr_shard_overrides[{group!r}] must be rank-4 "
                        f"(time, y, x, channel), got {shard_shape!r}"
                    )
                if any(dimension <= 0 for dimension in shard_shape):
                    raise ValueError(
                        f"zarr_shard_overrides[{group!r}] must contain positive ints, "
                        f"got {shard_shape!r}"
                    )

        if self.zarr_chunk_overrides is not None:
            for group, chunk_shape in self.zarr_chunk_overrides.items():
                if group not in _VALID_SHARD_GROUPS:
                    raise ValueError(
                        f"Invalid zarr_chunk_overrides group: {group!r}. "
                        f"Expected one of {sorted(_VALID_SHARD_GROUPS)}"
                    )
                if len(chunk_shape) != 4:
                    raise ValueError(
                        f"zarr_chunk_overrides[{group!r}] must be rank-4 "
                        f"(time, y, x, channel), got {chunk_shape!r}"
                    )
                if any(dim <= 0 for dim in chunk_shape):
                    raise ValueError(
                        f"zarr_chunk_overrides[{group!r}] must contain positive ints, "
                        f"got {chunk_shape!r}"
                    )
                if chunk_shape[0] != 1:
                    raise ValueError(
                        f"zarr_chunk_overrides[{group!r}] time dim must be 1 "
                        f"(FCI streams one time slot per write), got {chunk_shape!r}"
                    )
                if chunk_shape[3] != 1:
                    raise ValueError(
                        f"zarr_chunk_overrides[{group!r}] channel dim must be 1 "
                        f"(FCI writes channels independently), got {chunk_shape!r}"
                    )
                res = group.replace("data_", "")
                max_y = MAX_CHUNK_Y_PER_RESOLUTION.get(res)
                if max_y is not None and chunk_shape[1] > max_y:
                    raise ValueError(
                        f"zarr_chunk_overrides[{group!r}] y={chunk_shape[1]} "
                        "exceeds max supported for chunk-owned assembly at "
                        f"resolution {res}; use y <= {max_y} or file an issue."
                    )

    def get_resolutions(self, product_type: str | None = None) -> list[str]:
        """Return configured resolutions that are valid for the product type."""
        valid = (
            VALID_RESOLUTIONS.get(product_type, ["1km", "2km"])
            if product_type
            else ["1km", "2km"]
        )
        if self.resolutions is None:
            return valid
        requested = [r.strip() for r in self.resolutions.split(",")]
        return [r for r in requested if r in valid]

    def get_channels(
        self, product_type: str | None = None
    ) -> dict[str, list[str]] | None:
        """Map requested channel names to their valid output resolutions."""
        if self.channels is None:
            return None

        resolved_product_type = product_type or self.product_type or PRODUCT_TYPE_FDHSI
        if resolved_product_type not in CONSTANTS:
            raise ValueError(
                f"Unsupported product type: {resolved_product_type!r}. "
                f"Expected one of {sorted(CONSTANTS)}"
            )

        requested = [ch.strip() for ch in self.channels.split(",") if ch.strip()]
        if not requested:
            return None

        valid_resolutions = VALID_RESOLUTIONS.get(resolved_product_type, ["1km", "2km"])
        channel_to_resolution = logical_channel_resolution_map(resolved_product_type)

        matched: dict[str, list[str]] = {}
        for channel in requested:
            res = channel_to_resolution.get(channel)
            found = res in valid_resolutions
            if found and res is not None:
                matched.setdefault(res, [])
                if channel not in matched[res]:
                    matched[res].append(channel)

            if not found:
                consts = CONSTANTS[resolved_product_type]
                valid_channels = sorted(
                    {ch for res in valid_resolutions for ch in consts[res]["channels"]}
                )
                raise ValueError(
                    f"Unknown channel {channel!r} for {resolved_product_type}. "
                    f"Valid: {valid_channels}"
                )

        return matched

    def get_group_chunk_shape(self, group: str) -> tuple[int, ...]:
        """Return streaming-optimized chunk shape for a group.

        Precedence (highest wins):
          1. ``zarr_chunk_overrides[group]`` — explicit per-group rank-4 override
          2. ``zarr_chunk_y`` — Y-axis-only override applied across all groups
          3. Resolution-aware nc_part-aligned default from CHUNK_DEFAULTS_BY_RESOLUTION
        """
        if self.zarr_chunk_overrides is not None and group in self.zarr_chunk_overrides:
            override = self.zarr_chunk_overrides[group]
            # Phase 2 dimsize cross-check (dimsize unknown at __post_init__ time)
            res = group.replace("data_", "")
            if res == "500m":
                dimsize = 22272
            elif res == "1km":
                dimsize = 11136
            elif res == "2km":
                dimsize = 5568
            else:
                raise ValueError(f"Unknown resolution group: {group!r}")
            if override[1] > dimsize:
                raise ValueError(
                    f"zarr_chunk_overrides[{group!r}] y dim {override[1]} "
                    f"exceeds dimsize {dimsize} for {res!r}"
                )
            if override[2] > dimsize:
                raise ValueError(
                    f"zarr_chunk_overrides[{group!r}] x dim {override[2]} "
                    f"exceeds dimsize {dimsize} for {res!r}"
                )
            return tuple(override)
        res = group.replace("data_", "")
        if res == "500m":
            y_chunk = (
                self.zarr_chunk_y
                if self.zarr_chunk_y is not None
                else CHUNK_DEFAULTS_BY_RESOLUTION.get(res, 278)
            )
            return (1, y_chunk, 22272, 1)
        if res == "1km":
            y_chunk = (
                self.zarr_chunk_y
                if self.zarr_chunk_y is not None
                else CHUNK_DEFAULTS_BY_RESOLUTION.get(res, 278)
            )
            return (1, y_chunk, 11136, 1)
        if res == "2km":
            y_chunk = (
                self.zarr_chunk_y
                if self.zarr_chunk_y is not None
                else CHUNK_DEFAULTS_BY_RESOLUTION.get(res, 278)
            )
            return (1, y_chunk, 5568, 1)
        raise ValueError(f"Unknown resolution group: {group}")
