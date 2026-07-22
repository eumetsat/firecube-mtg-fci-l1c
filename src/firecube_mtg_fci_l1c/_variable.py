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

"""Infrastructure for the MTG FCI L1C declarative variable schema."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np  # pyright: ignore[reportMissingImports]
from firecube.ingestor.api import (  # pyright: ignore[reportMissingImports]
    ZarrArraySpec,
    ZarrGroupSpec,
)

from ._constants import CONSTANTS, VALID_RESOLUTIONS
from .config import MtgFciL1cConfig


TIME_COORD_NAME = "time"


@dataclasses.dataclass(frozen=True)
class VariableContext:
    """Per-emission context passed to source callables.

    Static-phase emitters populate ``group``, ``product_type``, ``config``,
    ``dimsize``, ``n_channels``, ``logical_channels``. Time-phase emitters add
    ``timestamp``. Spatial-phase emitters add ``y_slice`` and
    ``channel_payload``.
    """

    group: str
    product_type: str
    config: MtgFciL1cConfig
    dimsize: int
    n_channels: int
    logical_channels: tuple[str, ...]
    # Runtime-phase fields (None during static phase):
    y_slice: slice | None = None
    timestamp: Any = None
    geo_provider: Any = None
    calibration_table: dict[str, tuple[float, float]] | None = None
    channel_payload: Any = None  # ChannelSlicePayload; Any avoids circular import
    nc_channels: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class Variable:
    """One Zarr-array declaration.

    ``dims`` determines shape, chunking, time-indexed-ness, and the
    ``WriteIntent.kind`` used at write time. ``source(ctx)`` is the only way to
    provide data; it MUST be a module-level function (no lambdas) so that
    VARIABLES stays picklable for ProcessPoolExecutor workers.
    """

    name: str
    dims: tuple[str, ...]
    dtype: Any
    fill_value: Any
    attrs: Mapping[str, Any] | None = None
    source: Callable[[VariableContext], "np.ndarray | None"] | None = None
    enabled_by: str | None = None
    # Must be a module-level callable (no lambdas) if used from process workers.
    attrs_resolver: Callable[[MtgFciL1cConfig], Mapping[str, Any]] | None = None


def variable_enabled(variable: Variable, config: MtgFciL1cConfig) -> bool:
    """Return True if *variable* is enabled under *config*."""
    if variable.enabled_by is None:
        return True
    return bool(getattr(config, variable.enabled_by, True))


_RESERVED_ATTR_KEYS: frozenset[str] = frozenset(
    {
        "_ARRAY_DIMENSIONS",
        "_FillValue",
        "firecube_run_id",
        "firecube_span_id",
        "firecube_internal",
    }
)


def _copy_attrs(attrs: Mapping[str, Any] | None) -> dict[str, Any] | None:
    return dict(attrs) if attrs is not None else None


def _group_attrs(group: str) -> dict[str, str]:
    """Global attributes for a ``data_<res>`` group."""
    res = group.removeprefix("data_")
    return {
        "Conventions": "CF-1.8",
        "title": f"MTG FCI Level 1C effective radiances ({res})",
        "institution": "EUMETSAT",
        "source": "Meteosat Third Generation Flexible Combined Imager (FCI) Level 1C",
        "history": "Ingested to Zarr by firecube-mtg-fci-l1c",
    }


def _byte_budgeted_4d_shard(
    chunks: tuple[int, int, int, int],
    dtype: Any,
    *,
    dimsize: int,
    target_bytes: int,
) -> tuple[int, int, int, int]:
    """Return a chunk-aligned 4-D shard near the byte target, growing only y."""
    itemsize = np.dtype(dtype).itemsize
    chunk_y = chunks[1]
    chunk_bytes = math.prod(chunks) * itemsize
    if chunk_bytes >= target_bytes:
        return chunks
    multiples = max(1, target_bytes // chunk_bytes)
    cap = max(1, dimsize // chunk_y)
    shard_y = max(chunk_y, min(multiples, cap) * chunk_y)
    return (chunks[0], shard_y, chunks[2], chunks[3])


def _validate_shard_override(
    shards: tuple[int, ...],
    chunks: tuple[int, ...],
    *,
    group: str,
    name: str,
) -> None:
    """Raise unless *shards* is a whole multiple of *chunks* on every axis."""
    if len(shards) != len(chunks):
        raise ValueError(
            f"zarr_shard_overrides[{group!r}] rank {len(shards)} != chunk rank "
            f"{len(chunks)} for array {name!r}"
        )
    for axis, (shard_dim, chunk_dim) in enumerate(zip(shards, chunks, strict=True)):
        if chunk_dim <= 0 or shard_dim % chunk_dim != 0:
            raise ValueError(
                f"zarr_shard_overrides[{group!r}] shard dim {shard_dim} (axis {axis}) "
                f"is not a whole multiple of chunk dim {chunk_dim} for array {name!r}. "
                f"Hint: chunk shape may be from zarr_chunk_overrides or zarr_chunk_y; "
                f"ensure shard % chunk == 0 on every axis."
            )


def _static_2d_chunks(dimsize: int, dtype: Any, target_bytes: int) -> tuple[int, int]:
    """Return full-width row chunks for a large static ``(y, x)`` grid."""
    itemsize = np.dtype(dtype).itemsize
    row_bytes = dimsize * itemsize
    tile_y = max(1, min(dimsize, target_bytes // max(1, row_bytes)))
    return (tile_y, dimsize)


def _variable_dtype(variable: Variable, config: MtgFciL1cConfig, time_coord_name: str) -> Any:
    if variable.name == "pixel_time":
        dtype_map = {
            "float32": np.float32,
            "float64": np.float64,
            "int32": np.int32,
            "int64": np.int64,
        }
        return dtype_map.get(config.pixel_time_dtype, np.float64)
    if variable.name == time_coord_name:
        return np.dtype("datetime64[s]")
    return variable.dtype


def _effective_fill_value(variable: Variable, dtype: Any) -> Any:
    """Return appropriate fill_value for the given resolved dtype."""
    if variable.name == "pixel_time" and np.issubdtype(np.dtype(dtype), np.integer):
        return np.iinfo(np.dtype(dtype)).max
    return variable.fill_value


def _build_array_spec(
    variable: Variable, ctx: VariableContext, time_coord_name: str
) -> ZarrArraySpec:
    """Build a ZarrArraySpec from one flat Variable declaration."""
    dtype = _variable_dtype(variable, ctx.config, time_coord_name)
    dims = variable.dims
    attrs = _copy_attrs(variable.attrs)
    if variable.attrs_resolver is not None:
        resolved = variable.attrs_resolver(ctx.config)
        if resolved:
            if attrs is None:
                attrs = dict(resolved)
            else:
                attrs.update(resolved)
    fill_value = _effective_fill_value(variable, dtype)

    if dims == (time_coord_name, "y", "x", "channel"):
        chunks = ctx.config.get_streaming_chunk_shape(ctx.group)
        if len(chunks) != 4:
            raise ValueError(f"Expected rank-4 chunks for {ctx.group}, got {chunks!r}")
        chunks4 = (chunks[0], chunks[1], chunks[2], chunks[3])
        shard_override = None
        if ctx.config.zarr_shard_overrides is not None:
            shard_override = ctx.config.zarr_shard_overrides.get(ctx.group)
        if not ctx.config.zarr_sharding:
            shards: tuple[int, ...] | None = None
        elif shard_override is not None:
            _validate_shard_override(shard_override, chunks4, group=ctx.group, name=variable.name)
            shards = shard_override
        else:
            shards = _byte_budgeted_4d_shard(
                chunks4,
                dtype,
                dimsize=ctx.dimsize,
                target_bytes=ctx.config.zarr_shard_target_bytes,
            )
        return ZarrArraySpec(
            name=variable.name,
            shape=(1, ctx.dimsize, ctx.dimsize, ctx.n_channels),
            dtype=dtype,
            chunks=chunks4,
            fill_value=fill_value,
            shards=shards,
            dimension_names=dims,
            attrs=attrs,
        )

    if dims == (time_coord_name, "channel"):
        return ZarrArraySpec(
            name=variable.name,
            shape=(1, ctx.n_channels),
            dtype=dtype,
            chunks=(1, ctx.n_channels),
            fill_value=fill_value,
            dimension_names=dims,
            attrs=attrs,
        )

    if dims == (time_coord_name,):
        return ZarrArraySpec(
            name=variable.name,
            shape=(1,),
            dtype=dtype,
            chunks=(1,),
            fill_value=fill_value,
            dimension_names=dims,
            attrs=attrs,
        )

    if dims == ("y", "x"):
        return ZarrArraySpec(
            name=variable.name,
            shape=(ctx.dimsize, ctx.dimsize),
            dtype=dtype,
            chunks=_static_2d_chunks(ctx.dimsize, dtype, ctx.config.zarr_shard_target_bytes),
            fill_value=fill_value,
            shards=None,
            time_indexed=False,
            dimension_names=dims,
            attrs=attrs,
        )

    if dims == ("channel",):
        return ZarrArraySpec(
            name=variable.name,
            shape=(ctx.n_channels,),
            dtype=dtype,
            chunks=(ctx.n_channels,),
            fill_value=fill_value,
            time_indexed=False,
            dimension_names=dims,
            attrs=attrs,
        )

    if dims in {("x",), ("y",)}:
        return ZarrArraySpec(
            name=variable.name,
            shape=(ctx.dimsize,),
            dtype=dtype,
            chunks=(ctx.dimsize,),
            # ZarrArraySpec accepts None, preserving coord vars without _FillValue.
            fill_value=fill_value,
            time_indexed=False,
            dimension_names=dims,
            attrs=attrs,
        )

    if dims == ():
        return ZarrArraySpec(
            name=variable.name,
            shape=(),
            dtype=dtype,
            chunks=None,
            fill_value=fill_value,
            time_indexed=False,
            dimension_names=(),
            attrs=attrs,
        )

    raise ValueError(f"Unsupported dims for {variable.name!r}: {dims!r}")


def _coord_names_for(variables: list[Variable], time_coord_name: str) -> frozenset[str]:
    names: set[str] = set()
    for variable in variables:
        names.update(dim for dim in variable.dims if dim != time_coord_name)
        if time_coord_name not in variable.dims and variable.dims:
            names.add(variable.name)
    return frozenset(names)


def build_specs(config: MtgFciL1cConfig, product_type: str) -> list[ZarrGroupSpec]:
    """Project VARIABLES through *config* and return per-group ZarrGroupSpec list."""
    from .schema import TIME_COORD_NAME, VARIABLES

    if product_type not in CONSTANTS:
        raise ValueError(
            f"Unsupported product type: {product_type!r}. Expected one of {sorted(CONSTANTS)}"
        )

    valid = VALID_RESOLUTIONS.get(product_type, VALID_RESOLUTIONS[next(iter(CONSTANTS))])
    configured = config.get_resolutions(product_type)
    resolutions = [res for res in valid if res in configured]

    channel_selection = config.get_channels(product_type)
    if channel_selection is not None:
        resolutions = [res for res in resolutions if channel_selection.get(res)]

    group_specs: list[ZarrGroupSpec] = []
    for resolution in resolutions:
        group = f"data_{resolution}"
        res_info = CONSTANTS[product_type][resolution]
        dimsize = int(res_info["dimsize"])  # type: ignore[call-overload]
        logical_channels: tuple[str, ...] = tuple(res_info["channels"])  # type: ignore[arg-type]
        if channel_selection is not None:
            logical_channels = tuple(channel_selection[resolution])

        ctx = VariableContext(
            group=group,
            product_type=product_type,
            config=config,
            dimsize=dimsize,
            n_channels=len(logical_channels),
            logical_channels=logical_channels,
        )
        enabled_variables = [variable for variable in VARIABLES if variable_enabled(variable, config)]
        group_specs.append(
            ZarrGroupSpec(
                group=group,
                arrays=[
                    _build_array_spec(variable, ctx, TIME_COORD_NAME)
                    for variable in enabled_variables
                ],
                coord_names=_coord_names_for(enabled_variables, TIME_COORD_NAME),
                attrs=_group_attrs(group),
            )
        )
    return group_specs
