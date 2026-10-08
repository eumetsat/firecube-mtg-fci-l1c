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

"""Resolved per-resolution planning object used by every ingestor hook."""

from __future__ import annotations

import dataclasses

from firecube.ingestor.api import (  # pyright: ignore[reportMissingImports]
    ConfigurationError,
)

from ._constants import CONSTANTS, PRODUCT_TYPE_FDHSI, VALID_RESOLUTIONS, stripe_rows
from .config import MtgFciL1cConfig


def group_name(resolution: str, flat: bool) -> str:
    """Return the Zarr group for *resolution*: ``data_<res>``, or ``""`` (root) if flat."""
    return "" if flat else f"data_{resolution}"


def stripe_window(
    config: MtgFciL1cConfig, product_type: str, resolution: str, dimsize: int
) -> tuple[int, int] | None:
    """Return the group's ``(start, stop)`` full-disk rows, or ``None`` for the full disk.

    The rows of BODY chunks ``config.body_chunks`` are widened to whole output
    chunks of this group (its own chunk height), so the stripe store's chunk
    boundaries are the full-disk store's. ``stop`` is capped at ``dimsize``.
    """
    if config.body_chunks is None:
        return None
    first, last = config.body_chunks
    row_start, row_stop = stripe_rows(product_type, resolution, first, last)
    chunk_y = config.get_group_chunk_shape(resolution)[1]
    start = (row_start // chunk_y) * chunk_y
    stop = min(-(-row_stop // chunk_y) * chunk_y, dimsize)
    return start, stop


def stripe_token(config: MtgFciL1cConfig) -> str | None:
    """Return the store-identity token for ``body_chunks``, e.g. ``stripe_c32_40``."""
    if config.body_chunks is None:
        return None
    first, last = config.body_chunks
    return f"stripe_c{first}_{last}"


def validate_effective_resolutions(config: MtgFciL1cConfig, product_type: str) -> None:
    """Raise ``ConfigurationError`` when the selection leaves nothing to write.

    Both layouts need at least one effective resolution; ``flat_store`` needs
    exactly one.
    """
    resolutions = config.effective_resolutions(product_type)
    if not resolutions:
        raise ConfigurationError(
            f"No {product_type} resolution is left after applying "
            f"resolutions={config.resolutions!r} and channels={config.channels!r}. "
            f"Valid resolutions for {product_type}: {VALID_RESOLUTIONS[product_type]}."
        )
    if config.flat_store and len(resolutions) > 1:
        raise ConfigurationError(
            f"flat_store=true requires exactly one effective resolution for "
            f"{product_type}, got {list(resolutions)}. Select one with "
            f"--option resolutions=<res> or a channels selection from a single "
            f"resolution, or drop flat_store."
        )


@dataclasses.dataclass(frozen=True)
class GroupPlan:
    """Resolved planning object for one (product_type, resolution) combination."""

    product_type: str
    resolution: str
    group: str
    dimsize: int
    logical_channels: tuple[str, ...]
    nc_channels: tuple[str, ...]
    # Full-disk rows (start, stop) the group's arrays hold; None is the full disk.
    y_window: tuple[int, int] | None = None

    @property
    def y_start(self) -> int:
        """First full-disk row of the group's y axis."""
        return 0 if self.y_window is None else self.y_window[0]

    @property
    def y_stop(self) -> int:
        """End (exclusive) full-disk row of the group's y axis."""
        return self.dimsize if self.y_window is None else self.y_window[1]

    @property
    def ny(self) -> int:
        """Length of the group's y axis."""
        return self.y_stop - self.y_start


def resolve_group_plans(
    config: MtgFciL1cConfig,
    product_type: str | None = None,
) -> list[GroupPlan]:
    """Return one GroupPlan per effective resolution.

    The resolutions come from ``config.effective_resolutions``, which applies
    both resolution filtering (config.resolutions) and channel filtering
    (config.channels).
    """
    pt = product_type or config.product_type or PRODUCT_TYPE_FDHSI
    selection = config.get_channels(pt)

    plans: list[GroupPlan] = []
    for res in config.effective_resolutions(pt):
        if pt not in CONSTANTS or res not in CONSTANTS[pt]:
            continue
        info = CONSTANTS[pt][res]
        logical_all = tuple(info["channels"])
        nc_all = tuple(info["nc_channels"])
        logical_to_nc = dict(zip(logical_all, nc_all, strict=True))

        if selection is not None:
            logical = tuple(selection[res])
            nc = tuple(logical_to_nc[ch] for ch in logical)
        else:
            logical = logical_all
            nc = nc_all

        dimsize = int(info["dimsize"])
        plans.append(
            GroupPlan(
                product_type=pt,
                resolution=res,
                group=group_name(res, config.flat_store),
                dimsize=dimsize,
                logical_channels=logical,
                nc_channels=nc,
                y_window=stripe_window(config, pt, res, dimsize),
            )
        )
    return plans
