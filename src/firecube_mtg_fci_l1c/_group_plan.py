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

from ._constants import CONSTANTS, PRODUCT_TYPE_FDHSI, VALID_RESOLUTIONS
from .config import MtgFciL1cConfig


@dataclasses.dataclass(frozen=True)
class GroupPlan:
    """Resolved planning object for one (product_type, resolution) combination."""

    product_type: str
    resolution: str
    group: str
    dimsize: int
    logical_channels: tuple[str, ...]
    nc_channels: tuple[str, ...]


def resolve_group_plans(
    config: MtgFciL1cConfig,
    product_type: str | None = None,
) -> list[GroupPlan]:
    """Return one GroupPlan per resolution that survives config filtering.

    Applies both resolution filtering (config.resolutions) and channel
    filtering (config.channels). Replaces the old _resolution_groups()
    fallback-sentinel pattern.
    """
    pt = product_type or config.product_type or PRODUCT_TYPE_FDHSI
    valid = VALID_RESOLUTIONS.get(pt, ["1km", "2km"])
    configured = config.get_resolutions(pt)
    selection = config.get_channels(pt)

    plans: list[GroupPlan] = []
    for res in valid:
        if res not in configured:
            continue
        if pt not in CONSTANTS or res not in CONSTANTS[pt]:
            continue
        info = CONSTANTS[pt][res]
        logical_all = tuple(info["channels"])
        nc_all = tuple(info["nc_channels"])
        logical_to_nc = dict(zip(logical_all, nc_all, strict=True))

        if selection is not None:
            selected_logical = selection.get(res, [])
            if not selected_logical:
                continue
            logical = tuple(selected_logical)
            nc = tuple(logical_to_nc[ch] for ch in logical)
        else:
            logical = logical_all
            nc = nc_all

        plans.append(
            GroupPlan(
                product_type=pt,
                resolution=res,
                group=f"data_{res}",
                dimsize=int(info["dimsize"]),
                logical_channels=logical,
                nc_channels=nc,
            )
        )
    return plans
