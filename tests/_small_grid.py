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

"""The 4x4 FCI layout used by the Firecube CLI test shim.

Same grid as the ``small_fci_layout`` fixture: two FDHSI and two HRFI
resolutions, each 4x4 pixels, and a fake geolocation with one space pixel
(NaN) at ``[0, 0]`` and 0.0 elsewhere.
"""

from __future__ import annotations

import numpy as np

from firecube_mtg_fci_l1c._constants import PRODUCT_TYPE_FDHSI, PRODUCT_TYPE_HRFI

SMALL_CONSTANTS: dict[str, dict[str, dict[str, object]]] = {
    PRODUCT_TYPE_FDHSI: {
        "1km": {
            "channels": ["vis_04", "vis_06"],
            "dimsize": 4,
            "nc_channels": ["vis_04", "vis_06"],
        },
        "2km": {"channels": ["ir_38"], "dimsize": 4, "nc_channels": ["ir_38"]},
    },
    PRODUCT_TYPE_HRFI: {
        "500m": {"channels": ["vis_06"], "dimsize": 4, "nc_channels": ["vis_06_hr"]},
        "1km": {"channels": ["ir_38"], "dimsize": 4, "nc_channels": ["ir_38_hr"]},
    },
}


def fake_compute_latlon(_resolution_m: int) -> tuple[np.ndarray, np.ndarray]:
    """4x4 lat/lon with a NaN space pixel at ``[0, 0]`` and 0.0 elsewhere."""
    lat = np.zeros((4, 4), dtype=np.float32)
    lon = np.zeros((4, 4), dtype=np.float32)
    lat[0, 0] = lon[0, 0] = np.nan
    return lat, lon
