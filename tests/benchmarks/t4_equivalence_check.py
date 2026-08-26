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

import json
import os
import sys
import zarr
import numpy as np

ctrl_root = os.environ["T4_CONTROL_ROOT"]
cache_root = os.environ["T4_CACHE_ROOT"]

ctrl = zarr.open(ctrl_root, mode="r")
cache = zarr.open(cache_root, mode="r")

mismatches = []
checked = 0
for group_name in ("data_1km", "data_2km"):
    if group_name not in ctrl or group_name not in cache:
        continue
    ctrl_group = ctrl[group_name]
    cache_group = cache[group_name]
    for var_name in ctrl_group.array_keys():
        checked += 1
        ctrl_arr = ctrl_group[var_name][:]
        cache_arr = cache_group[var_name][:]
        try:
            np.testing.assert_array_equal(ctrl_arr, cache_arr, equal_nan=True)
        except AssertionError as exc:
            mismatches.append(
                {"group": group_name, "var": var_name, "reason": str(exc)[:200]}
            )

report = {
    "checked": checked,
    "mismatches": mismatches,
    "byte_identical": len(mismatches) == 0,
}
print(json.dumps(report, indent=2))
sys.exit(0 if not mismatches else 1)
