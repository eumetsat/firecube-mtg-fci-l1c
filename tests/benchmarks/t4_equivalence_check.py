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
