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

"""Input-data helpers for MTG FCI L1C ZIP products.

This module identifies product type, observation time, and valid source files.
It does not read nc_parts or build arrays; streaming data reads live in
``_streaming.py``.
"""

from __future__ import annotations

import datetime
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ._constants import PRODUCT_TYPE_FDHSI, PRODUCT_TYPE_HRFI


def extract_timestamp_from_path(path: Path) -> datetime.datetime | None:
    """Return the observation-start timestamp from an FCI filename, if present."""
    timestamps = re.findall(r"(\d{14})", path.name)
    if not timestamps:
        return None
    # FCI filenames usually end with observation-start then observation-end.
    ts_str = timestamps[-2] if len(timestamps) >= 2 else timestamps[0]
    try:
        return datetime.datetime.strptime(ts_str, "%Y%m%d%H%M%S")
    except ValueError:
        return None


def detect_product_type(path_or_name: str | Path) -> str:
    """Return ``FDHSI`` or ``HRFI`` from the ZIP filename."""
    name = Path(path_or_name).name
    if PRODUCT_TYPE_FDHSI in name:
        return PRODUCT_TYPE_FDHSI
    if PRODUCT_TYPE_HRFI in name:
        return PRODUCT_TYPE_HRFI
    raise ValueError(
        "Cannot detect product type from filename: "
        f"{name!r}. Expected 'FDHSI' or 'HRFI' in name."
    )


def validate_no_mixed_products(items: Sequence[Any]) -> str:
    """Return the common product type, rejecting mixed FDHSI/HRFI batches."""
    types: set[str] = {detect_product_type(str(item)) for item in items}
    if len(types) > 1:
        raise ValueError(
            "Mixed FDHSI and HRFI products detected in source directory. "
            f"Found: {sorted(types)}. Process one product type at a time."
        )
    if not types:
        return PRODUCT_TYPE_FDHSI
    return types.pop()


def is_valid_fci_zip(path: Path) -> bool:
    """Return True for ZIP filenames that look like FCI L1C RRAD products."""
    if path.suffix != ".zip":
        return False
    if "FCI-1C-RRAD" not in path.name:
        return False
    return extract_timestamp_from_path(path) is not None
