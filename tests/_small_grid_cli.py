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

"""A ``firecube`` executable that writes real, tiny stores.

``install_firecube_shim`` puts a ``firecube`` executable in a directory. It
shrinks ``CONSTANTS`` to the 4x4 grid from ``_small_grid`` and then runs the
installed ``firecube`` console script, so a subprocess writes real, tiny
stores. A ``python3`` wrapper next to it runs the test interpreter, as a
``python3`` next to a real ``firecube`` would.
"""

from __future__ import annotations

import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent

FIRECUBE_SHIM = '''"""Test shim: real Firecube CLI with a 4x4 FCI layout."""
import os
import sys
from importlib.metadata import entry_points

sys.path.insert(0, @TESTS_DIR@)

from _small_grid import SMALL_CONSTANTS, fake_compute_latlon

from firecube_mtg_fci_l1c import _constants as const_mod
from firecube_mtg_fci_l1c.geolocation import provider as geolocation_mod

if os.environ.get("SHIM_FAIL_ZARR_SLOTS") == "1" and sys.argv[1:3] == ["zarr", "slots"]:
    print("zarr slots disabled by test shim", file=sys.stderr)
    sys.exit(2)

const_mod.CONSTANTS.update(SMALL_CONSTANTS)
geolocation_mod.compute_latlon = fake_compute_latlon

(firecube_script,) = [
    ep for ep in entry_points(group="console_scripts") if ep.name == "firecube"
]
firecube_script.load()()
'''


def install_firecube_shim(bin_dir: Path) -> Path:
    """Write the ``firecube`` shim and a ``python3`` wrapper into *bin_dir*."""
    shim = bin_dir / "firecube"
    body = FIRECUBE_SHIM.replace("@TESTS_DIR@", repr(str(TESTS_DIR)))
    shim.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
    shim.chmod(0o755)
    python3 = bin_dir / "python3"
    python3.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
    python3.chmod(0o755)
    return shim
