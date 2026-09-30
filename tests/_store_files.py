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

"""Byte-level snapshot of a local store, for "store is untouched" assertions."""

from __future__ import annotations

from pathlib import Path


def store_files(store: Path, *, skip_control_plane: bool = True) -> dict[str, bytes]:
    """Every file of *store* with its bytes, by relative path.

    By default the ``.firecube/`` control plane is left out: it records every
    run, including rejected ones, while the Zarr data must stay unchanged.
    """
    return {
        path.relative_to(store).as_posix(): path.read_bytes()
        for path in sorted(store.rglob("*"))
        if path.is_file()
        and not (skip_control_plane and path.relative_to(store).parts[0] == ".firecube")
    }
