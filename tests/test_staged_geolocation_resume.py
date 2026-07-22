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

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_integration import _make_fdhsi_zip_at, _run_ingest

pytest_plugins = ["tests.test_integration"]


@pytest.mark.integration
@pytest.mark.plugin
def test_staged_reingest_with_geolocation_resumes_existing_store(
    tmp_path: Path, small_fci_layout: list[int]
):
    """Staged re-ingest with geolocation succeeds after core strips seed markers."""
    src = tmp_path / "src"
    _make_fdhsi_zip_at(src, "20240101000000")

    staged = {"write_mode": "staged", "include_geolocation": True}
    result1 = _run_ingest(src, tmp_path, options=staged)
    assert result1.exists()

    result2 = _run_ingest(
        src,
        tmp_path,
        options={**staged, "resume_existing": True, "force_reingest": False},
    )
    assert result2.exists()
