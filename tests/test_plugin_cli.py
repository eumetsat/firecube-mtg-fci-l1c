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

"""Tests for plugin_cli geo commands."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from firecube_mtg_fci_l1c.plugin_cli import cli


@pytest.mark.unit
def test_geo_generate_creates_npz(tmp_path: Path) -> None:
    output = tmp_path / "grids.npz"
    runner = CliRunner()
    result = runner.invoke(
        cli, ["geo", "generate", "--resolutions", "2km", "--output", str(output)]
    )
    assert result.exit_code == 0, result.output
    assert output.exists()
    assert output.stat().st_size > 0


@pytest.mark.unit
def test_geo_generate_overwrite_flag(tmp_path: Path) -> None:
    output = tmp_path / "grids.npz"
    runner = CliRunner()
    runner.invoke(
        cli, ["geo", "generate", "--resolutions", "2km", "--output", str(output)]
    )
    result = runner.invoke(
        cli, ["geo", "generate", "--resolutions", "2km", "--output", str(output)]
    )
    assert result.exit_code != 0
    assert "overwrite" in result.output.lower() or "exists" in result.output.lower()

    result = runner.invoke(
        cli,
        [
            "geo",
            "generate",
            "--resolutions",
            "2km",
            "--output",
            str(output),
            "--overwrite",
        ],
    )
    assert result.exit_code == 0


@pytest.mark.unit
def test_geo_generate_invalid_resolution(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "geo",
            "generate",
            "--resolutions",
            "3km",
            "--output",
            str(tmp_path / "x.npz"),
        ],
    )
    assert result.exit_code != 0
    assert "Unknown" in result.output or "3km" in result.output


@pytest.mark.unit
def test_geo_info(tmp_path: Path) -> None:
    output = tmp_path / "grids.npz"
    runner = CliRunner()
    runner.invoke(
        cli, ["geo", "generate", "--resolutions", "2km", "--output", str(output)]
    )
    result = runner.invoke(cli, ["geo", "info", "--grids-file", str(output)])
    assert result.exit_code == 0, result.output
    assert "2km" in result.output
