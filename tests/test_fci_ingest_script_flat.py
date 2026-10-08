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

"""End-to-end runs of ``scripts/fci-ingest.sh`` for flat and grouped stores.

Each test runs the whole script with ``bash``: env parsing, preallocate, the
fan-out plan, the slot-range ingest pods and the drift check. ``FIRECUBE``
points at the 4x4-layout shim from ``_small_grid_cli``, so preallocate and
ingest write real, tiny stores. The script sends stderr into its stdout log,
so assertions read stdout.

The option tests use a recorder in place of ``firecube``: it appends every
call's arguments to a file and fails ``zarr slots`` so the script takes its
local fan-out plan. They assert on the recorded command lines.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import zarr

from _small_grid_cli import install_firecube_shim
from tests._support import _make_fdhsi_zip_at

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "fci-ingest.sh"

pytestmark = [pytest.mark.integration, pytest.mark.plugin]


@pytest.fixture(scope="module")
def input_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("input")
    _make_fdhsi_zip_at(path, "20240101000000")
    return path


@pytest.fixture(scope="module")
def firecube_shim(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return install_firecube_shim(tmp_path_factory.mktemp("bin"))


def _env(
    tmp_path: Path, input_dir: Path, firecube: Path, **overrides: str
) -> dict[str, str]:
    env = os.environ.copy()
    for name in (
        "FLAT_STORE",
        "EXTRA_OPTIONS",
        "RESOLUTIONS",
        "SHIM_FAIL_ZARR_SLOTS",
        "FCI_CHUNKS",
        "PARTIAL_CHUNK",
    ):
        env.pop(name, None)
    env.update(
        FIRECUBE=str(firecube),
        ASSUME_YES="1",
        PARALLELISM="1",
        PRODUCT_TYPE="FDHSI",
        INPUT=str(input_dir),
        TARGET=f"file://{tmp_path / 'store.zarr'}",
        PRODUCT_NAME="store.zarr",
        TIME_EPOCH="2024-01-01",
        TIME_SLOTS="1",
        SLOTS_PER_POD="1",
        STORAGE_TYPE="local",
    )
    env.update(overrides)
    env["LOGDIR"] = str(tmp_path / "logs")
    return env


def _run(
    env: dict[str, str], cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT_PATH)],
        env=env,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def _output(result: subprocess.CompletedProcess[str]) -> str:
    return f"exit={result.returncode}\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"


def test_flat_store_writes_arrays_at_root(
    tmp_path: Path, input_dir: Path, firecube_shim: Path
) -> None:
    result = _run(
        _env(tmp_path, input_dir, firecube_shim, RESOLUTIONS="1km", FLAT_STORE="1")
    )

    assert result.returncode == 0, _output(result)
    assert "preallocate done" in result.stdout, _output(result)
    assert "DRIFT-CHECK OK: flat layout" in result.stdout, _output(result)
    root = zarr.open_group(str(tmp_path / "store.zarr"), mode="r")
    assert sorted(root.group_keys()) == []
    assert sorted(root.array_keys()) == [
        "channel",
        "channel_effective_solar_irradiance",
        "counts",
        "latitude",
        "longitude",
        "offset",
        "pixel_quality",
        "pixel_time",
        "platform_altitude",
        "radiance_to_bt_conversion_coefficient_a",
        "radiance_to_bt_conversion_coefficient_b",
        "radiance_to_bt_conversion_coefficient_wavenumber",
        "radiance_to_bt_conversion_constant_c1",
        "radiance_to_bt_conversion_constant_c2",
        "radiance_unit_conversion_coefficient",
        "slope",
        "spatial_ref",
        "subsatellite_latitude",
        "subsatellite_longitude",
        "sun_earth_distance",
        "time",
        "x",
        "y",
    ]


def test_default_layout_writes_one_group_per_resolution(
    tmp_path: Path, input_dir: Path, firecube_shim: Path
) -> None:
    result = _run(_env(tmp_path, input_dir, firecube_shim))

    assert result.returncode == 0, _output(result)
    assert (
        "DRIFT-CHECK OK: grouped layout, static markers present in data_1km, data_2km"
        in (result.stdout)
    ), _output(result)
    root = zarr.open_group(str(tmp_path / "store.zarr"), mode="r")
    assert sorted(root.group_keys()) == ["data_1km", "data_2km"]
    assert sorted(root.array_keys()) == []


def test_fallback_plan_with_channels_filter_that_empties_a_resolution(
    tmp_path: Path, input_dir: Path, firecube_shim: Path
) -> None:
    env = _env(
        tmp_path,
        input_dir,
        firecube_shim,
        EXTRA_OPTIONS="--option channels=vis_06",
        SHIM_FAIL_ZARR_SLOTS="1",
    )
    result = _run(env)

    assert result.returncode == 0, _output(result)
    assert "falling back to local range generation" in result.stdout, _output(result)
    assert "plan cross-check skipped (fallback plan)" in result.stdout, _output(result)
    groups = set(zarr.open_group(str(tmp_path / "store.zarr"), mode="r").group_keys())
    assert groups == {"data_1km"}


def test_flat_store_with_two_resolutions_is_rejected(
    tmp_path: Path, input_dir: Path, firecube_shim: Path
) -> None:
    result = _run(
        _env(
            tmp_path, input_dir, firecube_shim, FLAT_STORE="true", RESOLUTIONS="1km,2km"
        )
    )

    assert result.returncode != 0, _output(result)
    assert (
        "flat_store=true requires exactly one effective resolution"
        in result.stdout + result.stderr
    ), _output(result)
    assert "preallocate done" not in result.stdout, _output(result)
    assert not (tmp_path / "store.zarr" / "counts").exists()


def test_invalid_flat_store_value_stops_before_any_firecube_call(
    tmp_path: Path, input_dir: Path
) -> None:
    calls = tmp_path / "firecube-calls"
    recorder = tmp_path / "firecube"
    recorder.write_text(
        f'#!/usr/bin/env bash\necho "$@" >> "{calls}"\n', encoding="utf-8"
    )
    recorder.chmod(0o755)

    result = _run(
        _env(tmp_path, input_dir, recorder, FLAT_STORE="maybe", RESOLUTIONS="1km")
    )

    assert result.returncode == 2, _output(result)
    assert "FLAT_STORE='maybe' is not a boolean" in result.stdout, _output(result)
    assert not calls.exists()
    assert not (tmp_path / "store.zarr").exists()


def _recorder(tmp_path: Path) -> tuple[Path, Path]:
    """Return ``(firecube stand-in, calls file)``; ``zarr slots`` fails on purpose."""
    calls = tmp_path / "firecube-calls"
    recorder = tmp_path / "firecube"
    recorder.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$@" >> "{calls}"\n'
        '[[ "$1 $2" == "zarr slots" ]] && exit 2\n'
        "exit 0\n",
        encoding="utf-8",
    )
    recorder.chmod(0o755)
    return recorder, calls


def _recorded(calls: Path) -> dict[str, list[str]]:
    """Group the recorded command lines by verb: preallocate, slots, ingest."""
    verbs = {
        "zarr preallocate": "preallocate",
        "zarr slots": "slots",
        "ingest": "ingest",
    }
    grouped: dict[str, list[str]] = {name: [] for name in verbs.values()}
    for line in calls.read_text(encoding="utf-8").splitlines():
        for prefix, name in verbs.items():
            if line.startswith(prefix):
                grouped[name].append(line)
    return grouped


def test_fci_chunks_and_partial_chunk_reach_every_firecube_call(
    tmp_path: Path, input_dir: Path
) -> None:
    recorder, calls = _recorder(tmp_path)

    _run(
        _env(
            tmp_path,
            input_dir,
            recorder,
            TIME_SLOTS="2",
            FCI_CHUNKS="[32,40]",
            PARTIAL_CHUNK="error",
        )
    )

    recorded = _recorded(calls)
    assert [len(recorded[verb]) for verb in ("preallocate", "slots", "ingest")] == [
        1,
        1,
        2,
    ]
    for line in (line for lines in recorded.values() for line in lines):
        assert "--option fci_chunks=[32,40]" in line, line
        assert "--option partial_chunk=error" in line, line


def test_fci_chunks_and_partial_chunk_are_absent_when_unset(
    tmp_path: Path, input_dir: Path
) -> None:
    recorder, calls = _recorder(tmp_path)

    _run(_env(tmp_path, input_dir, recorder))

    recorded = _recorded(calls)
    assert recorded["preallocate"], "preallocate was not called"
    assert recorded["ingest"], "ingest was not called"
    for line in (line for lines in recorded.values() for line in lines):
        assert "--option fci_chunks=" not in line, line
        assert "--option partial_chunk=" not in line, line


@pytest.mark.parametrize("product_type", ["FDHSI", "HRFI"])
def test_product_type_reaches_every_firecube_call(
    tmp_path: Path, input_dir: Path, product_type: str
) -> None:
    recorder, calls = _recorder(tmp_path)

    _run(_env(tmp_path, input_dir, recorder, PRODUCT_TYPE=product_type))

    lines = [line for group in _recorded(calls).values() for line in group]
    assert len(lines) == 3
    for line in lines:
        assert f"--option product_type={product_type}" in line, line


def test_product_type_defaults_to_fdhsi(tmp_path: Path, input_dir: Path) -> None:
    recorder, calls = _recorder(tmp_path)
    env = _env(tmp_path, input_dir, recorder)
    del env["PRODUCT_TYPE"]

    _run(env)

    lines = [line for group in _recorded(calls).values() for line in group]
    assert len(lines) == 3
    for line in lines:
        assert "--option product_type=FDHSI" in line, line


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("FCI_CHUNKS", "32,40"),
        ("FCI_CHUNKS", "[32, 40]"),
        ("FCI_CHUNKS", "[32,x]"),
        ("PARTIAL_CHUNK", "maybe"),
    ],
)
def test_invalid_fci_chunks_or_partial_chunk_stops_before_any_firecube_call(
    tmp_path: Path, input_dir: Path, name: str, value: str
) -> None:
    recorder, calls = _recorder(tmp_path)

    result = _run(_env(tmp_path, input_dir, recorder, **{name: value}))

    assert result.returncode == 2, _output(result)
    assert f"{name}='{value}'" in result.stdout, _output(result)
    assert not calls.exists()


def test_bracketed_option_value_is_not_expanded_as_a_file_glob(
    tmp_path: Path, input_dir: Path
) -> None:
    """``fci_chunks=[32,40]`` is a glob pattern to the shell; it must stay literal."""
    recorder, calls = _recorder(tmp_path)
    workdir = tmp_path / "cwd"
    workdir.mkdir()
    (workdir / "fci_chunks=3").touch()

    _run(_env(tmp_path, input_dir, recorder, FCI_CHUNKS="[32,40]"), cwd=workdir)

    ingest_lines = _recorded(calls)["ingest"]
    assert ingest_lines, "ingest was not called"
    for line in ingest_lines:
        assert "--option fci_chunks=[32,40]" in line, line
        assert "fci_chunks=3" not in line, line
