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

"""Click command group exposed under `firecube plugins mtg_fci_l1c ...`."""

from __future__ import annotations

import json
from pathlib import Path

import click


@click.group(
    name="mtg_fci_l1c", context_settings={"help_option_names": ["-h", "--help"]}
)
def cli() -> None:
    """Register MTG FCI L1C plugin commands under ``firecube plugins``."""


@cli.group(context_settings={"help_option_names": ["-h", "--help"]})
def geo() -> None:
    """Commands for generating and inspecting FCI geolocation grids."""


@geo.command("generate")
@click.option(
    "--resolutions",
    default="1km,2km",
    show_default=True,
    help="Comma-separated resolutions to generate: 500m, 1km, 2km.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Output .npz file path. Defaults to fci_grids.npz in current directory.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing file if present.",
)
@click.option(
    "--sub-satellite-lon",
    type=float,
    default=0.0,
    show_default=True,
    help="Sub-satellite longitude in degrees (0.0 for MTG-I1 at 0°).",
)
def generate_grids(
    resolutions: str,
    output: Path | None,
    overwrite: bool,
    sub_satellite_lon: float,
) -> None:
    """Pre-compute and save FCI full-disk lat/lon grids to a ``.npz`` file."""
    # Imported lazily so CLI help/command discovery does not import heavy
    # numerical modules unless this subcommand is actually executed.
    import time

    import numpy as np

    from .geolocation.projection import DIMSIZE_BY_RESOLUTION, compute_latlon

    if output is None:
        output = Path("fci_grids.npz")

    if output.exists() and not overwrite:
        raise click.ClickException(
            f"Output file already exists: {output}. Use --overwrite to replace."
        )

    # Parse and validate resolution list
    res_map = {"500m": 500, "1km": 1000, "2km": 2000}
    requested = [r.strip() for r in resolutions.split(",") if r.strip()]
    invalid = [r for r in requested if r not in res_map]
    if invalid:
        raise click.ClickException(
            f"Unknown resolutions: {', '.join(invalid)}. Valid: {', '.join(res_map)}"
        )
    res_meters = [res_map[r] for r in requested]

    click.echo(f"Generating FCI lat/lon grids for resolutions: {', '.join(requested)}")
    click.echo(f"Sub-satellite longitude: {sub_satellite_lon}°")
    click.echo()

    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, object] = {
        "generated_by": "firecube plugins mtg_fci_l1c geo generate",
        "sub_satellite_lon": sub_satellite_lon,
        "resolutions": requested,
    }

    for res_m in res_meters:
        res_str = {500: "500m", 1000: "1km", 2000: "2km"}[res_m]
        dimsize = DIMSIZE_BY_RESOLUTION[res_m]
        mem_mb = dimsize * dimsize * 4 * 2 / 1024 / 1024

        click.echo(
            f"  {res_str}: {dimsize}×{dimsize} grid ({mem_mb:.0f} MB lat+lon)..."
        )
        t0 = time.time()
        lat, lon = compute_latlon(res_m, sub_satellite_lon=sub_satellite_lon)
        elapsed = time.time() - t0

        arrays[f"{res_str}_lat"] = lat
        arrays[f"{res_str}_lon"] = lon
        metadata[f"{res_str}_dimsize"] = dimsize
        metadata[f"{res_str}_valid_pixels"] = int(np.isfinite(lat).sum())

        click.echo(f"    Computed in {elapsed:.1f}s")

    arrays["_metadata"] = np.array([json.dumps(metadata)], dtype="U10000")

    click.echo()
    click.echo(f"Saving to {output} ...")
    t0 = time.time()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)  # type: ignore[arg-type]
    elapsed = time.time() - t0

    file_mb = output.stat().st_size / 1024 / 1024
    click.echo(f"Done in {elapsed:.1f}s — {file_mb:.0f} MB ({output.resolve()})")
    click.echo()
    click.echo("Use in ingestion with:")
    click.echo(f"  --option 'fci_grids_file={output.resolve()}'")


@geo.command("info")
@click.option(
    "--grids-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to .npz grids file.",
)
def grids_info(grids_file: Path) -> None:
    """Print metadata and per-resolution stats for a generated grids file."""

    # Keep NumPy and plugin loader imports local to execution of this command.
    import numpy as np

    from .geolocation.grids import FciGrids

    loader = FciGrids(grids_file)
    metadata = loader.get_metadata()

    click.echo(f"File:         {grids_file.resolve()}")
    click.echo(f"Size:         {grids_file.stat().st_size / 1024 / 1024:.0f} MB")
    click.echo(f"Generated by: {metadata.get('generated_by', 'unknown')}")
    click.echo(f"Sub-sat lon:  {metadata.get('sub_satellite_lon', 0.0)}°")
    click.echo(f"Resolutions:  {', '.join(metadata.get('resolutions', []))}")
    click.echo()

    for res_m in loader.available_resolutions():
        res_str = {500: "500m", 1000: "1km", 2000: "2km"}[res_m]
        lat, lon = loader.get_coordinates(res_m)
        valid = int(np.isfinite(lat).sum())
        total = lat.size
        click.echo(
            f"  {res_str}: shape={lat.shape}, dtype={lat.dtype}, "
            f"valid={valid:,}/{total:,} ({100 * valid / total:.1f}%)"
        )
