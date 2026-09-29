# MTG FCI L1C Data in Zarr

Reference for the Zarr cube produced by the `mtg_fci_l1c` ingestor.

## Groups

```
output.zarr/
├── data_500m/     # HRFI only
├── data_1km/      # FDHSI and HRFI
└── data_2km/      # FDHSI only
```

| Group | Product | Y × X | Channels | Channel names |
|---|---|---|---|---|
| `data_500m` | HRFI | 22272 × 22272 | 2 | `vis_06`, `nir_22` |
| `data_1km` | FDHSI | 11136 × 11136 | 8 | `vis_04`, `vis_05`, `vis_06`, `vis_08`, `vis_09`, `nir_13`, `nir_16`, `nir_22` |
| `data_1km` | HRFI | 11136 × 11136 | 2 | `ir_38`, `ir_105` |
| `data_2km` | FDHSI | 5568 × 5568 | 8 | `ir_38`, `wv_63`, `wv_73`, `ir_87`, `ir_97`, `ir_105`, `ir_123`, `ir_133` |

Channel names are also stored per group in the `channel_name[c]` array.

## Variables

Every group contains:

| Variable | Shape | Storage | Notes |
|---|---|---|---|
| `counts` | `(time, y, x, channel)` | `uint16`, ~2 GB per slot at 1 km | raw detector counts |
| `pixel_quality` | `(time, y, x, channel)` | `uint8`, ~1 GB per slot at 1 km | 8-bit warning flags (see [bit table](#pixel_quality-bits)) |
| `pixel_time` | `(time, y, x, channel)` | `float64`, ~7.9 GB per slot at 1 km | seconds since 2000-01-01 UTC; `pixel_time_dtype=float32` halves it, `include_pixel_time=false` drops it |
| `slope`, `offset` | `(time, channel)` | negligible | radiometric calibration (see [formula](#radiometric-calibration)) |
| `time` | `(time,)` | negligible | slot timestamp coordinate, anchored by `time_epoch`; stored as `datetime64[s]` |
| `channel_name` | `(channel,)` | negligible | logical channel names such as `vis_06` and `ir_105` |
| radiance conversion constants | `(channel,)` | negligible | seven static per-channel constants for brightness temperature and reflectance (see [below](#brightness-temperature-and-reflectance)) |
| `x`, `y` | `(x,)`, `(y,)` | negligible | GEOS projection coordinates. Default units: metres (east-positive x, north-positive y). Use `--option projection_units=radian` for radian output. See [Projection units](customization.md#projection-units). |
| `latitude`, `longitude` | `(y, x)` | `float32`, 1.9 GB / 475 MB / 120 MB per array at 500 m / 1 km / 2 km | static, computed once per group; `NaN` beyond Earth's limb |
| `spatial_ref` | `()` | negligible | CF grid-mapping container with geostationary projection metadata |

### `pixel_quality` bits

| Bit | Warning |
|---|---|
| 0 | Missing |
| 1 | Radiometric |
| 2 | Noise |
| 3 | Geolocation |
| 4 | Saturation |
| 5 | Straylight correction |
| 6 | Extended dynamic range |
| 7 | Encoding saturation |

### `x` and `y` coordinates

`x` is east-positive. `y` is north-positive. Both default to metres
(`standard_name=projection_x_coordinate`, `units=m`), which works out of the
box with rioxarray, cartopy, GDAL, and satpy.

To write radian coordinates (the native unit from the source netCDF), pass
`--option projection_units=radian` at ingest time. See
[Projection units](customization.md#projection-units) for the full option
reference and the schema-drift warning.

### `time` coordinate

`time` is stored as `datetime64[s]`. The `units` and `calendar` attributes are
not written to Zarr array metadata; xarray manages them via encoding on the
native `datetime64[s]` dtype. Reading with xarray returns proper `datetime64`
values without any extra configuration. This is tracked in
[issue #3](https://github.com/eumetsat/firecube-mtg-fci-l1c/issues/3).

### FillValue

Firecube stamps the `_FillValue` attribute on `counts` and other numeric
arrays at ingest time. xarray uses `_FillValue` to mask fill pixels to `NaN`
on read, so masking works without extra configuration.

Stores written before this behavior existed lack the attribute. Stamp it
post-hoc with the [fix-fillvalue command](customization.md#fix-fillvalue-legacy-stores),
or let a resumed or re-run ingest add it when the arrays are next touched.

## Radiometric calibration

```python
radiance = counts * slope + offset  # mW m-2 sr-1 (cm-1)-1
```

`slope` and `offset` are recorded per acquisition: one value per `(time, channel)`,
not per pixel.

## Brightness temperature and reflectance

Each group stores the per-channel constants from the FCI L1C files as
`(channel,)` variables. They do not change over time, so they have no `time`
dimension and are written once when the store is first ingested. `NaN` marks
a constant that does not apply to a channel.

| Variable | Units | Set for |
|---|---|---|
| `radiance_unit_conversion_coefficient` | W mW-1 cm-1 um-1 | all channels |
| `radiance_to_bt_conversion_coefficient_wavenumber` | cm-1 | IR/WV channels |
| `radiance_to_bt_conversion_coefficient_a` | 1 | IR/WV channels |
| `radiance_to_bt_conversion_coefficient_b` | K | IR/WV channels |
| `radiance_to_bt_conversion_constant_c1` | mW m-2 sr-1 (cm-1)-4 | IR/WV channels |
| `radiance_to_bt_conversion_constant_c2` | cm K | IR/WV channels |
| `channel_effective_solar_irradiance` | mW m-2 (cm-1)-1, at 1 AU | VIS/NIR channels and `ir_38` |

They are written when calibration is enabled (the default,
`include_calibration=true`).

```python
import numpy as np
import xarray as xr

ds = xr.open_zarr(store, group="data_2km")
ds = ds.assign_coords(channel=ds.channel_name.astype(str))

radiance = ds.counts * ds.slope + ds.offset  # mW m-2 sr-1 (cm-1)-1

# Brightness temperature (K), IR/WV channels
nu = ds.radiance_to_bt_conversion_coefficient_wavenumber
a = ds.radiance_to_bt_conversion_coefficient_a
b = ds.radiance_to_bt_conversion_coefficient_b
c1 = ds.radiance_to_bt_conversion_constant_c1
c2 = ds.radiance_to_bt_conversion_constant_c2
bt = (c2 * nu / np.log(1.0 + c1 * nu**3 / radiance) - b) / a

# Radiance in W m-2 sr-1 um-1
radiance_um = radiance * ds.radiance_unit_conversion_coefficient
```

Reflectance for VIS/NIR channels is
`pi * radiance * d**2 / (channel_effective_solar_irradiance * cos(sza))`,
where `d` is the Sun–Earth distance in AU and `sza` is the solar zenith
angle. Neither is stored yet.

## Projection

GEOS projection parameters (sampling angles, scan origins, detector dimensions)
are FCI-specific. Pre-generated `latitude`/`longitude` grids are not portable to
other geostationary imagers.

## Related

- [Geolocation grids workflow](customization.md#geolocation-grids-workflow)
- [Performance Tuning](performance-tuning.md)
- [Documentation Index](index.md)
