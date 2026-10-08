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

Channel names are the `channel` coordinate of each group, so a channel can be
selected by name: `ds.sel(channel="ir_105")`.

```python
import xarray as xr

ds = xr.open_zarr("output.zarr", group="data_1km")
```

### Flat layout

A store ingested with `--option flat_store=true` holds a single resolution
(one row of the table above) and has no `data_<res>/` group. The variables sit
at the store root:

```
fci-1km.zarr/
├── counts/  pixel_quality/  pixel_time/  slope/  offset/
└── latitude/  longitude/  x/  y/  time/  channel/  spatial_ref/
```

```python
ds = xr.open_zarr("fci-1km.zarr")
```

Options, rules, and examples are in
[Customization → Flat store layout](customization.md#flat-store-layout).

## Variables

Every resolution group, or the root of a flat store, contains:

| Variable | Shape | Storage | Notes |
|---|---|---|---|
| `counts` | `(time, y, x, channel)` | `uint16`, ~2 GB per slot at 1 km | raw detector counts |
| `pixel_quality` | `(time, y, x, channel)` | `uint8`, ~1 GB per slot at 1 km | 8-bit warning flags (see [bit table](#pixel_quality-bits)) |
| `pixel_time` | `(time, y, x, channel)` | `float64`, ~7.9 GB per slot at 1 km | seconds since 2000-01-01 UTC; `pixel_time_dtype=float32` halves it, `include_pixel_time=false` drops it |
| `slope`, `offset` | `(time, channel)` | negligible | radiometric calibration (see [formula](#radiometric-calibration)) |
| `warm_slope`, `warm_offset` | `(time, channel)` | negligible | IR 3.8 dual-gain calibration for counts above 4095; only in groups with `ir_38` (FDHSI `data_2km`, HRFI `data_1km`), `NaN` for the other channels there (see [formula](#radiometric-calibration)) |
| `subsatellite_latitude`, `subsatellite_longitude`, `platform_altitude`, `sun_earth_distance` | `(time,)` | negligible | satellite position and Sun–Earth distance per slot (see [below](#satellite-position-and-sunearth-distance)) |
| `time` | `(time,)` | negligible | slot time coordinate: the product's start time floored to the minute, anchored by `time_epoch`; stored as `datetime64[s]` |
| `channel` | `(channel,)` | negligible | coordinate of the `channel` dimension: logical channel names such as `vis_06` and `ir_105`, stored as text |
| radiance conversion constants | `(time, channel)` | negligible | seven per-acquisition constants for brightness temperature and reflectance (see [below](#brightness-temperature-and-reflectance)) |
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

Each slot is labelled with its product's observation start time, floored to
the full minute: a product that starts at `12:20:06` gets the label
`12:20:00`. A slot can therefore be selected by its round time:

```python
ds.sel(time="2026-09-28T12:20:00")
```

A slot without an ingested product has no label (`NaT`). The exact
acquisition time of every pixel is in `pixel_time`.

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

### IR 3.8 dual-gain calibration

The IR 3.8 channel (`ir_38`) stores counts up to 8191 to cover its extended
radiometric range. Counts up to 4095 use `slope` and `offset`; counts above
4095 use `warm_slope` and `warm_offset`:

```python
ir38 = ds.sel(channel="ir_38")
radiance = xr.where(
    ir38.counts > 4095,
    ir38.counts * ir38.warm_slope + ir38.warm_offset,
    ir38.counts * ir38.slope + ir38.offset,
)
```

`warm_slope` and `warm_offset` are recorded per acquisition like `slope` and
`offset`, so reprocessed data with different coefficients stays consistent.
They exist only in groups that contain `ir_38` (FDHSI `data_2km`, HRFI
`data_1km`) and are `NaN` for the other channels in those groups.

## Brightness temperature and reflectance

Each group stores the radiance conversion constants of every product as
`(time, channel)` variables, recorded per acquisition like `slope` and
`offset`. The values come from the scalars in `data/<channel>/measured` of
each product. `NaN` marks a constant that the product declares not
applicable to a channel. `ir_38` carries a solar irradiance value in the
products although it is an IR channel.

Slots ingested before the store had these variables hold no values and read
as `NaN`, as do slots ingested before `warm_slope` and `warm_offset` existed.
Every product ingested afterwards fills them. To fill earlier slots, re-run the
ingest over their products with `resume_existing=true`: the per-acquisition
variables are written again from each product, image data that is already
present is left alone.

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
angle. `d` is stored as `sun_earth_distance` (see
[Satellite position and Sun–Earth distance](#satellite-position-and-sunearth-distance)).
The solar zenith angle is not stored; compute it from `latitude`,
`longitude` and `pixel_time`.

## Satellite position and Sun–Earth distance

Each group stores one value per slot for:

| Variable | Units | Source |
|---|---|---|
| `subsatellite_latitude` | degrees_north | `state/platform/subsatellite_latitude` in the L1C product |
| `subsatellite_longitude` | degrees_east | `state/platform/subsatellite_longitude` in the L1C product |
| `platform_altitude` | m | `state/platform/platform_altitude` in the L1C product |
| `sun_earth_distance` | AU | computed for the slot time |

The product records the satellite position about once per second of the
repeat cycle. The store keeps the mean over the cycle; within a cycle the
values vary by less than 0.03 degrees and 520 m. A slot is `NaN` if the
product lacks them.

`sun_earth_distance` is the distance from the Earth's centre to the Sun at
the slot time, computed with astropy. It is not copied from the product:
`state/celestial/earth_sun_distance` there holds the Sun–satellite distance,
which differs by up to about 42 000 km (0.03 %). Over one repeat cycle the
Sun–Earth distance changes by less than 300 km.

```python
reflectance = np.pi * radiance * ds.sun_earth_distance**2 / (E_sun * np.cos(np.deg2rad(sza)))
```

## Projection

GEOS projection parameters (sampling angles, scan origins, detector dimensions)
are FCI-specific. Pre-generated `latitude`/`longitude` grids are not portable to
other geostationary imagers.

## Related

- [Geolocation grids workflow](customization.md#geolocation-grids-workflow)
- [Performance Tuning](performance-tuning.md)
- [Documentation Index](index.md)
