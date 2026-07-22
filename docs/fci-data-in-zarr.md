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
| `time` | `(time,)` | negligible | slot timestamp coordinate, anchored by `time_epoch` |
| `channel_name` | `(channel,)` | negligible | logical channel names such as `vis_06` and `ir_105` |
| `x`, `y` | `(x,)`, `(y,)` | negligible | GEOS projection angles in radians |
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

## Radiometric calibration

```python
radiance = counts * slope + offset  # mW m-2 sr-1 (cm-1)-1
```

`slope` and `offset` are recorded per acquisition: one value per `(time, channel)`,
not per pixel.

## Projection

GEOS projection parameters (sampling angles, scan origins, detector dimensions)
are FCI-specific. Pre-generated `latitude`/`longitude` grids are not portable to
other geostationary imagers.

## Related

- [Geolocation grids workflow](customization.md#geolocation-grids-workflow)
- [Performance Tuning](performance-tuning.md)
- [Documentation Index](index.md)
