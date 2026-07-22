# CF Attributes

Change, add, or remove CF attributes on any variable. All attrs live inline in the `Variable(...)` entry inside `schema.py`.

## How

Edit `src/firecube_mtg_fci_l1c/schema.py`. Find the `Variable(...)` entry for the target variable and update its `attrs` dict:

```python
Variable(
    name="counts",
    dims=("time", "y", "x", "channel"),
    dtype=np.uint16,
    fill_value=np.iinfo(np.uint16).max,
    attrs={
        "units": "1",
        "long_name": "FCI raw detector counts",
        "standard_name": "toa_brightness_count",  # add / change / remove here
        "grid_mapping": "spatial_ref",
    },
    source=_counts_source,
),
```

That's the only edit. No other file needs touching for an attrs-only change.

## Files touched

| File | Change |
|------|--------|
| `src/firecube_mtg_fci_l1c/schema.py` | Update `attrs` dict of the target `Variable` entry |

## Renaming a variable

> **Note**: For the MTG FCI L1C plugin, `slope` and `offset` were intentionally kept with
> their original names (per the mlcast-community/mlcast-datasets#43 contract). The rename
> to `scale_factor`/`add_offset` shown below is an example of the capability: use it only
> if you are starting a new plugin or have downstream agreement.

CF and downstream tooling occasionally suggest using canonical names like `scale_factor` and `add_offset` instead of `slope` and `offset` for calibration coefficients. Renaming a variable changes the Zarr array name, so this is more than an attrs edit.

To rename `slope` to `scale_factor`:

1. Change the `name=` field on the `Variable(...)` entry:
   ```python
   Variable(
       name="scale_factor",   # was: "slope"
       dims=("time", "channel"),
       dtype=np.float64,
       fill_value=np.nan,
       attrs={
           "units": "1",      # was: "mW m-2 sr-1 (cm-1)-1"
           "long_name": "Linear scaling factor applied to raw counts",
       },
       source=_slope_source,
   ),
   ```

2. The `units` attribute changes to `"1"` because, per CF §8.1, `scale_factor` is a dimensionless multiplier: `physical_value = raw × scale_factor + add_offset`. The physical units belong on the raw variable's `units`, not on the scaling coefficient.

3. The source function name (`_slope_source`) does NOT need to change: only the Variable entry's `name=` controls the Zarr array name.

4. Existing Zarr stores written with the old name will have `data_<res>/slope`. New stores will have `data_<res>/scale_factor`. There is no automatic migration: readers consuming both must either be updated or use an alias map.

5. Re-run the golden output snapshot test (`tests/test_golden_output.py`) and review the diff: the array name change is expected; nothing else should drift.

## Reserved keys

Don't put these in `attrs`. The ingestor manages them automatically:

- `_FillValue`
- `_ARRAY_DIMENSIONS`
- `firecube_run_id`, `firecube_span_id`, `firecube_internal`

## Common mistakes

- Editing `ingestor.py` for a metadata-only change. Attrs live in `schema.py`.
- Adding a separate attrs registry or dict. Attrs are inline per `Variable`.
- Setting `_FillValue` in `attrs`. Use the `fill_value` field on `Variable` instead.

## See also

[How to add a Zarr variable](../add-zarr-variable.md)
