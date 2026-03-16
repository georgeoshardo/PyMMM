# Why We Do Not Use OME-Zarr As The Primary Trench Store

PyMMM stores extracted trenches as a trench-first analysis dataset, with a layout like:

- `Trench x T x C x Y x X`
- sometimes `Trench x T x C x Z x Y x X`

This does **not** fit cleanly into a single OME-Zarr image.

## The Core Issue

OME-NGFF / OME-Zarr multiscale images are designed around standard image axes:

- 2 or 3 spatial axes
- optionally time
- optionally channel or one custom axis

That works for:

- `T x C x Y x X`
- `T x C x Z x Y x X`

But it does **not** work cleanly for:

- `Trench x T x C x Y x X`
- `Trench x T x C x Z x Y x X`

because `Trench` is an extra non-spatial axis on top of the usual `T` and `C` axes.

## Why Not Force It

We do not want to:

- overload `C` to mean trenches
- hide trenches inside a custom axis and lose normal channel semantics
- publish something that looks like OME-Zarr but does not match the way OME-Zarr tools expect image data to be organized

That would reduce interoperability rather than improve it.

## What OME-Zarr *Could* Support

OME-Zarr could still be used as a **companion export** if we store:

- one trench per image group

In that model, each trench image could be a standard:

- `T x C x Y x X`
- or `T x C x Z x Y x X`

This is valid, but it is not a good primary analysis layout for PyMMM because:

- it turns one logical dataset into thousands of separate image groups
- it is less convenient for xarray-based analysis across trenches
- normalized metadata sharing becomes more awkward

## Decision

PyMMM uses an **xarray-native Zarr layout** as the primary format because it matches the natural analysis structure of trench datasets and keeps trench-level metadata, image data, and derived outputs aligned in one coherent hierarchy.

OME-Zarr remains a possible future export format for interoperability, but not the primary internal or distributed trench store.

## References

- OME-NGFF / OME-Zarr spec: <https://ngff.openmicroscopy.org/0.5/>
- OME-Zarr Python docs: <https://ome-zarr.readthedocs.io/en/latest/>
