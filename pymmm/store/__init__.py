"""The stores PyMMM writes, and their formats (docs/layout.md).

``experiment_group`` (the per-trench OME-Zarr images and their labels), ``ome``,
``geff`` and ``provenance`` need only numpy and zarr, and geff to write tracks;
``companion`` (the detection checkpoints beside the ND2) also xarray and pandas.
"""
