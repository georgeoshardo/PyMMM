from __future__ import annotations

import asyncio
import json

import dask.array as da
from dask import delayed
import numpy as np
import xarray as xr
import zarr
from zarr.core.buffer import default_buffer_prototype

from pymmm.lazy_zarr import LazyXarrayZarrStore


def _get_bytes(store: LazyXarrayZarrStore, key: str) -> bytes | None:
    async def _get() -> bytes | None:
        buffer = await store.get(key, prototype=default_buffer_prototype())
        return None if buffer is None else buffer.to_bytes()

    return asyncio.run(_get())


def test_lazy_store_reports_zarr_v2_metadata_without_computing_data() -> None:
    computed = False

    def make_block() -> np.ndarray:
        nonlocal computed
        computed = True
        return np.arange(12, dtype=np.uint16).reshape(3, 4)

    array = da.from_delayed(
        delayed(make_block)(),
        shape=(3, 4),
        dtype=np.uint16,
    )
    data = xr.DataArray(array, dims=("Y", "X"), attrs={"source": "synthetic"})

    store = LazyXarrayZarrStore(data, chunks=(2, 2), attrs={"axis_labels": ["y", "x"]})

    metadata = json.loads(_get_bytes(store, ".zarray").decode())
    attrs = json.loads(_get_bytes(store, ".zattrs").decode())

    assert computed is False
    assert metadata["zarr_format"] == 2
    assert metadata["shape"] == [3, 4]
    assert metadata["chunks"] == [2, 2]
    assert metadata["dtype"] == "<u2"
    assert metadata["compressor"] is None
    assert metadata["filters"] is None
    assert metadata["order"] == "C"
    assert attrs["axis_labels"] == ["y", "x"]
    assert attrs["source"] == "synthetic"


def test_lazy_store_returns_padded_uncompressed_chunk_bytes() -> None:
    values = np.arange(15, dtype=np.uint16).reshape(3, 5)
    data = xr.DataArray(da.from_array(values, chunks=(2, 3)), dims=("Y", "X"))
    store = LazyXarrayZarrStore(data, chunks=(2, 3))

    chunk = _get_bytes(store, "1.1")

    expected = np.array(
        [
            [13, 14, 0],
            [0, 0, 0],
        ],
        dtype=np.uint16,
    )
    assert chunk == expected.tobytes(order="C")


def test_lazy_store_can_be_read_by_zarr_open_array() -> None:
    values = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)
    data = xr.DataArray(
        da.from_array(values, chunks=(1, 2, 2)),
        dims=("T", "Y", "X"),
    )
    store = LazyXarrayZarrStore(data, chunks=(1, 2, 2))

    opened = zarr.open_array(store=store, mode="r")

    np.testing.assert_array_equal(opened[:], values)


def test_lazy_store_rejects_unknown_chunk_keys() -> None:
    data = xr.DataArray(
        da.from_array(np.arange(12, dtype=np.uint16).reshape(3, 4), chunks=(2, 2)),
        dims=("Y", "X"),
    )
    store = LazyXarrayZarrStore(data, chunks=(2, 2))

    assert _get_bytes(store, "2.0") is None
    assert _get_bytes(store, "0") is None
    assert _get_bytes(store, ".zgroup") is None


def test_lazy_store_skips_non_json_attrs() -> None:
    class NotJson:
        pass

    data = xr.DataArray(
        da.from_array(np.arange(4, dtype=np.uint16).reshape(2, 2), chunks=(1, 1)),
        dims=("Y", "X"),
        attrs={"keep": "yes", "skip": NotJson()},
    )
    store = LazyXarrayZarrStore(data, chunks=(1, 1), attrs={"axis_labels": ["y", "x"]})

    attrs = json.loads(_get_bytes(store, ".zattrs").decode())

    assert attrs == {"keep": "yes", "axis_labels": ["y", "x"]}
