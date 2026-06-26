from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable
from math import ceil
from typing import Any

import numpy as np
import xarray as xr
from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    Store,
    SuffixByteRequest,
)
from zarr.core.buffer import Buffer, BufferPrototype, default_buffer_prototype


class LazyXarrayZarrStore(Store):
    """Read-only Zarr v2 store backed by a lazy xarray DataArray."""

    supports_writes = False
    supports_deletes = False
    supports_listing = False

    def __init__(
        self,
        data: xr.DataArray,
        *,
        chunks: tuple[int, ...],
        attrs: dict[str, Any] | None = None,
        fill_value: int | float = 0,
    ) -> None:
        if len(chunks) != data.ndim:
            raise ValueError(
                f"chunks must have one entry per data dimension: got {chunks}, "
                f"data has dims {data.dims}"
            )
        if any(chunk <= 0 for chunk in chunks):
            raise ValueError(f"chunks must be positive integers: got {chunks}")

        super().__init__(read_only=True)
        self.data = data
        self.shape = tuple(int(size) for size in data.shape)
        self.chunks = tuple(int(chunk) for chunk in chunks)
        self.dtype = np.dtype(data.dtype)
        self.fill_value = fill_value
        self.attrs = dict(data.attrs)
        if attrs:
            self.attrs.update(attrs)
        self.grid_shape = tuple(ceil(size / chunk) for size, chunk in zip(self.shape, self.chunks))

    def with_read_only(self, read_only: bool = False) -> LazyXarrayZarrStore:
        return self

    def __eq__(self, value: object) -> bool:
        return self is value

    async def get(
        self,
        key: str,
        prototype: BufferPrototype | None = None,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        if prototype is None:
            prototype = default_buffer_prototype()
        await self._ensure_open()

        if key == ".zarray":
            data = self._zarray_bytes()
        elif key == ".zattrs":
            data = self._zattrs_bytes()
        else:
            chunk_index = self._parse_chunk_key(key)
            if chunk_index is None:
                return None
            data = self._chunk_bytes(chunk_index)

        start, stop = _byte_range_indices(len(data), byte_range)
        return prototype.buffer.from_bytes(data[start:stop])

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        return [
            await self.get(key, prototype=prototype, byte_range=byte_range)
            for key, byte_range in key_ranges
        ]

    async def exists(self, key: str) -> bool:
        return key in {".zarray", ".zattrs"} or self._parse_chunk_key(key) is not None

    async def set(self, key: str, value: Buffer) -> None:
        raise NotImplementedError("LazyXarrayZarrStore is read-only")

    async def delete(self, key: str) -> None:
        raise NotImplementedError("LazyXarrayZarrStore is read-only")

    async def list(self) -> AsyncIterator[str]:
        raise NotImplementedError("LazyXarrayZarrStore does not support listing")
        yield

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        raise NotImplementedError("LazyXarrayZarrStore does not support listing")
        yield

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        raise NotImplementedError("LazyXarrayZarrStore does not support listing")
        yield

    def _zarray_bytes(self) -> bytes:
        metadata = {
            "shape": list(self.shape),
            "chunks": list(self.chunks),
            "dtype": self.dtype.str,
            "fill_value": self.fill_value,
            "order": "C",
            "filters": None,
            "dimension_separator": ".",
            "compressor": None,
            "zarr_format": 2,
        }
        return json.dumps(metadata, indent=2).encode()

    def _zattrs_bytes(self) -> bytes:
        return json.dumps(_json_safe_attrs(self.attrs)).encode()

    def _parse_chunk_key(self, key: str) -> tuple[int, ...] | None:
        parts = key.split(".")
        if len(parts) != len(self.shape):
            return None
        try:
            index = tuple(int(part) for part in parts)
        except ValueError:
            return None
        if any(i < 0 or i >= limit for i, limit in zip(index, self.grid_shape)):
            return None
        return index

    def _chunk_bytes(self, chunk_index: tuple[int, ...]) -> bytes:
        slices = []
        for index, chunk, size in zip(chunk_index, self.chunks, self.shape):
            start = index * chunk
            stop = min(start + chunk, size)
            slices.append(slice(start, stop))

        chunk = self.data.isel(dict(zip(self.data.dims, slices))).compute().values
        chunk = np.asarray(chunk, dtype=self.dtype)

        if chunk.shape != self.chunks:
            padded = np.full(self.chunks, self.fill_value, dtype=self.dtype)
            target = tuple(slice(0, size) for size in chunk.shape)
            padded[target] = chunk
            chunk = padded

        return np.ascontiguousarray(chunk).tobytes(order="C")


def _byte_range_indices(length: int, byte_range: ByteRequest | None) -> tuple[int, int]:
    if byte_range is None:
        return 0, length
    if isinstance(byte_range, RangeByteRequest):
        return byte_range.start, byte_range.end
    if isinstance(byte_range, OffsetByteRequest):
        return byte_range.offset, length
    if isinstance(byte_range, SuffixByteRequest):
        return max(length - byte_range.suffix, 0), length
    raise ValueError(f"Unexpected byte range: {byte_range!r}")


def _json_safe_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    safe = {}
    for key, value in attrs.items():
        try:
            json.dumps(value)
        except TypeError:
            continue
        safe[key] = value
    return safe
