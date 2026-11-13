"""Write a small ND2 from a large one: some positions, channels and frames, and a
window of the image.

    python subset_nd2.py big.nd2 small.nd2 --positions 2 5 --times 20 30 \
        --channels 0 1 --y 232 1064 --x 0 768

The ``nd2`` package reads ND2 files and does not write them, so this copies the file's
own chunks and changes only what the subset changes. It handles the layout our
acquisitions have (uncompressed, a time loop around a position loop around a channel
loop) and refuses anything else. The result is checked against ``nd2``'s reader only;
Nikon's software has not been tried on it.

An ND2 file is a run of chunks, each ``magic, name length, data length`` (4, 4 and 8
bytes), then the name padded with zeros, then the data. The padding is chosen so every
chunk starts on a multiple of 4096 bytes. The last chunk is a map from chunk name to
offset and data length, and the file's last 8 bytes give that map's offset.

The metadata chunks (``Image...LV``) hold a tree of named values in Nikon's binary
encoding ("CLX lite"). :func:`decode` and :func:`encode` read and write it; encoding a
decoded chunk gives back the same bytes, which :func:`subset` checks for every chunk
it changes.

What is copied and what is changed:

- ``ImageDataSeq|N!``, one per frame: an 8-byte time, then the pixels as (Y, X,
  channel). The kept frames are cropped to the window and the kept channels, and
  renumbered from 0.
- ``ImageAttributesLV!``: width, height, channels and frame count.
- ``ImageMetadataSeqLV|0!``: the kept channels' planes (name, colour, camera setting).
- ``ImageMetadataLV!``: the time loop's count; the dropped positions are marked
  invalid in the position loop's ``pItemValid``, which is how the format records a
  skipped position (the reader names the kept ones again from 0: ``XYPos:0``,
  ``XYPos:1``); the channel loop keeps the kept channels.
- ``CustomData|X!`` and the other arrays with one value per frame (stage position,
  focus offset, acquisition time, exposure): the kept frames' values. Their length is
  also written in the XML of ``CustomDataVar|CustomDataV2_0!``, which is updated.
- The ``Authentication`` chunks are left out; they sign the original frames.
- Every other chunk is copied as it is. That includes the text description, which
  still gives the original's dimensions and channels.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path
from typing import Any, BinaryIO

import nd2
import numpy as np
from nd2._parse._chunk_decode import get_chunkmap

MAGIC = 0x0ABECEDA
ALIGN = 4096
HEADER = struct.Struct("<IIQ")  # magic, name length, data length
FILE_SIGNATURE = b"ND2 FILE SIGNATURE CHUNK NAME01!"
MAP_NAME = b"ND2 FILEMAP SIGNATURE NAME 0001!"
MAP_SIGNATURE = b"ND2 CHUNK MAP SIGNATURE 0000001!"
FRAME_NAME_LENGTH = 4072  # nd2 reads a frame's pixels 4096 bytes after its chunk starts

# CLX lite value types: sizes of the fixed-width ones, and the others used here
FIXED = {1: 1, 2: 4, 3: 4, 4: 8, 5: 8, 6: 8, 7: 8}
STRING, BYTES, LEVEL = 8, 9, 11
U32 = struct.Struct("<I")

Node = dict[str, Any]


def _parse(data: bytes, pos: int, count: int) -> tuple[list[Node], int]:
    """``count`` values from ``pos``: nodes ``{kind, name, head, value | kids, order}``."""
    nodes = []
    for _ in range(count):
        start = pos
        kind, name_length = data[pos], data[pos + 1]
        head = data[pos : pos + 2 + 2 * name_length]
        name = head[2:].decode("utf-16-le")[:-1]
        node: Node = {"kind": kind, "head": head, "name": name, "start": start}
        pos += len(head)
        if kind in FIXED:
            node["value"] = data[pos : pos + FIXED[kind]]
        elif kind == STRING:
            end = pos
            while data[end : end + 2] != b"\x00\x00":
                end += 2
            node["value"] = data[pos : end + 2]
        elif kind == BYTES:
            (size,) = struct.unpack_from("<Q", data, pos)
            node["value"] = data[pos : pos + 8 + size]
        elif kind == LEVEL:
            # its children, then a table of their offsets from the level's start
            children, length = struct.unpack_from("<IQ", data, pos)
            kids, end = _parse(data, pos + 12, children)
            if end != start + length:
                raise ValueError(f"{name}: level length does not match")
            at = {kid["start"] - start: k for k, kid in enumerate(kids)}
            table = struct.unpack_from(f"<{children}Q", data, end)
            node["kids"], node["order"] = kids, [at[offset] for offset in table]
            pos = end + 8 * children
        else:
            raise ValueError(f"metadata value of unknown type {kind}")
        if kind != LEVEL:
            pos += len(node["value"])
        nodes.append(node)
    return nodes, pos


def decode(data: bytes) -> Node:
    """A metadata chunk as a tree."""
    (root,), end = _parse(data, 0, 1)
    if end != len(data):
        raise ValueError("metadata chunk has bytes after its tree")
    return root


def encode(node: Node) -> bytes:
    """A tree as metadata chunk bytes (the inverse of :func:`decode`)."""
    if node["kind"] != LEVEL:
        return node["head"] + node["value"]
    start = len(node["head"]) + 12
    body, offsets = b"", []
    for kid in node["kids"]:
        offsets.append(start + len(body))
        body += encode(kid)
    table = struct.pack(f"<{len(offsets)}Q", *(offsets[k] for k in node["order"]))
    size = struct.pack("<IQ", len(node["kids"]), start + len(body))
    return node["head"] + size + body + table


def find(node: Node, path: str) -> Node:
    """The value at ``path`` below ``node``: names joined by ``/``, or a number for
    an item of a list."""
    for part in path.split("/"):
        kids = node["kids"]
        if part.isdigit():
            node = kids[int(part)]
            continue
        found = [kid for kid in kids if kid["name"] == part]
        if not found:
            raise ValueError(f"metadata has no {path}")
        node = found[0]
    return node


def set_u32(node: Node, path: str, number: int) -> None:
    value = find(node, path)
    if value["kind"] not in (2, 3):
        raise ValueError(f"{path} is not a 32-bit integer")
    value["value"] = U32.pack(number)


def keep_items(level: Node, keep: list[int], names: bool = True) -> None:
    """Keep the items ``keep`` of a list, in that order. With ``names``, the items
    are named ``a0``, ``a1``... and are named again from ``a0``."""
    kids = level["kids"]
    if names and [kid["name"] for kid in kids] != [f"a{k}" for k in range(len(kids))]:
        raise ValueError(f"{level['name']} is not a list of a0, a1, ...")
    level["kids"] = []
    for new, old in enumerate(keep):
        kid = dict(kids[old])
        if names:
            name = f"a{new}\x00".encode("utf-16-le")
            kid["head"] = bytes([kid["kind"], len(name) // 2]) + name
        level["kids"].append(kid)
    level["order"] = list(range(len(keep)))


def read_chunk(fh: BinaryIO, offset: int) -> bytes:
    """The data of the chunk at ``offset``."""
    fh.seek(offset)
    magic, name_length, data_length = HEADER.unpack(fh.read(HEADER.size))
    if magic != MAGIC:
        raise ValueError(f"no chunk at {offset}")
    fh.seek(name_length, 1)
    return fh.read(data_length)


class Writer:
    """Appends chunks on 4096-byte boundaries and keeps the map of what it wrote."""

    def __init__(self, fh: BinaryIO) -> None:
        self.fh = fh
        self.map: list[tuple[bytes, int, int]] = []

    def chunk(self, name: bytes, *data: bytes, name_length: int | None = None) -> None:
        size = sum(len(d) for d in data)
        offset = self.fh.tell()
        if offset % ALIGN:
            raise RuntimeError("chunk off its boundary")
        if name_length is None:  # pad the name so the chunk ends on a boundary
            name_length = len(name) + (-(HEADER.size + len(name) + size)) % ALIGN
        self.fh.write(HEADER.pack(MAGIC, name_length, size))
        self.fh.write(name.ljust(name_length, b"\x00"))
        for d in data:
            self.fh.write(d)
        self.fh.write(b"\x00" * (-self.fh.tell() % ALIGN))
        self.map.append((name, offset, size))

    def finish(self) -> None:
        """Write the chunk map and the pointer to it that ends the file."""
        offset = self.fh.tell()
        entries = b"".join(n + struct.pack("<QQ", o, s) for n, o, s in self.map)
        tail = MAP_SIGNATURE + struct.pack("<Q", offset)
        name_length = 4128
        # the entries end with the signature, and the file ends with it again
        gap = -(HEADER.size + name_length + len(entries) + 2 * len(tail)) % ALIGN
        size = len(entries) + gap + 2 * len(tail)
        self.fh.write(HEADER.pack(MAGIC, name_length, size))
        self.fh.write(MAP_NAME.ljust(name_length, b"\x00"))
        self.fh.write(entries + tail + b"\x00" * gap + tail)


def subset(
    source: Path,
    target: Path,
    positions: list[int],
    times: tuple[int, int],
    channels: list[int] | None = None,
    y: tuple[int, int] | None = None,
    x: tuple[int, int] | None = None,
) -> None:
    """Write ``target`` with ``positions`` and ``channels`` (indices), frames
    ``times[0]`` up to ``times[1]`` and the image window ``y``, ``x`` of ``source``.
    """
    with nd2.ND2File(source) as f:
        sizes = dict(f.sizes)
        attributes = f.attributes
        loops = f.loop_indices  # one {"T": t, "P": p} per stored frame, in file order
    if list(sizes) != ["T", "P", "C", "Y", "X"] or attributes.compressionType:
        raise ValueError(f"only uncompressed T, P, C, Y, X files are handled: {sizes}")
    height, width, n_channels = sizes["Y"], sizes["X"], sizes["C"]
    if attributes.componentCount != n_channels:
        raise ValueError("expected one component per channel")
    y0, y1 = y or (0, height)
    x0, x1 = x or (0, width)
    t0, t1 = times
    positions = sorted(set(positions))
    channels = list(range(n_channels)) if channels is None else list(channels)
    if not (0 <= t0 < t1 <= sizes["T"]):
        raise ValueError(f"frames must be within 0-{sizes['T']}")
    if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
        raise ValueError("the window is outside the image")
    if not positions or positions[0] < 0 or positions[-1] >= sizes["P"]:
        raise ValueError(f"positions must be within 0-{sizes['P'] - 1}")
    if not channels or len(set(channels)) != len(channels):
        raise ValueError("give each channel once")
    if min(channels) < 0 or max(channels) >= n_channels:
        raise ValueError(f"channels must be within 0-{n_channels - 1}")
    keep = [
        k for k, at in enumerate(loops) if t0 <= at["T"] < t1 and at["P"] in positions
    ]
    if len(keep) != (t1 - t0) * len(positions):
        raise ValueError("the file's frames do not follow its time and position loops")
    pixel = np.dtype(f"<u{attributes.bitsPerComponentInMemory // 8}")
    frame_bytes = height * width * n_channels * pixel.itemsize

    def attributes_of_subset(root: Node) -> None:
        for key, number in {
            "uiWidth": x1 - x0,
            "uiWidthBytes": (x1 - x0) * len(channels) * pixel.itemsize,
            "uiHeight": y1 - y0,
            "uiTileWidth": x1 - x0,
            "uiTileHeight": y1 - y0,
            "uiComp": len(channels),
            "uiVirtualComponents": len(channels),
            "uiSequenceCount": len(keep),
        }.items():
            set_u32(root, key, number)

    def planes_of_subset(planes: Node) -> None:
        for key in ("uiCount", "uiCompCount", "uiSampleCount"):
            set_u32(planes, key, len(channels))
        keep_items(find(planes, "sPlaneNew"), channels)
        keep_items(find(planes, "sSampleSetting"), channels)
        for new, plane in enumerate(find(planes, "sPlaneNew")["kids"]):
            index = [kid for kid in plane["kids"] if kid["name"] == "uiSampleIndex"]
            if index:  # the picture planes name their camera setting
                if U32.unpack(index[0]["value"])[0] != channels[new]:
                    raise ValueError("a channel's plane and setting do not match")
                index[0]["value"] = U32.pack(new)

    def pictures_of_subset(root: Node) -> None:
        planes_of_subset(find(root, "sPicturePlanes"))

    def experiment_of_subset(root: Node) -> None:
        position = find(root, "ppNextLevelEx/0")
        channel = find(position, "ppNextLevelEx/0")
        kinds = [find(level, "eType")["value"] for level in (root, position, channel)]
        if kinds != [U32.pack(1), U32.pack(2), U32.pack(6)]:
            raise ValueError("expected a time loop around positions around channels")
        set_u32(root, "uLoopPars/uiCount", t1 - t0)
        for level, count, kept in (
            (position, sizes["P"], [p in positions for p in range(sizes["P"])]),
            (channel, n_channels, [True] * len(channels)),
        ):
            valid = find(level, "pItemValid")
            if len(valid["value"]) != 8 + count or not all(valid["value"][8:]):
                raise ValueError("the file already has skipped positions or channels")
            valid["value"] = struct.pack("<Q", len(kept)) + bytes(kept)
        planes_of_subset(find(channel, "uLoopPars/pPlanes"))
        keep_items(find(channel, "uLoopPars/Points"), channels, names=False)

    def tags_of_subset(data: bytes) -> bytes:
        # the XML list of the per-frame arrays gives each one's length
        size = b'<Size runtype="lx_int32" value="%d"/>'
        return data.replace(size % len(loops), size % len(keep))

    def per_frame(data: bytes) -> bytes:
        # an array with one value per stored frame keeps the kept frames' values
        if len(data) % len(loops) or len(data) // len(loops) not in (1, 2, 4, 8):
            return data
        return np.frombuffer(data, f"V{len(data) // len(loops)}")[keep].tobytes()

    tree_changes = {
        b"ImageAttributesLV!": attributes_of_subset,
        b"ImageMetadataSeqLV|0!": pictures_of_subset,
        b"ImageMetadataLV!": experiment_of_subset,
    }

    with open(source, "rb") as src, open(target, "wb") as dst:
        chunks = get_chunkmap(src)
        src.seek(0)
        start = src.read(HEADER.size + 32 + 64)  # the signature chunk, kept as it is
        if start[HEADER.size : HEADER.size + 32] != FILE_SIGNATURE:
            raise ValueError(f"{source} is not an ND2 file")
        dst.write(start.ljust(ALIGN, b"\x00"))
        out = Writer(dst)
        before_frames = True
        for name in sorted(chunks, key=lambda name: chunks[name][0]):
            if b"Authentication" in name:
                continue
            if name.startswith(b"ImageDataSeq|"):
                if before_frames:  # all kept frames go where the first frame was
                    before_frames = False
                    for new, old in enumerate(keep):
                        data = read_chunk(src, chunks[b"ImageDataSeq|%d!" % old][0])
                        if len(data) != 8 + frame_bytes:
                            raise ValueError(f"frame {old} has an unexpected size")
                        image = np.frombuffer(data, pixel, offset=8).reshape(
                            height, width, n_channels
                        )
                        window = image[y0:y1, x0:x1][:, :, channels]
                        out.chunk(
                            b"ImageDataSeq|%d!" % new,
                            data[:8],
                            np.ascontiguousarray(window).tobytes(),
                            name_length=FRAME_NAME_LENGTH,
                        )
                continue
            data = read_chunk(src, chunks[name][0])
            if name in tree_changes:
                root = decode(data)
                if encode(root) != data:
                    raise ValueError(f"{name!r} does not encode back to itself")
                tree_changes[name](root)
                data = encode(root)
            elif name == b"CustomDataVar|CustomDataV2_0!":
                data = tags_of_subset(data)
            elif name.startswith(b"CustomData|"):
                data = per_frame(data)
            out.chunk(name, data)
        out.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--positions", type=int, nargs="+", required=True)
    parser.add_argument("--times", type=int, nargs=2, required=True)
    parser.add_argument("--channels", type=int, nargs="+")
    parser.add_argument("--y", type=int, nargs=2)
    parser.add_argument("--x", type=int, nargs=2)
    a = parser.parse_args()
    subset(a.source, a.target, a.positions, tuple(a.times), a.channels, a.y, a.x)
    with nd2.ND2File(a.target) as f:
        megabytes = a.target.stat().st_size / 1e6
        print(f"{a.target}: {dict(f.sizes)}, {megabytes:.1f} MB")


if __name__ == "__main__":
    main()
