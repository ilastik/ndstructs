from typing import Union, Optional, Callable
from pathlib import Path
import urllib.parse
import enum
from functools import partial
import re
import gzip
import bz2
import lzma
import json

import numpy as np
import z5py

from ndstructs import Point5D, Shape5D, Slice5D, Array5D

from ndstructs.datasource.DataSourceUrl import DataSourceUrl
from ndstructs.datasource.DataSource import DataSource, guess_axiskeys
from .UnsupportedUrlException import UnsupportedUrlException
from ndstructs.datasource.DataSourceSlice import DataSourceSlice

from fs import open_fs
from fs.base import FS
from fs.errors import ResourceNotFound


class N5Block(Array5D):
    class Modes(enum.IntEnum):
        DEFAULT = 0
        VARLENGTH = 1

    COMPRESSORS = {
        "gzip": gzip.compress,  # FIXME: compression arguments? level and stuff?
        "bzip2": bz2.compress,
        "xz": lzma.compress,
        "raw": lambda data: data,
    }

    DECOMPRESSORS = {"gzip": gzip.decompress, "bzip2": bz2.decompress, "xz": lzma.decompress, "raw": lambda data: data}

    @classmethod
    def from_bytes(cls, data: bytes, on_disk_axiskeys: str, dtype: np.dtype, compression_type: str):
        decompressor = cls.DECOMPRESSORS[compression_type]
        data = np.frombuffer(data, dtype=np.uint8)

        header_types = [
            ("mode", ">u2"),  # mode (uint16 big endian, default = 0x0000, varlength = 0x0001)
            ("num_dims", ">u2"),  # number of dimensions (uint16 big endian)
        ]
        preamble = np.frombuffer(data, dtype=header_types, count=1)
        header_types.append(
            ("dimensions", str(preamble["num_dims"].item()) + ">u4")  # dimension 1[,...,n] (uint32 big endian)
        )

        if preamble["mode"].item() == cls.Modes.VARLENGTH.value:
            header_types.append(("num_elements", ">u4"))  # mode == varlength ? number of elements (uint32 big endian)
            raise RuntimeError("Don't know how to handle varlen N5 blocks")

        header_dtype = np.dtype(header_types)
        header_data = np.frombuffer(data, dtype=header_dtype, count=1)
        array_shape = header_data["dimensions"].squeeze()

        compressed_buffer = np.frombuffer(data, offset=header_dtype.itemsize, dtype=np.uint8)
        decompressed_buffer = decompressor(compressed_buffer.tobytes())
        raw_array = np.frombuffer(decompressed_buffer, dtype=dtype.newbyteorder(">")).reshape(array_shape, order="F")

        return cls(raw_array, axiskeys=on_disk_axiskeys)

    def to_n5_bytes(self, axiskeys: str, compression_type: str):
        compressor = self.COMPRESSORS[compression_type]
        # because the axistags are already reversed, bytes must be written in C order. If we wrote using tobytes("F"),
        # we'd need to leave the axistags as they were originally
        data_buffer = compressor(self.raw(axiskeys).astype(self.dtype.newbyteorder(">")).tobytes("C"))
        tile_types = [
            ("mode", ">u2"),  # mode (uint16 big endian, default = 0x0000, varlength = 0x0001)
            ("num_dims", ">u2"),  # number of dimensions (uint16 big endian)
            ("dimensions", f"{len(axiskeys)}>u4"),  # dimension 1[,...,n] (uint32 big endian)
            ("data", f"{len(data_buffer)}u1"),
        ]
        tile = np.zeros(1, dtype=tile_types)
        tile["mode"] = self.Modes.DEFAULT.value
        tile["num_dims"] = len(axiskeys)
        tile["dimensions"] = [self.shape[k] for k in axiskeys[::-1]]
        tile["data"] = np.ndarray(len(data_buffer), dtype=np.uint8, buffer=data_buffer)
        return tile.tobytes()


class N5DataSource(DataSource):
    def __init__(self, path: Path, *, location: Point5D = Point5D.zero(), filesystem: FS):
        if not re.search(r"\w\.n5/\w", path.as_posix(), re.IGNORECASE):
            raise UnsupportedUrlException(path.as_posix())
        self.filesystem = filesystem.opendir(path.as_posix())

        with self.filesystem.openbin("attributes.json", "r") as f:
            attributes_json_bytes = f.read()
        attributes = json.loads(attributes_json_bytes.decode("utf8"))

        dimensions = attributes["dimensions"]
        blockSize = attributes["blockSize"]
        self.axiskeys = "".join(attributes["axes"]).lower() or guess_axiskeys(dimensions[::-1])[::-1]
        if not (len(dimensions) == len(blockSize) == len(self.axiskeys)):
            raise ValueError("Shape/axis mismatch: {json.dumps(attributes, indent=4)}")

        super().__init__(
            path=path,
            tile_shape=Shape5D(**{axis: length for axis, length in zip(self.axiskeys, blockSize)}),
            shape=Shape5D(**{axis: length for axis, length in zip(self.axiskeys, dimensions)}),
            dtype=np.dtype(attributes["dataType"]).newbyteorder(">"),
            location=location,
        )
        self.compression_type = attributes["compression"]["type"]
        if self.compression_type not in N5Block.DECOMPRESSORS.keys():
            raise NotImplementedError(f"Don't know how to decompress from {self.compression_type}")

    def _get_tile(self, tile: Slice5D) -> Array5D:
        slice_address_components = (tile.start // self.tile_shape).to_tuple(self.axiskeys)
        slice_address = "/".join(str(int(comp)) for comp in slice_address_components)
        try:
            with self.filesystem.openbin(slice_address) as f:
                raw_tile = f.read()
            tile_5d = N5Block.from_bytes(
                data=raw_tile, on_disk_axiskeys=self.axiskeys, dtype=self.dtype, compression_type=self.compression_type
            )
        except ResourceNotFound as e:
            tile_5d = self._allocate(roi=tile, fill_value=0)
        return tile_5d.translated(tile.start)


DataSource.REGISTRY.insert(0, N5DataSource)
