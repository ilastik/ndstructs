from typing import Union, Optional, Callable
from pathlib import Path
import urllib.parse
import enum
from functools import partial
import re

import json
import numpy as np
import z5py

from ndstructs import Point5D, Shape5D, Slice5D, Array5D

from ndstructs.datasource.DataSourceUrl import DataSourceUrl
from ndstructs.datasource.DataSource import DataSource, guess_axiskeys
from .UnsupportedUrlException import UnsupportedUrlException
from ndstructs.datasource.BackedSlice5D import BackedSlice5D

from fs.base import FS
from fs.osfs import OSFS
from fs.errors import ResourceNotFound


class N5Block(Array5D):
    class Modes(enum.IntEnum):
        DEFAULT = 0
        VARLENGTH = 1

    @classmethod
    def from_bytes(cls, data: bytes, on_disk_axiskeys: str, dtype: np.dtype, decompressor: Callable[[bytes], bytes]):
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

        compressed_buffer = np.frombuffer(data, offset=header_dtype.itemsize, dtype=np.uint8)
        decompressed_buffer = decompressor(compressed_buffer.tobytes())
        raw_array = np.frombuffer(decompressed_buffer, dtype=dtype.newbyteorder(">"))

        array_shape_f = header_data["dimensions"].squeeze()[::-1]  # reversed because bytes are stored in F order
        array_axiskeys_c = on_disk_axiskeys[::-1]  # reversed because Array5D wants c-order
        return cls(raw_array.reshape(array_shape_f), axiskeys=array_axiskeys_c)


class N5DataSource(DataSource):
    def __init__(self, url: Union[Path, str], *, fs: Optional[FS] = None, location: Point5D = Point5D.zero()):
        url = Path(url).as_posix()
        if not re.search(r"\w\.n5/\w", url, re.IGNORECASE):
            raise UnsupportedUrlException(url)
        self.fs = fs.opendir(url) if fs else OSFS(url)

        with self.fs.openbin("attributes.json", "r") as f:
            attributes_json_bytes = f.read()
        attributes = json.loads(attributes_json_bytes.decode("utf8"))

        dimensions = attributes["dimensions"]
        blockSize = attributes["blockSize"]
        self.axiskeys = "".join(attributes["axes"]).lower() or guess_axiskeys(dimensions[::-1])[::-1]
        if not (len(dimensions) == len(blockSize) == len(self.axiskeys)):
            raise ValueError("Shape/axis mismatch: {json.dumps(attributes, indent=4)}")

        super().__init__(
            url=url,
            tile_shape=Shape5D(**{axis: length for axis, length in zip(self.axiskeys, blockSize)}),
            shape=Shape5D(**{axis: length for axis, length in zip(self.axiskeys, dimensions)}),
            dtype=np.dtype(attributes["dataType"]).newbyteorder(">"),
            name=Path(url).name,
            location=location,
        )
        compression_type = attributes["compression"]["type"]
        # fmt: off
        if compression_type == "gzip":
            import gzip
            self.decompressor = gzip.decompress
        elif compression_type == "bzip2":
            import bz2
            self.decompressor = bz2.decompress
        elif compression_type == "xz":
            import lzma
            self.decompressor = lzma.decompress
        elif compression_type == "raw":
            self.decompressor = lambda data: data
        elif compression_type == "lz4":
            raise NotImplementedError(f"lz4 decompression not currently supported")
        else:
            raise ValueError(f"Bad compression type {compression_type}")
        # fmt: on

    def _get_tile(self, tile: Slice5D) -> Array5D:
        slice_address_components = (tile.start // self.tile_shape).to_tuple(self.axiskeys)
        slice_address = "/".join(str(int(comp)) for comp in slice_address_components)
        try:
            with self.fs.openbin(slice_address) as f:
                raw_tile = f.read()
            tile_5d = N5Block.from_bytes(
                data=raw_tile, on_disk_axiskeys=self.axiskeys, dtype=self.dtype, decompressor=self.decompressor
            )
        except ResourceNotFound as e:
            tile_5d = self._allocate(roi=tile, fill_value=0)
        return tile_5d.translated(tile.start)


DataSource.REGISTRY.insert(0, N5DataSource)
