from abc import ABC, abstractmethod
from ndstructs.datasource.n5_attributes import N5Compressor, N5DatasetAttributes
from ndstructs.utils.json_serializable import JsonObject, JsonValue, ensureJsonInt, ensureJsonString, ensureJsonIntArray, ensureJsonObject, ensureJsonStringArray
from typing import TypeVar, Dict, Any, Type
from pathlib import Path
import enum
import re
import gzip
import bz2
import lzma
import json
import pickle

import numpy as np

from ndstructs import Point5D, Shape5D, Interval5D, Array5D

from ndstructs.datasource.DataSource import DataSource, guess_axiskeys
from .UnsupportedUrlException import UnsupportedUrlException
from ndstructs.datasource.DataRoi import DataRoi

from fs import open_fs
from fs.base import FS
from fs.errors import ResourceNotFound
from fs import open_fs


class N5Block(Array5D):
    class Modes(enum.IntEnum):
        DEFAULT = 0
        VARLENGTH = 1

    @classmethod
    def from_bytes(cls, data: bytes, axiskeys: str, dtype: np.dtype, compression: N5Compressor):
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
        decompressed_buffer = compression.decompress(compressed_buffer.tobytes())
        raw_array = np.frombuffer(decompressed_buffer, dtype=dtype.newbyteorder(">")).reshape(array_shape, order="F") # type: ignore

        return cls(raw_array, axiskeys=axiskeys[::-1])

    def to_n5_bytes(self, axiskeys: str, compression: N5Compressor):
        # because the axistags are written in reverse order to attributes.json, bytes must be written in C order.
        data_buffer = compression.compress(self.raw(axiskeys).astype(self.dtype.newbyteorder(">")).tobytes("C")) # type: ignore
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
        tile["data"] = np.ndarray((len(data_buffer),), dtype=np.uint8, buffer=data_buffer)
        return tile.tobytes()



class N5DataSource(DataSource):
    """A DataSource representing an N5 dataset. "axiskeys" are, like everywhere else in ndstructs, C-ordered."""

    def __init__(self, path: Path, *, location: Point5D = Point5D.zero(), filesystem: FS):
        url = filesystem.geturl(path.as_posix())
        match = re.search(r"[^/]+\.n5/.*$", url, re.IGNORECASE)
        if not match:
            raise UnsupportedUrlException(url)
        name = match.group(0)
        self.filesystem = filesystem.opendir(path.as_posix())

        with self.filesystem.openbin("attributes.json", "r") as f:
            attributes_json = f.read().decode("utf8")
        self.attributes = N5DatasetAttributes.from_json_data(json.loads(attributes_json))

        super().__init__(
            url=url,
            name=name,
            tile_shape=self.attributes.blockSize,
            shape=self.attributes.dimensions,
            dtype=self.attributes.dataType,
            location=location,
            axiskeys=self.attributes.axiskeys,
        )

    def _get_tile(self, tile: Interval5D) -> Array5D:
        slice_address = self.attributes.get_tile_path(tile)
        try:
            with self.filesystem.openbin(slice_address.as_posix()) as f:
                raw_tile = f.read()
            tile_5d = N5Block.from_bytes(
                data=raw_tile, axiskeys=self.axiskeys, dtype=self.dtype, compression=self.attributes.compression
            )
        except ResourceNotFound as e:
            tile_5d = self._allocate(interval=tile, fill_value=0)
        return tile_5d.translated(tile.start)

    def __getstate__(self) -> Dict[str, Any]:
        out = {"path": Path("."), "location": self.location}
        try:
            pickle.dumps(self.filesystem)
            out["filesystem"] = self.filesystem
        except Exception:
            out["filesystem"] = self.filesystem.desc("")
        return out

    def __setstate__(self, data: Dict[str, Any]):
        serialized_filesystem = data["filesystem"]
        if isinstance(serialized_filesystem, str):
            filesystem = open_fs(serialized_filesystem)
        else:
            filesystem = serialized_filesystem
        self.__init__(path=data["path"], location=data["location"], filesystem=filesystem)


DataSource.REGISTRY.insert(0, N5DataSource)
