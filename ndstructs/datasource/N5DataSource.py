from typing import Optional, Union
from pathlib import Path
import enum
import re
import json
import pickle
from typing_extensions import TypedDict

import numpy as np
from fs import open_fs
from fs.base import FS
from fs.errors import ResourceNotFound
from fs import open_fs

from ndstructs.datasource.n5_attributes import N5Compressor, N5DatasetAttributes
from ndstructs import Point5D, Interval5D, Array5D
from ndstructs.datasource.DataSource import DataSource
from .UnsupportedUrlException import UnsupportedUrlException

class N5Block(Array5D):
    class Modes(enum.IntEnum):
        DEFAULT = 0
        VARLENGTH = 1

    @classmethod
    def from_bytes(cls, data: bytes, axiskeys: str, dtype: np.dtype, compression: N5Compressor, location: Point5D):
        data = np.frombuffer(data, dtype=np.uint8)

        header_types = [
            ("mode", ">u2"),  # mode (uint16 big endian, default = 0x0000, varlength = 0x0001)
            ("num_dims", ">u2"),  # number of dimensions (uint16 big endian)
        ]
        preamble = np.frombuffer(data, dtype=header_types, count=1)
        header_types.append(
              # dimension 1[,...,n] (uint32 big endian)
            ("dimensions", str(preamble["num_dims"].item()) + ">u4") # type: ignore
        )

        if preamble["mode"].item() == cls.Modes.VARLENGTH.value:
            # mode == varlength ? number of elements (uint32 big endian)
            header_types.append(("num_elements", ">u4")) # type: ignore
            raise RuntimeError("Don't know how to handle varlen N5 blocks")

        header_dtype = np.dtype(header_types)
        header_data = np.frombuffer(data, dtype=header_dtype, count=1)
        array_shape = header_data["dimensions"].squeeze()

        compressed_buffer = np.frombuffer(data, offset=header_dtype.itemsize, dtype=np.uint8)
        decompressed_buffer = compression.decompress(compressed_buffer.tobytes())
        raw_array = np.frombuffer(decompressed_buffer, dtype=dtype.newbyteorder(">")).reshape(array_shape, order="F") # type: ignore

        return cls(raw_array, axiskeys=axiskeys[::-1], location=location)

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


class SerializedN5Datasource(TypedDict):
    path: Path
    location: Point5D
    filesystem: Union[str, FS]

class N5DataSource(DataSource):
    """A DataSource representing an N5 dataset. "axiskeys" are, like everywhere else in ndstructs, C-ordered."""

    def __init__(self, path: Path, *, location: Optional[Point5D] = None, filesystem: FS):
        url = filesystem.geturl(path.as_posix())
        match = re.search(r"[^/]+\.n5/.*$", url, re.IGNORECASE)
        if not match:
            raise UnsupportedUrlException(url)
        name = match.group(0)
        self.path = path
        self.filesystem = filesystem

        with self.filesystem.openbin(path.joinpath("attributes.json").as_posix(), "r") as f:
            attributes_json = f.read().decode("utf8")
        self.attributes = N5DatasetAttributes.from_json_data(json.loads(attributes_json), location_override=location)

        super().__init__(
            url=url,
            name=name,
            tile_shape=self.attributes.blockSize,
            shape=self.attributes.dimensions,
            dtype=self.attributes.dataType,
            location=self.attributes.location,
            axiskeys=self.attributes.axiskeys,
        )

    def _get_tile(self, tile: Interval5D) -> Array5D:
        slice_address = self.path / self.attributes.get_tile_path(tile)
        try:
            with self.filesystem.openbin(slice_address.as_posix()) as f:
                raw_tile = f.read()
            tile_5d = N5Block.from_bytes(
                data=raw_tile, axiskeys=self.axiskeys, dtype=self.dtype, compression=self.attributes.compression, location=tile.start
            )
        except ResourceNotFound:
            tile_5d = self._allocate(interval=tile, fill_value=0)
        return tile_5d

    def __getstate__(self) -> SerializedN5Datasource:
        try:
            pickle.dumps(self.filesystem)
            filesystem = self.filesystem
        except Exception:
            filesystem = self.filesystem.desc("")
        return SerializedN5Datasource(
            path=self.path,
            location=self.location,
            filesystem=filesystem
        )

    def __setstate__(self, data: SerializedN5Datasource):
        serialized_filesystem = data["filesystem"]
        if isinstance(serialized_filesystem, str):
            filesystem: FS = open_fs(serialized_filesystem)
        else:
            filesystem: FS = serialized_filesystem
        self.__init__(path=data["path"], location=data["location"], filesystem=filesystem)
