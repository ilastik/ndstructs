from typing import Union, Optional, Callable
from pathlib import Path
import urllib.parse
import enum
from functools import partial

import json
import numpy as np
import z5py

from ndstructs import Point5D, Shape5D, Slice5D, Array5D

from ndstructs.datasource.DataSourceUrl import DataSourceUrl
from ndstructs.datasource.DataSource import DataSource, guess_axiskeys
from .UnsupportedUrlException import UnsupportedUrlException
from ndstructs.datasource.BackedSlice5D import BackedSlice5D


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
        array_axiskeys_f = on_disk_axiskeys[::-1]  # reversed because bytes are stored in F order
        return cls(raw_array.reshape(array_shape_f), axiskeys=array_axiskeys_f)


class N5DataSource(DataSource):
    def __init__(
        self, url: Union[Path, str], *, tile_shape: Optional[Shape5D] = None, location: Point5D = Point5D.zero()
    ):
        url = str(url)
        if ".n5" not in url:
            raise UnsupportedUrlException(url)
        self.outer_path = url.split(".n5")[0] + ".n5"
        self.inner_path = url.split(".n5")[1]
        if not self.inner_path:
            raise ValueError(f"{url} does not have an inner path")
        if tile_shape:
            print(f"Warning!!! Ignoring tile_shape hint {tile_shape}")

        attributes_json_bytes = DataSourceUrl.fetch_bytes(urllib.parse.urljoin(url + "/", "attributes.json"))
        attributes = json.loads(attributes_json_bytes.decode("utf8"))

        dimensions = attributes["dimensions"]
        blockSize = attributes["blockSize"]
        axiskeys = "".join(attributes["axes"]).lower() or guess_axiskeys(dimensions[::-1])[::-1]
        if not (len(dimensions) == len(blockSize) == len(axiskeys)):
            raise ValueError("Shape/axis mismatch: {json.dumps(attributes, indent=4)}")

        super().__init__(
            url,
            tile_shape=Shape5D(**{axis: length for axis, length in zip(axiskeys, blockSize)}),
            shape=Shape5D(**{axis: length for axis, length in zip(axiskeys, dimensions)}),
            dtype=np.dtype(attributes["dataType"]).newbyteorder(">"),
            axiskeys=axiskeys[::-1],  # axiskeys outside is always C-order
            name=self.outer_path.split("/")[-1] + self.inner_path,
            location=location,
        )
        compression_type = attributes["compression"]["type"]
        if compression_type == "gzip":
            import gzip

            self.decompressor = gzip.decompress
            level = attributes["compression"].get("level", 1)
            self.compressor = partial(gzip.compress, compresslevel=level)
        elif compression_type == "raw":
            noop = lambda data: data
            self.decompressor = noop
            self.compressor = noop
        else:
            raise NotImplementedError(f"Don't know how to decompress {compression_type}")

    def _get_tile(self, tile: Slice5D) -> Array5D:
        # get axiskeys back to F order
        on_disk_axiskeys = self.axiskeys[::-1]
        slice_address_components = (tile.start // self.tile_shape).to_tuple(on_disk_axiskeys)
        slice_address = "/".join(str(int(comp)) for comp in slice_address_components)
        full_path = urllib.parse.urljoin(self.url + "/", slice_address)
        # import pydevd; pydevd.settrace()
        raw_tile = DataSourceUrl.fetch_bytes(full_path)
        tile_5d = N5Block.from_bytes(
            data=raw_tile, on_disk_axiskeys=on_disk_axiskeys, dtype=self.dtype, decompressor=self.decompressor
        )
        return tile_5d.translated(tile.start)


DataSource.REGISTRY.insert(0, N5DataSource)
