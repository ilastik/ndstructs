from typing import Union, Optional, Callable, Any, Iterator, Dict, List
from pathlib import Path
import urllib.parse
import enum
from functools import partial
import gzip

import json
import numpy as np
from fs import open_fs
from fs.base import FS

from ndstructs import Point5D, Shape5D, Slice5D, Array5D

from ndstructs.datasource.DataSourceUrl import DataSourceUrl
from ndstructs.datasource.DataSource import DataSource
from .UnsupportedUrlException import UnsupportedUrlException
from ndstructs.datasource.DataSourceSlice import DataSourceSlice
from ndstructs.datasource.DataSourceUrl import Url
from ndstructs.utils import JsonSerializable, Dereferencer, Referencer


class BadPrecomputedChunksInfo(Exception):
    def __init__(self, message: str, info: Dict):
        super().__init__(message + "\n\n" + json.dumps(info, indent=4))


class PrecomputedChunksScale(JsonSerializable):
    """An object reporesenting a Precomputed Chunks Scale

    All ordered tuples, list and axiskeys are in fortran order, as per spec"""

    def __init__(
        self,
        key: str,
        size: List[int],
        resolution: List[int],
        voxel_offset: List[int],
        chunk_sizes: List[List[int]],
        encoding: str,
    ):
        self.key = key
        self.size = size
        self.resolution = resolution
        self.voxel_offset = voxel_offset
        self.chunk_sizes = chunk_sizes
        self.encoding = encoding

        chunk_size_lengths = set(len(cs) for cs in chunk_sizes)
        lengths = {len(size), len(resolution), len(voxel_offset)}.union(chunk_size_lengths)
        if len(lengths) != 1:
            raise BadPrecomputedChunksInfo("Missmatching lengths!", self.__dict__)
        self.spatial_axiskeys = "xyz"[: len(size)]
        self.axiskeys = self.spatial_axiskeys + "c"

    def get_shape_5d(self, num_channels: int) -> Shape5D:
        return Shape5D(**dict(zip(self.axiskeys, self.size + [num_channels])))

    def get_chunk_sizes_5d(self, num_channels: int) -> List[Shape5D]:
        return [Shape5D(**dict(zip(self.axiskeys, cs + [num_channels]))) for cs in self.chunk_sizes]

    def get_tile_shape_5d(self, num_channels: int, tile_shape_hint: Optional[Shape5D] = None) -> Shape5D:
        valid_chunk_shapes: List[Shape5D] = self.get_chunk_sizes_5d(num_channels)
        tile_shape_hint = tile_shape_hint if tile_shape_hint is not None else valid_chunk_shapes[0]
        if tile_shape_hint not in valid_chunk_shapes:
            raise ValueError("{tile_shape_hint} is not a valid chunk size for {json.dumps(self.__dict__, indent=4)}")
        return tile_shape_hint


class PrecomputedChunksInfo(JsonSerializable):
    def __init__(
        self,
        *,
        at_type: str = "neuroglancer_multiscale_volume",
        type_: str,
        data_type: np.dtype,
        num_channels: int,
        scales: List[PrecomputedChunksScale],
    ):
        self.at_type = at_type
        self.type_ = type_
        self.data_type = data_type
        self.num_channels = num_channels
        self.scales = scales

        if self.at_type != "neuroglancer_multiscale_volume":
            raise BadPrecomputedChunksInfo("@type should be 'neuroglancer_multiscale_volume", self.__dict__)
        if self.type_ != "image":
            raise NotImplementedError(f"Don't know how to interpret type '{self.type_}'")
        if num_channels <= 0:
            raise BadPrecomputedChunksInfo("num_channels must be greater than 0", self.__dict__)
        if len(scales) == 0:
            raise BadPrecomputedChunksInfo("Must provide at least one scale", self.__dict__)

        self.spatial_axiskeys = scales[0].spatial_axiskeys
        self.axiskeys = self.spatial_axiskeys + "c"

    @classmethod
    def from_json_data(cls, data: Dict[str, Any], dereferencer: Dereferencer):
        if "@type" in data:
            data["at_type"] = data.pop("@type")
        if "type" in data:
            data["type_"] = data.pop("type")
        return super().from_json_data(data, dereferencer=dereferencer)

    @classmethod
    def load(cls, path: Path, filesystem: Optional[FS] = None) -> "PrecomputedChunksInfo":
        filesystem = filesystem or OSFS(path.anchor)
        if path.name != "info":
            raise ValueError("PrecomputedChunksInfo url should end with '/info'")
        with filesystem.openbin(path.as_posix()) as f:
            info_json_text = f.read().decode("utf-8")
        return cls.from_json(info_json_text)

    def get_scale(self, key: str) -> PrecomputedChunksScale:
        for s in self.scales:
            if s.key == key:
                return s
        raise KeyError(key)


class PrecomputedChunksDataSource(DataSource):
    def __init__(
        self, path: Path, *, location: Point5D = Point5D.zero(), chunk_size: Optional[Shape5D] = None, filesystem: FS
    ):
        """A DataSource that handles Neurogancer's precomputed chunks

        path: a path all the pay down to the scale, i.e., if some scale has
                "key": "my_scale"
              then your path should end in "my_scale"
        chunk_size: a valid chunk_size for the scale selected by 'path'
        """
        self.filesystem = filesystem.opendir(path.parent.as_posix())
        self.info = PrecomputedChunksInfo.load(path=Path("info"), filesystem=self.filesystem)
        self.scale = self.info.get_scale(key=path.name)
        super().__init__(
            url="precomputed://" + filesystem.desc(path.as_posix()),
            tile_shape=self.scale.get_tile_shape_5d(self.info.num_channels, tile_shape_hint=chunk_size),
            shape=self.scale.get_shape_5d(self.info.num_channels),
            dtype=self.info.data_type,
            name=path.name,
            location=location,
            axiskeys=self.scale.axiskeys[::-1],  # externally reported axiskeys are always c-ordered
        )
        encoding_type = self.scale.encoding
        if encoding_type == "raw":
            noop = lambda data: data
            self.decompressor = noop
            self.compressor = noop
        else:
            raise NotImplementedError(f"Don't know how to decompress {compression_type}")

    def _get_tile(self, tile: Slice5D) -> Array5D:
        slice_address = "_".join(f"{s.start}-{s.stop}" for s in tile.to_slices(self.scale.spatial_axiskeys))
        path = self.scale.key + "/" + slice_address
        with self.filesystem.openbin(path) as f:
            raw_tile_bytes = f.read()
        raw_tile_c_shape = tile.shape.to_tuple(self.axiskeys)
        raw_tile = np.frombuffer(raw_tile_bytes, dtype=self.dtype).reshape(raw_tile_c_shape)
        tile_5d = Array5D(raw_tile, axiskeys=self.axiskeys)
        return tile_5d.translated(tile.start)


# DataSource.REGISTRY.insert(0, PrecomputedChunksDataSource)
