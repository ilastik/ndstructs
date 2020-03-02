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
from ndstructs.utils import JsonSerializable


class BadPrecomputedChunksInfo(Exception):
    def __init__(self, message: str, info: Dict):
        super().__init__(message + "\n\n" + json.dumps(info, indent=4))


class PrecomputedChunksScale(JsonSerializable):
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
    def from_json_data(cls, data: Dict[str, Any]):
        if "@type" in data:
            data["at_type"] = data.pop("@type")
        if "type" in data:
            data["type_"] = data.pop("type")
        return super().from_json_data(data)

    @classmethod
    def from_url(cls, url: Union[str, Path], filesystem: Optional[FS] = None) -> "PrecomputedChunksInfo":
        url = Url.parse(url)
        if url.path_name != "info":
            raise ValueError("PrecomputedChunksInfo url should end with '/info'")
        info_dir_path = url.parent.geturl()
        filesystem = filesystem.opendir(info_dir_path) if filesystem else open_fs(info_dir_path)
        with filesystem.openbin("info") as f:
            info_json_text = f.read().decode("utf-8")
        return cls.from_json(info_json_text)

    def get_scale(self, key: str) -> PrecomputedChunksScale:
        for s in self.scales:
            if s.key == key:
                return s
        raise KeyError(key)

    def deserializeChunkSize(self, chunk_size: str) -> Shape5D:
        size_values = map(int, chunk_size.split("_")) + [self.num_channels]
        size_keys = "xyz"[: len(size_values)] + "c"
        return Shape5D(**dict(zip(size_keys, size_values)))


class PrecomputedChunksDataSource(DataSource):
    def __init__(self, url: Union[Path, str], *, location: Point5D = Point5D.zero(), filesystem: Optional[FS] = None):
        """A DataSource that reads Neuroglancer's Precomputed Chunks.

          url: url into the chosen scale
            if, e.g. there is a info json at:
                http://exampe.com/something/info
            that looks like this:
            {
                ...
                "scales": [
                    ...
                    {
                        "key": "my_scale"
                    }
                ]
            }
            then the url should provided to this Data Source should be:
                http://example.com/something/my_scale?chunk_size=100_200_50

            which will also select a chunk size with x=100 y=200 z=50 (if available),
        """

        scale_url = Url.parse(url)
        base_url: str = scale_url.parent.geturl()
        self.filesystem = filesystem.opendir(base_url) if filesystem else open_fs(base_url)
        self.info = PrecomputedChunksInfo.from_url(url="info", filesystem=self.filesystem)
        if "chunk_size" in scale_url.query_dict:
            tile_shape_hint = self.info.deserializeChunkSize(scale_url.query_dict["chunk_size"])
        else:
            tile_shape_hint = None
        self.scale = self.info.get_scale(key=scale_url.path_name)
        super().__init__(
            url,
            tile_shape=self.scale.get_tile_shape_5d(self.info.num_channels, tile_shape_hint=tile_shape_hint),
            shape=self.scale.get_shape_5d(self.info.num_channels),
            dtype=self.info.data_type,
            name=scale_url.path_name,
            location=location,
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
        raw_tile_fortran_shape = tile.shape.to_tuple(self.scale.axiskeys[::-1])
        raw_tile = np.frombuffer(raw_tile_bytes, dtype=self.dtype).reshape(raw_tile_fortran_shape)
        tile_5d = Array5D(raw_tile, axiskeys=self.scale.axiskeys[::-1])
        return tile_5d.translated(tile.start)


# DataSource.REGISTRY.insert(0, PrecomputedChunksDataSource)
