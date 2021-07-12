from typing import Optional, Any, Dict, List, Callable, Tuple
from pathlib import Path
import pickle
import io

import json
import numpy as np
from fs import open_fs
from fs.base import FS
from fs.osfs import OSFS
import skimage.io

from ndstructs import Point5D, Shape5D, Interval5D, Array5D

from ndstructs.datasource.DataSource import DataSource
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

    def get_shape(self, c: int) -> Shape5D:
        return Shape5D(x=self.size[0], y=self.size[1], z=self.size[2], c=c)

    def get_chunk_shapes(self, c: int) -> List[Shape5D]:
        return [
            Shape5D(x=chunk_size[0], y=chunk_size[1], z=chunk_size[2], c=c)
            for chunk_size in self.chunk_sizes
        ]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PrecomputedChunksScale):
            return False
        return (
            self.key == other.key and
            self.size == other.size and
            self.resolution == other.resolution and
            self.voxel_offset == other.voxel_offset and
            self.chunk_sizes == other.chunk_sizes and
            self.encoding == other.encoding
        )

    @classmethod
    def from_datasource(cls, key: str, resolution: Tuple[int, int, int], datasource: DataSource) -> "PrecomputedChunksScale":
        return PrecomputedChunksScale(
            key=key,
            size=list(datasource.shape.to_tuple("xyz")),
            resolution=list(resolution),
            voxel_offset=list(datasource.location.to_tuple("xyz")),
            chunk_sizes=[list(datasource.tile_shape.to_tuple("xyz"))],
            encoding="raw",
        )

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

    def contains(self, scale: PrecomputedChunksScale) -> bool:
        return any(scale == s for s in self.scales)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PrecomputedChunksInfo):
            return False
        return (
            self.at_type == other.at_type and
            self.type_ == other.type_ and
            self.data_type == other.data_type and
            self.num_channels == other.num_channels and
            self.scales == other.scales
        )

    @classmethod
    def from_datasource(cls, scale_key: str, resolution: Tuple[int, int, int], datasource: DataSource) -> "PrecomputedChunksInfo":
        return PrecomputedChunksInfo(
            type_="image",
            data_type=datasource.dtype,
            num_channels=datasource.shape.c,
            scales=[
                PrecomputedChunksScale.from_datasource(
                    key=scale_key, resolution=resolution, datasource=datasource
                )
            ],
        )

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

    def to_json_data(self, referencer: Optional[Referencer] = None):
        return {
            "@type": self.at_type,
            "type": self.type_,
            "data_type": self.data_type.name,
            "num_channels": self.num_channels,
            "scales": [scale.to_json_data() for scale in self.scales],
        }


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

        chunk_sizes_5d = [Shape5D(x=cs[0], y=cs[1], z=cs[2], c=self.info.num_channels) for cs in self.scale.chunk_sizes]
        if chunk_size:
            if chunk_size not in chunk_sizes_5d:
                raise ValueError(f"Bad chunk size: {chunk_size}. Availabel are: {chunk_sizes_5d}")
            tile_shape = chunk_size
        else:
            tile_shape = chunk_sizes_5d[0]

        super().__init__(
            url="precomputed://" + filesystem.desc(path.as_posix()),
            tile_shape=tile_shape,
            shape=Shape5D(x=self.scale.size[0], y=self.scale.size[1], z=self.scale.size[2], c=self.info.num_channels),
            dtype=self.info.data_type,
            name=path.name,
            location=location,
            axiskeys="zyxc",  # externally reported axiskeys are always c-ordered
            spatial_resolution=(self.scale.resolution[0], self.scale.resolution[1], self.scale.resolution[2])
        )
        encoding_type = self.scale.encoding
        self.decompressor: Callable[[Interval5D, bytes], Array5D]
        if encoding_type == "raw":
            self.decompressor = self.decompress_raw_chunk
        elif encoding_type == "jpeg":
            self.decompressor = self.decompress_jpeg_chunk
        else:
            raise NotImplementedError(f"Don't know how to decompress {encoding_type}")

    def decompress_jpeg_chunk(self, roi: Interval5D, raw_chunk: bytes) -> Array5D:
        # "The width and height of the JPEG image may be arbitrary (...)"
        # "the total number of pixels is equal to the product of the x, y, and z dimensions of the subvolume"
        # "(...) the 1-D array obtained by concatenating the horizontal rows of the image corresponds to the
        # flattened [x, y, z] Fortran-order (i,e. zyx C order) representation of the subvolume."

        # FIXME: check if this works with any sort of funny JPEG shapes
        raw_jpg = skimage.io.imread(io.BytesIO(raw_chunk))
        tile_5d = Array5D(raw_jpg.reshape(roi.shape.to_tuple("zyxc")), axiskeys="zyxc")
        return tile_5d

    def decompress_raw_chunk(self, roi: Interval5D, raw_chunk: bytes) -> Array5D:
        # "The (...) data (...) chunk is stored directly in little-endian binary format in [x, y, z, channel] Fortran order"
        raw_tile = np.frombuffer(raw_chunk, dtype=self.dtype).reshape(roi.shape.to_tuple("xyzc"), order="F")
        tile_5d = Array5D(raw_tile, axiskeys="xyzc")
        return tile_5d

    def _get_tile(self, tile: Interval5D) -> Array5D:
        slice_address = f"/{tile.x[0]}-{tile.x[1]}_{tile.y[0]}-{tile.y[1]}_{tile.z[0]}-{tile.z[1]}"
        path = self.scale.key.rstrip("/") + slice_address
        with self.filesystem.openbin(path) as f:
            raw_tile_bytes = f.read()
        tile_5d = self.decompressor(tile, raw_tile_bytes)
        return tile_5d.translated(tile.start)

    def __getstate__(self) -> Dict[str, Any]:
        out = {"path": Path(self.scale.key), "location": self.location}
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


# DataSource.REGISTRY.insert(0, PrecomputedChunksDataSource)
