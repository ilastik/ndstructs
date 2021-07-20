from abc import ABC, abstractmethod
from typing import Tuple, List
from pathlib import Path
import io

import numpy as np
import skimage.io

from ndstructs.utils.json_serializable import (
    JsonValue, JsonObject, ensureJsonObject, ensureJsonString, ensureJsonIntTripplet, ensureJsonArray, ensureJsonInt
)
from ndstructs.datasource.DataSource import DataSource
from ndstructs.point5D import Point5D, Shape5D, Interval5D
from ndstructs.array5D import Array5D

class PrecomputedChunksEncoder(ABC):
    @abstractmethod
    def decode(self, *, roi: Interval5D, dtype: np.dtype, raw_chunk: bytes) -> Array5D:
        pass

    @abstractmethod
    def encode(self, data: Array5D) -> bytes:
        pass

    @abstractmethod
    def to_json_data(self) -> JsonValue:
        pass

    @classmethod
    def from_json_data(cls, data: JsonValue) -> "PrecomputedChunksEncoder":
        label = ensureJsonString(data)
        if label == "raw":
            return RawEncoder()
        if label == "jpeg" or label == "jpg":
            return JpegEncoder()
        raise ValueError(f"Bad encoding value: {label}")

class RawEncoder(PrecomputedChunksEncoder):
    def to_json_data(self) -> JsonValue:
        return "raw"

    def decode(self, *, roi: Interval5D, dtype: np.dtype, raw_chunk: bytes) -> Array5D:
        # "The (...) data (...) chunk is stored directly in little-endian binary format in [x, y, z, channel] Fortran order"
        raw_tile = np.frombuffer(
            raw_chunk,
            dtype=dtype.newbyteorder("<") # type: ignore
        ).reshape(roi.shape.to_tuple("xyzc"), order="F")
        tile_5d = Array5D(raw_tile, axiskeys="xyzc", location=roi.start)
        return tile_5d

    def encode(self, data: Array5D) -> bytes:
        return data.raw("xyzc").tobytes("F")

class JpegEncoder(PrecomputedChunksEncoder):
    def to_json_data(self) -> JsonValue:
        return "jpeg"

    def decode(self, *, roi: Interval5D, dtype: np.dtype, raw_chunk: bytes) -> Array5D:
        # "The width and height of the JPEG image may be arbitrary (...)"
        # "the total number of pixels is equal to the product of the x, y, and z dimensions of the subvolume"
        # "(...) the 1-D array obtained by concatenating the horizontal rows of the image corresponds to the
        # flattened [x, y, z] Fortran-order (i,e. zyx C order) representation of the subvolume."

        # FIXME: check if this works with any sort of funny JPEG shapes
        # FIXME: Also, what to do if dtype is weird?
        raw_jpg: np.ndarray = skimage.io.imread(io.BytesIO(raw_chunk)) # type: ignore
        tile_5d = Array5D(raw_jpg.reshape(roi.shape.to_tuple("zyxc")), axiskeys="zyxc")
        return tile_5d

    def encode(self, data: Array5D) -> bytes:
        raise NotImplementedError


class PrecomputedChunksScale:
    def __init__(
        self,
        key: Path,
        size: Shape5D,
        voxel_size_in_nm: Shape5D,
        voxel_offset: Point5D,
        chunk_sizes: Tuple[Shape5D, ...],
        encoding: PrecomputedChunksEncoder,
    ):
        assert size.t == voxel_size_in_nm.t == 1 and all(cs.t == 1 for cs in chunk_sizes)
        assert voxel_offset.c == 0 and voxel_offset.t == 0, f"Bad voxel_offset: {voxel_offset}"
        assert all(cs.c == size.c for cs in chunk_sizes)

        self.key = key
        self.size = size
        self.voxel_size_in_nm = voxel_size_in_nm
        self.voxel_offset = voxel_offset
        self.chunk_sizes = chunk_sizes
        self.encoding = encoding
        self.interval = self.size.to_interval5d(self.voxel_offset)

    @classmethod
    def from_datasource(
        cls, *, datasource: DataSource, key: Path, voxel_size_in_nm: Shape5D = Shape5D(x=1, y=1, z=1), encoding: PrecomputedChunksEncoder
    ) -> "PrecomputedChunksScale":
        return PrecomputedChunksScale(
            key=key,
            chunk_sizes=tuple([datasource.tile_shape]),
            size=datasource.shape,
            voxel_size_in_nm=voxel_size_in_nm,
            voxel_offset=datasource.location,
            encoding=encoding
        )

    def to_json_data(self) -> JsonObject:
        return {
            "key": self.key.as_posix(),
            "size": self.size.to_tuple("xyz"),
            "resolution": self.voxel_size_in_nm.to_tuple("xyz"),
            "voxel_offset": self.voxel_offset.to_tuple("xyz"),
            "chunk_sizes": tuple(cs.to_tuple("xyz") for cs in self.chunk_sizes),
            "encoding": self.encoding.to_json_data(),
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PrecomputedChunksScale):
            return False
        return (
            self.key == other.key and
            self.size == other.size and
            self.voxel_size_in_nm == other.voxel_size_in_nm and
            self.voxel_offset == other.voxel_offset and
            self.chunk_sizes == other.chunk_sizes and
            self.encoding == other.encoding
        )

    def get_tile_path(self, tile: Interval5D) -> Path:
        assert any(tile.is_tile(tile_shape=cs, full_interval=self.interval, clamped=True) for cs in self.chunk_sizes), f"Bad tile: {tile}"
        return self.key / f"{tile.x[0]}-{tile.x[1]}_{tile.y[0]}-{tile.y[1]}_{tile.z[0]}-{tile.z[1]}"



class PrecomputedChunksInfo:
    def __init__(
        self,
        *,
        type_: str,
        data_type: np.dtype,
        num_channels: int,
        scales: Tuple[PrecomputedChunksScale, ...],
    ):
        self.type_ = type_
        self.data_type = data_type
        self.num_channels = num_channels
        self.scales = scales

        if self.type_ != "image":
            raise NotImplementedError(f"Don't know how to interpret type '{self.type_}'")
        if num_channels <= 0:
            raise ValueError("num_channels must be greater than 0", self.__dict__)
        if len(scales) == 0:
            raise ValueError("Must provide at least one scale", self.__dict__)

    def get_scale(self, voxel_size_in_nm: Shape5D) -> PrecomputedChunksScale:
        for scale in self.scales:
            if scale.voxel_size_in_nm == voxel_size_in_nm:
                return scale
        raise ValueError(f"Scale with resolution {voxel_size_in_nm} not found")

    def contains(self, scale: PrecomputedChunksScale) -> bool:
        return any(scale == s for s in self.scales)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, PrecomputedChunksInfo) and
            self.type_ == other.type_ and
            self.data_type == other.data_type and
            self.num_channels == other.num_channels and
            self.scales == other.scales
        )

    @classmethod
    def from_json_data(cls, data: JsonValue):
        data_dict = ensureJsonObject(data)
        num_channels = ensureJsonInt(data_dict.get("num_channels"))
        raw_scales = ensureJsonArray(data_dict.get("scales"))
        scales: List[PrecomputedChunksScale] = []
        for raw_scale in raw_scales:
            scale_dict = ensureJsonObject(raw_scale)
            key = ensureJsonString(scale_dict.get("key"))
            size = ensureJsonIntTripplet(scale_dict.get("size"))
            resolution = ensureJsonIntTripplet(scale_dict.get("resolution"))
            voxel_offset = ensureJsonIntTripplet(scale_dict.get("voxel_offset"))
            chunk_sizes = [ensureJsonIntTripplet(cs) for cs in ensureJsonArray(scale_dict.get("chunk_sizes"))]

            scales.append(PrecomputedChunksScale(
                key=Path(key),
                size=Shape5D(x=size[0], y=size[1], z=size[2], c=num_channels),
                voxel_size_in_nm=Shape5D(x=resolution[0], y=resolution[1], z=resolution[2], c=num_channels),
                voxel_offset=Point5D.zero(x=voxel_offset[0], y=voxel_offset[1], z=voxel_offset[2]),
                chunk_sizes=tuple(Shape5D(x=cs[0], y=cs[1], z=cs[2], c=num_channels) for cs in chunk_sizes),
                encoding=PrecomputedChunksEncoder.from_json_data(scale_dict.get("encoding")),
            ))

        return PrecomputedChunksInfo(
            type_=ensureJsonString(data_dict.get("type")),
            data_type=np.dtype(ensureJsonString(data_dict.get("data_type"))),
            num_channels=num_channels,
            scales=tuple(scales)
        )

    def to_json_data(self) -> JsonObject:
        return {
            "@type": "neuroglancer_multiscale_volume",
            "type": self.type_,
            "data_type": str(self.data_type.name),
            "num_channels": self.num_channels,
            "scales": tuple(scale.to_json_data() for scale in self.scales),
        }
