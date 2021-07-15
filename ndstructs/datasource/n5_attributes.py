from abc import ABC, abstractmethod
from typing import Type, TypeVar

from pathlib import Path
import gzip
import bz2
import lzma
from fs.base import FS as FileSystem
import json
import numpy as np

from ndstructs.point5D import Shape5D
from ndstructs.utils.json_serializable import (
    JsonValue, JsonObject, ensureJsonObject, ensureJsonInt, ensureJsonIntArray, ensureJsonStringArray, ensureJsonString
)
from ndstructs.datasource.DataSource import guess_axiskeys

Compressor = TypeVar("Compressor", bound="N5Compressor")

class N5Compressor(ABC):
    @classmethod
    @abstractmethod
    def get_label(cls) -> str:
        pass

    @classmethod
    @abstractmethod
    def from_json_data(cls: Type[Compressor], data: JsonValue) -> Compressor:
        data_dict = ensureJsonObject(data)
        label = ensureJsonString(data_dict.get("type"))
        if label == GzipCompressor.get_label():
            return GzipCompressor.from_json_data(data)
        if label == Bzip2Compressor.get_label():
            return Bzip2Compressor.from_json_data(data)
        if label == XzCompressor.get_label():
            return XzCompressor.from_json_data(data)
        if label == RawCompressor.get_label():
            return RawCompressor.from_json_data(data)
        raise ValueError(f"Could not interpret {json.dumps(data)} as an n5 compressor")

    @abstractmethod
    def to_json_data(self) -> JsonObject:
        return {"type": self.get_label()}

    @abstractmethod
    def compress(self, raw: bytes) -> bytes:
        pass

    @abstractmethod
    def decompress(self, compressed: bytes) -> bytes:
        pass


class GzipCompressor(N5Compressor):
    def __init__(self, level: int = 1):
        self.level = level

    @classmethod
    def get_label(cls) -> str:
        return "gzip"

    @classmethod
    def from_json_data(cls, data: JsonValue) -> "GzipCompressor":
        return GzipCompressor(
            level=ensureJsonInt(ensureJsonObject(data).get("level", 1))
        )

    def to_json_data(self) -> JsonObject:
        return {
            **super().to_json_data(),
            "level": self.level
        }

    def compress(self, raw: bytes) -> bytes:
        return gzip.compress(raw, compresslevel=self.level)

    def decompress(self, compressed: bytes) -> bytes:
        return gzip.decompress(compressed)


class Bzip2Compressor(N5Compressor):
    def __init__(self, blockSize: int = 9):
        self.blockSize = blockSize

    @classmethod
    def get_label(cls) -> str:
        return "bzip2"

    @classmethod
    def from_json_data(cls, data: JsonValue) -> "Bzip2Compressor":
        return Bzip2Compressor(
            blockSize=ensureJsonInt(ensureJsonObject(data).get("blockSize", 9))
        )

    def to_json_data(self) -> JsonObject:
        return {
            **super().to_json_data(),
            "blockSize": self.blockSize
        }

    def compress(self, raw: bytes) -> bytes:
        return bz2.compress(raw, self.blockSize)

    def decompress(self, compressed: bytes) -> bytes:
        return bz2.decompress(compressed)


class XzCompressor(N5Compressor):
    def __init__(self, preset: int = 6):
        self.preset = preset

    @classmethod
    def get_label(cls) -> str:
        return "xz"

    @classmethod
    def from_json_data(cls, data: JsonValue) -> "XzCompressor":
        return XzCompressor(
            preset=ensureJsonInt(ensureJsonObject(data).get("preset"))
        )

    def to_json_data(self) -> JsonObject:
        return {
            **super().to_json_data(),
            "preset": self.preset
        }

    def compress(self, raw: bytes) -> bytes:
        return lzma.compress(raw, preset=self.preset)

    def decompress(self, compressed: bytes) -> bytes:
        return lzma.decompress(compressed)


class RawCompressor(N5Compressor):
    @classmethod
    def get_label(cls) -> str:
        return "raw"

    @classmethod
    def from_json_data(cls, data: JsonValue) -> "RawCompressor":
        return RawCompressor()

    def to_json_data(self) -> JsonObject:
        return super().to_json_data()

    def compress(self, raw: bytes) -> bytes:
        return raw

    def decompress(self, compressed: bytes) -> bytes:
        return compressed

class N5DatasetAttributes:
    def __init__(self, dimensions: Shape5D, blockSize: Shape5D, axes: str, dataType: np.dtype, compression: N5Compressor):
        self.dimensions = dimensions
        self.blockSize = blockSize
        self.axes = axes
        self.dataType = dataType
        self.compression = compression

    @classmethod
    def load(cls, path: Path, filesystem: FileSystem) -> "N5DatasetAttributes":
        with filesystem.openbin(path.joinpath("attributes.json").as_posix(), "r") as f:
            attributes_json = f.read().decode("utf8")
        raw_attributes = json.loads(attributes_json)
        return cls.from_json_data(raw_attributes)

    @classmethod
    def from_json_data(cls, data: JsonValue) -> "N5DatasetAttributes":
        raw_attributes = ensureJsonObject(data)

        dimensions = ensureJsonIntArray(raw_attributes.get("dimensions"))[::-1]
        blockSize = ensureJsonIntArray(raw_attributes.get("blockSize"))[::-1]
        raw_axiskeys = raw_attributes.get("axes")
        if raw_axiskeys is None:
            axiskeys = guess_axiskeys(dimensions)
        else:
            axiskeys = "".join(ensureJsonStringArray(raw_axiskeys)).lower()[::-1]

        return N5DatasetAttributes(
            blockSize=Shape5D.create(raw_shape=blockSize, axiskeys=axiskeys),
            dimensions=Shape5D.create(raw_shape=dimensions, axiskeys=axiskeys),
            dataType=np.dtype(ensureJsonString(raw_attributes.get("dataType"))).newbyteorder(">"), # type: ignore
            axes=axiskeys,
            compression=N5Compressor.from_json_data(raw_attributes["compression"])
        )

    def to_json_data(self) -> JsonObject:
        return {
            "dimensions": self.dimensions.to_tuple(self.axes)[::-1],
            "blockSize": self.blockSize.to_tuple(self.axes)[::-1],
            "axes": self.axes[::-1],
            "dataType": str(self.dataType.name),
            "compression": self.compression.to_json_data(),
        }
