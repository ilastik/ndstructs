from logging import root
from ndstructs.datasource.PrecomputedChunksDataSource import PrecomputedChunksDataSource, PrecomputedChunksInfo, PrecomputedChunksScale
from typing import Optional, Union
import re
from pathlib import Path
import json
import enum

import numpy as np
from fs.base import FS as FileSystem
from fs.osfs import OSFS

from ndstructs.point5D import Point5D, Interval5D, Shape5D
from ndstructs.array5D import Array5D
from ndstructs.datasource.DataSource import DataSource, UnsupportedUrlException
from ndstructs.datasource.N5DataSource import N5Block
from ndstructs.datasource.DataRoi import DataRoi
from ndstructs.datasink.DataSink import DataSink


class PrecomputedChunksDataSink:
    # @privatemethod
    def __init__(
        self, *, root_path: Path, filesystem: FileSystem, info: PrecomputedChunksInfo
    ):
        self.root_path = root_path
        self.filesystem = filesystem
        self.info = info

    @classmethod
    def create(
        cls,
        *,
        root_path: Path,
        filesystem: FileSystem,
        info: PrecomputedChunksInfo,
    ) -> "PrecomputedChunksDataSink":
        if filesystem.exists(root_path.as_posix()):
            filesystem.removedir(root_path.as_posix())
        filesystem.makedirs(root_path.as_posix())
        with filesystem.openbin(root_path.joinpath("info").as_posix(), "w") as info_file:
            info_file.write(json.dumps(info.to_json_data()).encode("utf8"))
        for scale in info.scales:
            filesystem.makedirs(root_path.joinpath(scale.key).as_posix())
        return PrecomputedChunksDataSink(root_path=root_path, filesystem=filesystem, info=info)

    @classmethod
    def open_(cls, root_path: Path, filesystem: FileSystem) -> "PrecomputedChunksDataSink":
        info = PrecomputedChunksInfo.load(path=root_path.joinpath("info"), filesystem=filesystem)
        return PrecomputedChunksDataSink(root_path=root_path, filesystem=filesystem, info=info)

    def write(self, scale: PrecomputedChunksScale, chunk: Array5D):
        assert(self.info.contains(scale))
        interval = chunk.interval
        chunk_name = f"{interval.x[0]}-{interval.x[1]}_{interval.y[0]}-{interval.y[1]}_{interval.z[0]}-{interval.z[1]}"
        chunk_path = self.root_path / scale.key / chunk_name
        # https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed#raw-chunk-encoding
        # "(...) data for the chunk is stored directly in little-endian binary format in [x, y, z, channel] Fortran order"
        chunk_bytes = chunk.raw("xyzc").tobytes("F")
        with self.filesystem.openbin(chunk_path.as_posix(), "w") as f:
            f.write(chunk_bytes)


class PrecomputedChunksScaleDataSink:
    # @privatemethod
    def __init__(self, *, num_channels: int, scale: PrecomputedChunksScale, path: Path, filesystem: FileSystem):
        self.num_channels = num_channels
        self.scale = scale
        self.chunk_shapes = set(self.scale.get_chunk_shapes(num_channels))
        self.path = path
        self.filesystem = filesystem

    def write(self, chunk: Array5D):
        assert chunk.shape in self.chunk_shapes, f""
        interval = chunk.interval
        chunk_name = f"{interval.x[0]}-{interval.x[1]}_{interval.y[0]}-{interval.y[1]}_{interval.z[0]}-{interval.z[1]}"
        chunk_path = self.path / chunk_name
        # https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed#raw-chunk-encoding
        # "(...) data for the chunk is stored directly in little-endian binary format in [x, y, z, channel] Fortran order"
        chunk_bytes = chunk.raw("xyzc").tobytes("F")
        with self.filesystem.openbin(chunk_path.as_posix(), "w") as f:
            f.write(chunk_bytes)
