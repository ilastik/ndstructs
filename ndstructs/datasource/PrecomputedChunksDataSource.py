from typing import Optional, Tuple, Union, cast
from pathlib import Path
import pickle
from typing_extensions import TypedDict
import json

from fs import open_fs
from fs.base import FS as FileSystem

from ndstructs import Point5D, Shape5D, Interval5D, Array5D
from ndstructs.datasource.DataSource import DataSource
from ndstructs.datasource.precomputed_chunks_info import PrecomputedChunksInfo

class SerializedPrecomputedChunksDatasource(TypedDict):
    path: Path
    voxel_size_in_nm: Shape5D
    location: Point5D
    chunk_size: Shape5D
    filesystem: Union[str, FileSystem]


class PrecomputedChunksDataSource(DataSource):
    def __init__(
        self,
        *,
        path: Path,
        voxel_size_in_nm: Shape5D,
        location: Optional[Point5D] = None,
        chunk_size: Optional[Shape5D] = None,
        filesystem: FileSystem,
    ):
        self.path = path
        self.filesystem = filesystem
        with self.filesystem.openbin(path.joinpath("info").as_posix(), "r") as f:
            info_json = f.read().decode("utf8")
        self.info = PrecomputedChunksInfo.from_json_data(json.loads(info_json))
        self.scale = self.info.get_scale(voxel_size_in_nm=voxel_size_in_nm)

        if chunk_size:
            if chunk_size not in self.scale.chunk_sizes:
                raise ValueError(f"Bad chunk size: {chunk_size}. Available are: {self.scale.chunk_sizes}")
            tile_shape = chunk_size
        else:
            tile_shape = self.scale.chunk_sizes[0]

        super().__init__(
            url="precomputed://" + filesystem.desc(path.joinpath(self.scale.key).as_posix()), # FIXME
            tile_shape=tile_shape,
            shape=self.scale.size,
            dtype=self.info.data_type,
            name=self.scale.key.name,
            location=location or self.scale.voxel_offset,
            axiskeys="zyxc",  # externally reported axiskeys are always c-ordered
            spatial_resolution=cast(Tuple[int, int, int], self.scale.voxel_size_in_nm.to_tuple("xyz")) # FIXME
        )

    def _get_tile(self, tile: Interval5D) -> Array5D:
        if self.location != self.scale.voxel_offset:
            tile = tile.translated(-self.location).translated(self.scale.voxel_offset)
        tile_path = self.path / self.scale.get_tile_path(tile)
        with self.filesystem.openbin(tile_path.as_posix()) as f:
            raw_tile_bytes = f.read()
        tile_5d = self.scale.encoding.decode(roi=tile, dtype=self.dtype, raw_chunk=raw_tile_bytes)
        return tile_5d

    def __getstate__(self) -> SerializedPrecomputedChunksDatasource:
        try:
            pickle.dumps(self.filesystem)
            filesystem = self.filesystem
        except Exception:
            filesystem = self.filesystem.desc("")
        return SerializedPrecomputedChunksDatasource(
            chunk_size=self.tile_shape,
            filesystem=filesystem,
            location=self.location,
            path=self.path,
            voxel_size_in_nm=self.scale.voxel_size_in_nm
        )

    def __setstate__(self, data: SerializedPrecomputedChunksDatasource):
        serialized_filesystem = data["filesystem"]
        if isinstance(serialized_filesystem, str):
            filesystem: FileSystem = open_fs(serialized_filesystem)
        else:
            filesystem: FileSystem = serialized_filesystem

        self.__init__(
            path=data["path"],
            voxel_size_in_nm=data["voxel_size_in_nm"],
            location=data["location"],
            chunk_size=data["chunk_size"],
            filesystem=filesystem
        )
