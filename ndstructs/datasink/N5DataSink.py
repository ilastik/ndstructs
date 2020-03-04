from typing import Optional, Union
import re
from pathlib import Path
import json
import enum

import z5py
import numpy as np
from fs.base import FS
from fs.osfs import OSFS

from ndstructs.point5D import Point5D, Slice5D, Shape5D
from ndstructs.array5D import Array5D
from ndstructs.datasource.DataSource import DataSource, UnsupportedUrlException
from ndstructs.datasource.N5DataSource import N5Block
from ndstructs.datasource.DataSourceSlice import DataSourceSlice
from ndstructs.datasink.DataSink import DataSink


class N5DataSink(DataSink):
    N5_SIGNATURE = re.compile(r"\w\.n5/\w", re.IGNORECASE)
    N5_PATH_SPLITTER = re.compile(r"(\.n5)\b", re.IGNORECASE)
    N5_ROOT_ATTRIBUTES = json.dumps({"n5": "2.0.0"}).encode("utf8")

    class Mode(enum.Enum):
        CREATE = "create"
        OPEN = "open"

    def __init__(
        self,
        *,
        path: Path,  # dataset path, e.g. "mydata.n5/mydataset"
        data_slice: DataSourceSlice,
        axiskeys: str = "tzyxc",
        compression_type: str = "raw",
        tile_shape: Optional[Shape5D] = None,
        filesystem: Optional[FS] = None,
        mode: Mode = Mode.CREATE,
    ):
        super().__init__(data_slice=data_slice, tile_shape=tile_shape)
        if not set(data_slice.shape.present_spatial_axes.keys()).issubset(set(axiskeys)):
            raise ValueError(f"Cannot represent data source {data_slice} using axiskeys '{axiskeys}'")
        if not self.N5_SIGNATURE.search(path.as_posix()):
            raise UnsupportedUrlException(path)
        self.axiskeys = axiskeys
        self.compression_type = compression_type

        if mode == self.Mode.OPEN:
            self.filesystem = filesystem.opendir(path.as_posix()) if filesystem else OSFS(path.as_posix())
            return

        path_components = self.N5_PATH_SPLITTER.split(path.as_posix())
        outer_path = "".join(path_components[0:2])
        inner_path = "".join(path_components[2:])

        root_fs = filesystem or OSFS(path.anchor)
        n5_root_fs = root_fs.makedirs(outer_path, recreate=True)
        if not n5_root_fs.isfile("attributes.json"):
            with n5_root_fs.openbin("attributes.json", "w") as f:
                f.write(self.N5_ROOT_ATTRIBUTES)

        self.filesystem = n5_root_fs.makedirs(inner_path, recreate=True)
        attributes = {
            "dimensions": self.data_slice.shape.to_tuple(axiskeys[::-1]),
            "blockSize": self.tile_shape.to_tuple(axiskeys[::-1]),
            "axes": list(self.axiskeys[::-1]),
            "dataType": str(data_slice.dtype),
            "compression": {"type": self.compression_type},
        }
        with self.filesystem.openbin("attributes.json", "w") as f:
            f.write(json.dumps(attributes).encode("utf-8"))

        # create all directories in the constructor to avoid races when processing tiles
        created_dirs = set()
        for tile in self.data_slice.split(self.tile_shape):
            dir_path = self.get_tile_dir_dataset_path(global_roi=tile)
            if dir_path and dir_path not in created_dirs:
                self.filesystem.makedirs(dir_path)
                created_dirs.add(dir_path)

    def get_tile_dataset_path(self, global_roi: Slice5D) -> str:
        "Gets the relative path into the n5 dataset where 'tile' should be stored"
        local_roi = global_roi.translated(-self.data_slice.start)
        slice_address_components = (local_roi.start // self.tile_shape).to_np(self.axiskeys[::-1]).astype(np.uint32)
        return "/".join(map(str, slice_address_components))

    def get_tile_dir_dataset_path(self, global_roi: Slice5D) -> str:
        return "/".join(self.get_tile_dataset_path(global_roi).split("/")[:-1])

    def _process_tile(self, tile: Array5D) -> None:
        tile = N5Block.fromArray5D(tile)
        tile_path = self.get_tile_dataset_path(global_roi=tile.roi)
        with self.filesystem.openbin(tile_path, "w") as f:
            f.write(tile.to_n5_bytes(axiskeys=self.axiskeys, compression_type=self.compression_type))
