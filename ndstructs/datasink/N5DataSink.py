from typing import Optional
import re
from pathlib import Path
import json

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
    def __init__(
        self,
        *,
        url: str,
        data_slice: DataSourceSlice,
        axiskeys: str = "tzyxc",
        mode: str = "w",
        compression_type: str = "raw",
        tile_shape: Optional[Shape5D] = None,
        fs: Optional[FS] = None,
    ):
        assert set(data_slice.shape.present_spatial_axes.keys()).issubset(set(axiskeys))
        url = Path(url).as_posix()
        if not re.search(r"\w\.n5/\w", url, re.IGNORECASE):
            raise UnsupportedUrlException(url)
        super().__init__(data_slice=data_slice, tile_shape=tile_shape)
        self.axiskeys = axiskeys
        self.compression_type = compression_type
        self.fs = fs.opendir(url) if fs else OSFS("/").makedirs(Path(url).absolute().as_posix())
        attributes = {
            "dimensions": self.data_slice.shape.to_tuple(axiskeys[::-1]),
            "blockSize": self.tile_shape.to_tuple(axiskeys[::-1]),
            "axes": list(self.axiskeys[::-1]),
            "dataType": str(data_slice.dtype),
            "compression": {"type": self.compression_type},
        }
        with self.fs.openbin("attributes.json", "w") as f:
            f.write(json.dumps(attributes).encode("utf-8"))

        # create all directories in the constructor to avoid races when processing tiles
        created_dirs = set()
        for tile in self.data_slice.split(self.tile_shape):
            dir_path = self.get_tile_dir_dataset_path(global_roi=tile)
            if dir_path and dir_path not in created_dirs:
                self.fs.makedirs(dir_path)
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
        with self.fs.openbin(tile_path, "w") as f:
            f.write(tile.to_n5_bytes(axiskeys=self.axiskeys, compression_type=self.compression_type))
