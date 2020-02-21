from typing import Optional

import z5py

from ndstructs.point5D import Point5D, Slice5D, Shape5D
from ndstructs.array5D import Array5D
from ndstructs.datasource.DataSource import DataSource, UnsupportedUrlException
from ndstructs.datasink.DataSink import DataSink


class N5DataSink(DataSink):
    def __init__(
        self,
        *,
        url: str,
        datasource: DataSource,
        axiskeys: str = "tzyxc",
        mode: str = "w",
        compression: str = "raw",
        tile_shape: Optional[Shape5D] = None,
    ):
        self.url = str(url)
        if ".n5" not in self.url:
            raise UnsupportedUrlException(self.url)
        self.outer_path = self.url.split(".n5")[0] + ".n5"
        self.inner_path = self.url.split(".n5")[1]
        if not self.inner_path:
            raise ValueError(f"{url} does not have an inner path")
        super().__init__(datasource=datasource, tile_shape=tile_shape)
        self.axiskeys = axiskeys
        self._file = z5py.File(self.outer_path, mode=mode)
        self._dataset = self._file.create_dataset(
            self.inner_path,
            shape=datasource.shape.to_tuple(self.axiskeys),
            chunks=self.tile_shape.to_tuple(self.axiskeys),
            dtype=datasource.dtype.name,
            compression=compression,
        )
        self._dataset.attrs["axes"] = list(axiskeys.lower()[::-1])

    def _process_tile(self, tile: Array5D) -> None:
        self._dataset[tile.roi.to_slices(self.axiskeys)] = tile.raw(self.axiskeys)
