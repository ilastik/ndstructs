from abc import abstractmethod
from typing import Optional

from ndstructs import Point5D, Shape5D, Slice5D, Array5D
from ndstructs.datasource import UnsupportedUrlException
from ndstructs.datasource.DataSource import DataSource, AddressMode
from ndstructs.datasource.DataSourceSlice import DataSourceSlice


class DataSink:
    def __init__(self, *, data_slice: DataSourceSlice, tile_shape: Optional[Shape5D] = None):
        self.data_slice = data_slice
        self.tile_shape = tile_shape or data_slice.tile_shape

    def process(self, roi: Slice5D, address_mode: AddressMode = AddressMode.BLACK) -> None:
        defined_roi = roi.defined_with(self.data_slice)
        assert self.data_slice.contains(defined_roi)
        for piece in defined_roi.split(self.tile_shape):
            source_data = self.data_slice.datasource.retrieve(piece, address_mode=address_mode)
            self._process_tile(source_data)

    @abstractmethod
    def _process_tile(self, tile: Array5D) -> None:
        pass
