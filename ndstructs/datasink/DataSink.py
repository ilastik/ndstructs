from abc import abstractmethod
from typing import Optional

from ndstructs import Point5D, Shape5D, Slice5D, Array5D
from ndstructs.datasource import UnsupportedUrlException
from ndstructs.datasource.DataSource import DataSource, AddressMode


class DataSink:
    def __init__(self, *, datasource: DataSource, tile_shape: Optional[Shape5D] = None):
        self.datasource = datasource
        self.tile_shape = tile_shape or datasource.tile_shape

    def process(self, roi: Slice5D, address_mode: AddressMode = AddressMode.BLACK, allow_missing: bool = True) -> None:
        for piece in roi.defined_with(self.datasource.roi).split(self.tile_shape):
            source_data = self.datasource.retrieve(piece, address_mode=address_mode, allow_missing=allow_missing)
            self._process_tile(source_data)

    @abstractmethod
    def _process_tile(self, tile: Array5D) -> None:
        pass
