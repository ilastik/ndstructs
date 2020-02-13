from typing import Optional

from ndstructs.datasource.DataSource import DataSource, AddressMode
from ndstructs import Array5D, Shape5D, Slice5D, Point5D


class MismatchingAxisKeysException(Exception):
    pass


class RelabelingDataSource(DataSource):
    def __init__(self, datasource: DataSource, axiskeys: str = "", location: Point5D = Point5D.zero()):
        self.datasource = datasource
        axiskeys = axiskeys or datasource.axiskeys
        if len(axiskeys) != len(datasource.axiskeys):
            raise MismatchingAxisKeysException(f"axiskeys {datasource.axiskeys} cannot be relabeled to {axiskeys}")
        self.override_to_native_map = {over: native for over, native in zip(axiskeys, datasource.axiskeys)}
        self.native_to_override_map = {native: over for native, over in zip(datasource.axiskeys, axiskeys)}

        super().__init__(
            url=datasource.url,
            axiskeys=axiskeys,
            shape=datasource.shape.relabeled(self.native_to_override_map),
            tile_shape=datasource.tile_shape.relabeled(self.native_to_override_map),
            dtype=datasource.dtype,
            name=datasource.name,
        )
        self.datasource = datasource

    def retrieve(
        self, roi: Slice5D, address_mode: AddressMode = AddressMode.BLACK, allow_missing: bool = True
    ) -> Array5D:
        # FIXME: Remove address_mode or implement all variations and make feature extractors use the correct one
        internal_roi = roi.relabeled(self.override_to_native_map)
        internal_data = self.datasource.retrieve(
            roi=internal_roi, address_mode=address_mode, allow_missing=allow_missing
        )
        return internal_data.relabeled(self.native_to_override_map)

    def _get_tile(self, tile: Slice5D) -> Array5D:
        pass
