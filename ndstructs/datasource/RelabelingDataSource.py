from typing import Optional

from ndstructs.datasource.DataSource import DataSource, AddressMode
from ndstructs import Array5D, Shape5D, Slice5D, Point5D, KeyMap


class RelabelingDataSource(DataSource):
    def __init__(self, datasource: DataSource, keymap: KeyMap):
        self.datasource = datasource
        self.keymap = keymap
        self.reverse_keymap = keymap.reversed()

        super().__init__(
            url=datasource.url,
            shape=datasource.shape.relabeled(keymap),
            tile_shape=datasource.tile_shape.relabeled(keymap),
            location=datasource.location.relabeled(keymap),
            dtype=datasource.dtype,
            name=datasource.name,
        )
        self.datasource = datasource

    def retrieve(
        self, roi: Slice5D, address_mode: AddressMode = AddressMode.BLACK, allow_missing: bool = True
    ) -> Array5D:
        # FIXME: Remove address_mode or implement all variations and make feature extractors use the correct one
        internal_roi = roi.relabeled(self.reverse_keymap)
        internal_data = self.datasource.retrieve(
            roi=internal_roi, address_mode=address_mode, allow_missing=allow_missing
        )
        return internal_data.relabeled(self.keymap)

    def _get_tile(self, tile: Slice5D) -> Array5D:
        pass
