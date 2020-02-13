from typing import List, Optional, Union
from pathlib import Path
import bisect

from ndstructs.datasource.DataSourceUrl import DataSourceUrl
from ndstructs.datasource.DataSource import DataSource
from ndstructs.datasource.RelabelingDataSource import RelabelingDataSource
from ndstructs.datasource.BackedSlice5D import BackedSlice5D
from ndstructs import Shape5D, Slice5D, Point5D, Array5D


class SequenceDataSource(DataSource):
    def __init__(
        self,
        url: str,
        *,
        stack_axis: str,
        layer_tile_shape: Optional[Shape5D] = None,
        location: Point5D = Point5D.zero(),
    ):
        self.stack_axis = stack_axis
        self.layers: List[DataSource] = []
        self.layer_offsets: List[int] = []
        layer_offset = Point5D.zero()
        for layer_url in DataSourceUrl.glob(url):
            layer = DataSource.create(layer_url, tile_shape=layer_tile_shape, location=layer_offset)
            self.layers.append(layer)
            self.layer_offsets.append(layer_offset[stack_axis])
            layer_offset += Point5D.zero(**{stack_axis: layer.shape[stack_axis]})

        if len(set(layer.shape.with_coord(**{stack_axis: 1}) for layer in self.layers)) > 1:
            raise ValueError("Provided files have different dimensions on the non-stacking axis")
        if any(layer.dtype != self.layers[0].dtype for layer in self.layers):
            raise ValueError("All layers must have the same data type!")

        stack_size = sum(layer.shape[self.stack_axis] for layer in self.layers)
        full_shape = self.layers[0].shape.with_coord(**{self.stack_axis: stack_size})

        layer_axiskeys = "".join(set("".join(layer.axiskeys for layer in self.layers))).replace(stack_axis, "")
        axiskeys = stack_axis + layer_axiskeys.replace(stack_axis, "")

        super().__init__(
            url=url,
            tile_shape=layer_tile_shape,
            shape=full_shape,
            name="Stack from {url}",
            dtype=self.layers[0].dtype,
            axiskeys=axiskeys,
            location=location,
        )

    def _get_tile(self, tile: Slice5D) -> Array5D:
        first_layer_idx = bisect.bisect_left(self.layer_offsets, tile.start[self.stack_axis])
        out = self._allocate(roi=tile, fill_value=0)
        for layer, layer_offset in zip(self.layers[first_layer_idx:], self.layer_offsets[first_layer_idx:]):
            if layer_offset > tile.stop[self.stack_axis]:
                break
            layer_tile = tile.clamped(layer.roi)
            layer_data = layer.retrieve(layer_tile)
            out.set(layer_data, autocrop=True)

        return out
