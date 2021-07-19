from typing import Iterable
import bisect

from ndstructs.datasource.DataSource import DataSource
from ndstructs import Interval5D, Point5D, Array5D


class SequenceDataSource(DataSource):
    def __init__(
        self,
        *,
        stack_axis: str,
        datasources: Iterable[DataSource],
    ):
        self.stack_axis = stack_axis
        self.datasources = sorted(datasources, key= lambda ds: ds.location[stack_axis])
        tile_shapes = {ds.tile_shape for ds in self.datasources}
        if len(tile_shapes) != 1:
            raise ValueError(f"All datasources must have the same tile shape. Tile shapes: {tile_shapes}")
        tile_shape = tile_shapes.pop()
        if any(ds.shape[stack_axis] % tile_shape[stack_axis] != 0 for ds in self.datasources):
            raise ValueError(f"Stacking over axis that are not multiple of the tile size is not supported")
        self.stack_levels = [ds.location[stack_axis] for ds in self.datasources]
        interval = Interval5D.enclosing(ds.interval for ds in self.datasources)
        super().__init__(
            url=":".join(ds.url for ds in self.datasources),
            shape=interval.shape,
            name="Stack from " + ":".join(ds.name for ds in self.datasources),
            dtype=self.datasources[0].dtype,
            location=interval.start,
            axiskeys=stack_axis + Point5D.LABELS.replace(stack_axis, ""),
            tile_shape=tile_shape
        )

    def _get_tile(self, tile: Interval5D) -> Array5D:
        tile_level = tile.start[self.stack_axis]
        datasource_idx = max(0, bisect.bisect_right(self.stack_levels, tile_level) - 1)
        return self.datasources[datasource_idx]._get_tile(tile)
