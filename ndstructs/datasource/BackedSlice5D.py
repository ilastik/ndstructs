from ndstructs.datasource import DataSource
from ndstructs import Slice5D, Shape5D, Array5D, Point5D
from typing import Iterator


class BackedSlice5D(Slice5D):
    def __init__(
        self, datasource: DataSource, *, t=slice(None), c=slice(None), x=slice(None), y=slice(None), z=slice(None)
    ):
        slc = Slice5D(t=t, c=c, x=x, y=y, z=z).defined_with(datasource.shape)
        super().__init__(**slc.to_dict())
        self.datasource = datasource

    def __repr__(self) -> str:
        return super().__repr__() + " " + self.datasource.url

    def rebuild(self, *, t=slice(None), c=slice(None), x=slice(None), y=slice(None), z=slice(None)):
        return self.__class__(self.datasource, t=t, c=c, x=x, y=y, z=z)

    def full(self) -> "BackedSlice5D":
        return self.rebuild(**Slice5D.all().to_dict())

    @property
    def full_shape(self) -> Shape5D:
        return self.datasource.shape

    def contains(self, slc: Slice5D) -> bool:
        return self.contains(slc.defined_with(self.full_shape))

    @property
    def tile_shape(self):
        return self.datasource.tile_shape

    def retrieve(self) -> Array5D:
        return self.datasource.retrieve(self)

    def get_tiles(self, tile_shape: Shape5D = None) -> Iterator["BackedSlice5D"]:
        if not self.is_defined():
            yield from self.defined_with(self.full_shape).get_tiles(tile_shape)
        for tile in super().get_tiles(tile_shape or self.tile_shape):
            clamped_tile = tile.clamped(self.full())
            yield self.rebuild(**clamped_tile.to_dict())

    def get_neighboring_tiles(self, tile_shape: Shape5D = None) -> Iterator["Slice5D"]:
        tile_shape = tile_shape or self.tile_shape
        assert self.is_tile(tile_shape)
        for axis in Point5D.LABELS:
            for axis_offset in (tile_shape[axis], -tile_shape[axis]):
                offset = Point5D.zero(**{axis: axis_offset})
                neighbor = self.translated(offset).clamped(self.full())
                if neighbor.shape.hypervolume > 0 and neighbor != self:
                    yield neighbor
