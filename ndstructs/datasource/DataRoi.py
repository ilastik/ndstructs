from ndstructs.datasource.DataSource import DataSource, AddressMode
from ndstructs import Interval5D, Shape5D, Array5D, Point5D
from ndstructs.point5D import INTERVAL_5D, SPAN
from typing import Iterator, Optional, Union


class DataRoi(Interval5D):
    def __init__(
        self, datasource: DataSource, *, t: SPAN = None, c: SPAN = None, x: SPAN = None, y: SPAN = None, z: SPAN = None
    ):
        super().__init__(
            t=t if t is not None else datasource.interval.t,
            c=c if c is not None else datasource.interval.c,
            x=x if x is not None else datasource.interval.x,
            y=y if y is not None else datasource.interval.y,
            z=z if z is not None else datasource.interval.z,
        )
        self.datasource = datasource

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.datasource))

    def __eq__(self, other: object) -> bool:
        if not super().__eq__(other):
            return False
        if isinstance(other, DataRoi) and self.datasource != other.datasource:
            return False
        return True

    def updated(
        self,
        *,
        x: Optional[SPAN] = None,
        y: Optional[SPAN] = None,
        z: Optional[SPAN] = None,
        t: Optional[SPAN] = None,
        c: Optional[SPAN] = None,
    ) -> "DataRoi":
        inter = self.interval.updated(t=t, c=c, x=x, y=y, z=z)
        return self.__class__(datasource=self.datasource, x=inter.x, y=inter.y, z=inter.z, t=inter.t, c=inter.c)

    def __repr__(self) -> str:
        return super().__repr__() + " " + self.datasource.url

    def full(self) -> "DataRoi":
        return self.updated(**self.full_shape.to_interval5d().to_dict())

    @property
    def full_shape(self) -> Shape5D:
        return self.datasource.shape

    @property
    def tile_shape(self) -> Shape5D:
        return self.datasource.tile_shape

    @property
    def dtype(self):
        return self.datasource.dtype

    def is_tile(self, tile_shape: Shape5D = None) -> bool:
        tile_shape = tile_shape or self.tile_shape
        has_tile_start = self.start % tile_shape == Point5D.zero()
        has_tile_end = self.stop % tile_shape == Point5D.zero() or self.stop == self.full().stop
        return has_tile_start and has_tile_end

    @property
    def interval(self) -> Interval5D:
        return Interval5D(t=self.t, c=self.c, x=self.x, y=self.y, z=self.z)

    def retrieve(self, address_mode: AddressMode = AddressMode.BLACK) -> Array5D:
        return self.datasource.retrieve(self.interval, address_mode=address_mode)

    def split(self, block_shape: Optional[Shape5D] = None) -> Iterator["DataRoi"]:
        yield from super().split(block_shape or self.tile_shape)

    def get_tiles(self, tile_shape: Shape5D = None, clamp: bool = True) -> Iterator["DataRoi"]:
        for tile in super().get_tiles(tile_shape or self.tile_shape):
            if clamp:
                clamped = tile.clamped(self)
                if not self.contains(clamped):
                    continue
                yield clamped
            else:
                yield tile

    # for this and the next method, tile_shape is needed because self could be an edge tile, and therefor
    # self.shape would not return a typical tile shape
    def get_neighboring_tiles(self, tile_shape: Shape5D) -> Iterator["DataRoi"]:
        for neighbor in super().get_neighboring_tiles(tile_shape):
            neighbor = neighbor.clamped(self.full())
            if neighbor.shape.hypervolume > 0 and neighbor != self:
                yield neighbor

    def get_neighbor_tile_adjacent_to(self, *, anchor: Interval5D, tile_shape: Shape5D) -> Optional["DataRoi"]:
        neighbor = super().get_neighbor_tile_adjacent_to(anchor=anchor, tile_shape=tile_shape)
        if not self.full().contains(neighbor):
            return None
        return neighbor.clamped(self.full())
