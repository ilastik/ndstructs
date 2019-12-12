from ndstructs import Array5D
from ndstructs.datasource import DataSource
from ndstructs import Shape5D


class ArrayDataSource(DataSource):
    """A DataSource backed by an Array5D"""

    def __init__(
        self,
        *,
        data: Array5D,
        tile_shape: Shape5D = None,
        t=slice(None),
        c=slice(None),
        x=slice(None),
        y=slice(None),
        z=slice(None),
    ):
        self._data = data
        self._tile_shape = tile_shape or Shape5D.hypercube(256).to_slice_5d().clamped(data.roi).shape
        super().__init__(f"[memory{id(data)}]", t=t, c=c, x=x, y=y, z=z)

    @property
    def full_shape(self) -> Shape5D:
        return self._data.shape

    @property
    def tile_shape(self):
        return self._tile_shape

    @property
    def dtype(self):
        return self._data.dtype

    def get(self) -> Array5D:
        return self._data.cut(self.roi, copy=True)

    def _allocate(self, fill_value: int) -> Array5D:
        return self._data.__class__.allocate(self.roi, dtype=self.dtype, value=fill_value)

    def rebuild(self, *, t=slice(None), c=slice(None), x=slice(None), y=slice(None), z=slice(None)) -> "Array5D":
        return self.__class__(data=self._data, t=t, c=c, x=x, y=y, z=z)
