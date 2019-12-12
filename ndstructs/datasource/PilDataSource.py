import numpy as np
from typing import Optional
from PIL import Image as PilImage

from ndstructs.datasource import ArrayDataSource, UnsupportedUrlException
from ndstructs import Array5D, Image, Point5D, Shape5D, Slice5D
from .UnsupportedUrlException import UnsupportedUrlException
import functools


class PilDataSource(ArrayDataSource):
    """A naive implementation of DataSource that can read images using PIL"""

    def __init__(
        self,
        url: str,
        *,
        data: Optional[Array5D] = None,
        t=slice(None),
        c=slice(None),
        x=slice(None),
        y=slice(None),
        z=slice(None),
    ):
        if data is None:
            try:
                raw_data = np.asarray(PilImage.open(url))
            except FileNotFoundError as e:
                raise e
            except OSError:
                raise UnsupportedUrlException(url)
            axiskeys = "yxc"[: len(raw_data.shape)]
            data = Image(raw_data, axiskeys=axiskeys)
        super().__init__(data=data, tile_shape=Shape5D(c=data.shape.c, x=1024, y=1024), t=t, c=c, x=x, y=y, z=z)
        self.url = url

    def rebuild(self, *, t=slice(None), c=slice(None), x=slice(None), y=slice(None), z=slice(None)) -> "PilDataSource":
        return self.__class__(url=self.url, data=self._data, t=t, c=c, x=x, y=y, z=z)

    def _allocate(self, fill_value: int) -> Image:
        return Image.allocate(self, dtype=self.dtype, value=fill_value)
