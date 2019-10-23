import numpy as np
from PIL import Image as PilImage

from ndstructs.datasource import DataSource, UnsupportedUrlException
from ndstructs import Array5D, Image, Point5D, Shape5D, Slice5D
from .UnsupportedUrlException import UnsupportedUrlException
import functools


class PilDataSource(DataSource):
    """A naive implementation of DataSource that can read images using PIL"""

    @classmethod
    @functools.lru_cache()
    def get_full_shape(cls, url: str) -> Shape5D:
        img = PilImage.open(url)
        return Shape5D(x=img.width, y=img.height, c=len(img.getbands()))

    def _allocate(self, fill_value: int) -> Image:
        return Image.allocate(self, dtype=self.dtype, value=fill_value)

    @property
    def tile_shape(self):
        return Shape5D(x=1024, y=1024, c=self.shape.c)

    def __init__(self, url: str, *, t=slice(None), c=slice(None), x=slice(None), y=slice(None), z=slice(None)):
        try:
            raw_data = np.asarray(PilImage.open(url))
        except FileNotFoundError as e:
            raise e
        except OSError:
            raise UnsupportedUrlException(url)
        axiskeys = "yxc"[: len(raw_data.shape)]
        self._data = Image(raw_data, axiskeys=axiskeys)
        super().__init__(url, t=t, c=c, x=x, y=y, z=z)

    @property
    def dtype(self):
        return self._data.dtype

    def get(self) -> Image:
        return self._data.cut(self.roi, copy=True)
