from abc import ABC, abstractmethod, abstractproperty
from functools import lru_cache
from typing import List, Iterator, Iterable, Optional, Union
from numbers import Number
import skimage
from pathlib import Path

import numpy as np

import enum
from enum import IntEnum

import z5py
from ndstructs import Array5D, Image, Point5D, Shape5D, Slice5D
from ndstructs.utils import JsonSerializable
from .UnsupportedUrlException import UnsupportedUrlException


@enum.unique
class AddressMode(IntEnum):
    BLACK = 0
    MIRROR = enum.auto()
    WRAP = enum.auto()


class DataSource(Slice5D):
    @classmethod
    def create(
        cls,
        url: str,
        *,
        tile_shape_hint: Optional[Shape5D] = None,
        t=slice(None),
        c=slice(None),
        x=slice(None),
        y=slice(None),
        z=slice(None),
    ):
        for klass in [
            SkimageDataSource,
            N5DataSource,
        ]:  # FIXME: every implementation of DataSource would have to be registered here
            try:
                return klass(url, tile_shape_hint=tile_shape_hint, t=t, c=c, x=x, y=y, z=z)
            except UnsupportedUrlException as e:
                pass
        else:
            raise UnsupportedUrlException(url)

    def __init__(
        self,
        url: str,
        *,
        tile_shape_hint: Optional[Shape5D] = None,
        t=slice(None),
        c=slice(None),
        x=slice(None),
        y=slice(None),
        z=slice(None),
    ):
        self.url = url
        self.roi = Slice5D(t=t, c=c, x=x, y=y, z=z).defined_with(self.full_shape)
        self._tile_shape_hint = tile_shape_hint.to_slice_5d().clamped(self.full_roi).shape if tile_shape_hint else None
        super().__init__(**self.roi.to_dict())

    @abstractproperty
    def full_shape(self) -> Shape5D:
        pass

    @property
    def full_roi(self) -> Slice5D:
        return self.full_shape.to_slice_5d()

    @classmethod
    def from_json_data(cls, data: dict):
        start = Point5D.from_json_data(data["start"]) if "start" in data else Point5D.zero()
        stop = Point5D.from_json_data(data["stop"]) if "stop" in data else Point5D.inf()
        slices = cls.make_slices(start, stop)
        tile_shape_hint = Shape5D.from_json_data(data["tile_shape_hint"]) if "tile_shape_hint" in data else None
        return cls.create(url=data["url"], tile_shape_hint=tile_shape_hint, **slices)

    @property
    def json_data(self):
        return {**super().json_data, "url": self.url, "full_shape": self.full_shape.json_data}

    def __repr__(self):
        return super().__repr__() + f"({self.url.split('/')[-1]})"

    @abstractmethod
    def rebuild(self, *, t=slice(None), c=slice(None), x=slice(None), y=slice(None), z=slice(None)) -> "DataSource":
        pass

    def __hash__(self):
        return hash((self.url, self.roi))

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        if isinstance(other, self.__class__):
            return self.url == other.url
        return True

    def full(self) -> "DataSource":
        return self.rebuild(**Slice5D.all().to_dict())

    def resize(self, slc: Slice5D):
        return self.rebuild(**slc.to_dict())

    @abstractproperty
    def dtype(self):
        pass

    @abstractmethod
    def get(self) -> Array5D:
        pass

    def contains(self, slc: Slice5D) -> bool:
        return self.roi.contains(slc.defined_with(self.full_shape))

    @abstractproperty
    def tile_shape(self):
        """A sensible tile shape. Override this with your data chunk size"""
        pass

    @property
    def name(self) -> str:
        return self.url.split("/")[-1]

    def clamped(self, slc: Slice5D = None) -> "DataSource":
        return super().clamped(slc or self.full())

    def _allocate(self, fill_value: int) -> Array5D:
        return Array5D.allocate(self.roi, dtype=self.dtype, value=fill_value)

    def retrieve(self, address_mode: AddressMode = AddressMode.BLACK) -> Array5D:
        # FIXME: Remove address_mode or implement all variations and make feature extractors
        # use te correct one
        out = self._allocate(fill_value=0)
        data_roi = self.clamped()
        for tile in data_roi.get_tiles():
            tile_data = tile.get()
            out.set(tile_data, autocrop=True)
        return out  # TODO: make slice read-only

    def get_tiles(self, tile_shape: Shape5D = None):
        for tile in super().get_tiles(tile_shape or self.tile_shape):
            clamped_tile = tile.clamped(self.full())
            yield self.rebuild(**clamped_tile.to_dict())

    def is_tile(self, tile_shape: Shape5D = None) -> bool:
        tile_shape = tile_shape or self.tile_shape
        has_tile_start = self.start % tile_shape == Point5D.zero()
        has_tile_end = self.stop % tile_shape == Point5D.zero() or self.stop == self.full_roi.stop
        return has_tile_start and has_tile_end

    def get_neighboring_tiles(self, tile_shape: Shape5D = None) -> Iterable["DataSource"]:
        tile_shape = tile_shape or self.tile_shape
        assert self.is_tile(tile_shape)
        for axis in Point5D.LABELS:
            for axis_offset in (tile_shape[axis], -tile_shape[axis]):
                offset = Point5D.zero(**{axis: axis_offset})
                neighbor = self.translated(offset).clamped(self.full_roi)
                if neighbor.shape.hypervolume > 0 and neighbor != self:
                    yield neighbor

    def __getstate__(self):
        return self.json_data

    def __setstate__(self, data: dict):
        self.__init__(data["url"], Slice5D.from_json(data["roi"]))


class N5DataSource(DataSource):
    def __init__(
        self,
        url: Union[Path, str],
        *,
        tile_shape_hint: Optional[int] = None,
        n5file: Optional[z5py.File] = None,
        t=slice(None),
        c=slice(None),
        x=slice(None),
        y=slice(None),
        z=slice(None),
    ):
        url = str(url)
        if ".n5" not in url:
            raise UnsupportedUrlException(url)
        self.outer_path = url.split(".n5")[0] + ".n5"
        self.inner_path = url.split(".n5")[1]
        if not self.inner_path:
            raise ValueError(f"{url} does not have an inner path")
        if n5file is None:
            self._file = z5py.File(self.outer_path, "r", use_zarr_format=False)
        else:
            self._file = n5file
        self._dataset = self._file[self.inner_path]
        self._axiskeys = "".join(reversed(self._dataset.attrs["axes"])).lower()
        self._native_tile_shape = Shape5D(**{key: size for key, size in zip(self._axiskeys, self._dataset.chunks)})
        super().__init__(url, tile_shape_hint=tile_shape_hint, t=t, c=c, x=x, y=y, z=z)

    @property
    def name(self) -> str:
        return self.outer_path.split("/")[-1] + self.inner_path

    @property
    def full_shape(self) -> Shape5D:
        return Shape5D(**{key: size for key, size in zip(self._axiskeys, self._dataset.shape)})

    @property
    def dtype(self):
        return self._dataset.dtype

    def get(self) -> Array5D:
        slices = self.roi.to_slices(self._axiskeys)
        raw = self._dataset[slices]
        return Array5D(raw, axiskeys=self._axiskeys, location=self.roi.start)

    def rebuild(self, *, t=slice(None), c=slice(None), x=slice(None), y=slice(None), z=slice(None)) -> "N5DataSource":
        return self.__class__(
            self.url, n5file=self._file, tile_shape_hint=self._tile_shape_hint, t=t, c=c, x=x, y=y, z=z
        )

    @property
    def tile_shape(self) -> Shape5D:
        return self._tile_shape_hint or self._native_tile_shape


class ArrayDataSource(DataSource):
    """A DataSource backed by an Array5D"""

    def __init__(
        self,
        *,
        data: Array5D,
        tile_shape_hint: Optional[Shape5D] = None,
        t=slice(None),
        c=slice(None),
        x=slice(None),
        y=slice(None),
        z=slice(None),
    ):
        self._data = data
        tile_shape_hint = tile_shape_hint or Shape5D.hypercube(256)
        super().__init__(f"[memory{id(data)}]", tile_shape_hint=tile_shape_hint, t=t, c=c, x=x, y=y, z=z)

    @property
    def full_shape(self) -> Shape5D:
        return self._data.shape

    @property
    def tile_shape(self):
        return self._tile_shape_hint

    @property
    def dtype(self):
        return self._data.dtype

    def get(self) -> Array5D:
        return self._data.cut(self.roi, copy=True)

    def _allocate(self, fill_value: int) -> Array5D:
        return self._data.__class__.allocate(self.roi, dtype=self.dtype, value=fill_value)

    def rebuild(
        self, *, t=slice(None), c=slice(None), x=slice(None), y=slice(None), z=slice(None)
    ) -> "ArrayDataSource":
        return self.__class__(data=self._data, t=t, c=c, x=x, y=y, z=z)


class SkimageDataSource(ArrayDataSource):
    """A naive implementation of DataSource that can read images using skimage"""

    def __init__(
        self,
        url: str,
        *,
        tile_shape_hint: Optional[Shape5D] = None,
        data: Optional[Array5D] = None,
        t=slice(None),
        c=slice(None),
        x=slice(None),
        y=slice(None),
        z=slice(None),
    ):
        if data is None:
            try:
                raw_data = skimage.io.imread(url)
            except ValueError:
                raise UnsupportedUrlException(url)
            axiskeys = "yxc"[: len(raw_data.shape)]
            data = Image(raw_data, axiskeys=axiskeys)
        super().__init__(data=data, tile_shape_hint=tile_shape_hint, t=t, c=c, x=x, y=y, z=z)
        self.url = url

    def rebuild(
        self, *, t=slice(None), c=slice(None), x=slice(None), y=slice(None), z=slice(None)
    ) -> "SkimageDataSource":
        return self.__class__(url=self.url, data=self._data, t=t, c=c, x=x, y=y, z=z)

    def _allocate(self, fill_value: int) -> Image:
        return Image.allocate(self, dtype=self.dtype, value=fill_value)
