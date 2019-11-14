from abc import ABC, abstractmethod, abstractproperty
from functools import lru_cache
from typing import List, Iterator, Iterable
from numbers import Number

import numpy as np

import enum
from enum import IntEnum

from ndstructs import Array5D, Point5D, Shape5D, Slice5D
from ndstructs.utils import JsonSerializable
from .UnsupportedUrlException import UnsupportedUrlException


@enum.unique
class AddressMode(IntEnum):
    BLACK = 0
    MIRROR = enum.auto()
    WRAP = enum.auto()


class DataSource(Slice5D):
    @classmethod
    @abstractmethod
    def get_full_shape(cls, url: str) -> Shape5D:
        pass

    def __new__(cls, url: str, *, t=slice(None), c=slice(None), x=slice(None), y=slice(None), z=slice(None)):
        if cls is not DataSource:
            return super().__new__(cls)
        for klass in cls.__subclasses__():
            try:
                return klass(url, t=t, c=c, x=x, y=y, z=z)
            except UnsupportedUrlException as e:
                pass
        else:
            raise UnsupportedUrlException(url)

    def __init__(self, url: str, *, t=slice(None), c=slice(None), x=slice(None), y=slice(None), z=slice(None)):
        self.url = url
        self.full_shape = self.get_full_shape(url)
        self.full_roi = self.full_shape.to_slice_5d()
        self.roi = Slice5D(t=t, c=c, x=x, y=y, z=z).defined_with(self.full_shape)
        super().__init__(**self.roi.to_dict())

    @classmethod
    def from_json_data(cls, data: dict):
        start = Point5D.from_json_data(data["start"]) if "start" in data else Point5D.zero()
        stop = Point5D.from_json_data(data["stop"]) if "stop" in data else Point5D.inf()
        slices = cls.make_slices(start, stop)
        return cls(data["url"], **slices)

    @property
    def json_data(self):
        return {**super().json_data, "url": self.url}

    def __repr__(self):
        return super().__repr__() + f"({self.url.split('/')[-1]})"

    def rebuild(self, *, t=slice(None), c=slice(None), x=slice(None), y=slice(None), z=slice(None)) -> "DataSource":
        return self.__class__(self.url, t=t, c=c, x=x, y=y, z=z)

    def __hash__(self):
        return hash((self.url, self.roi))

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        if isinstance(other, self.__class__):
            return self.url == other.url
        return True

    def full(self) -> "DataSource":
        return self.__class__(self.url, **Slice5D.all().to_dict())

    def resize(self, slc: Slice5D):
        return self.__class__(self.url, **slc.to_dict())

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

    def clamped(self, slc: Slice5D = None) -> "DataSource":
        return super().clamped(slc or self.full())

    @abstractmethod
    def _allocate(self, fill_value: Number) -> Array5D:
        pass

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
            yield self.__class__(self.url, **clamped_tile.to_dict())

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
