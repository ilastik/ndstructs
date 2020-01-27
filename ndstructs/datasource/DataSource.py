from abc import ABC, abstractmethod, abstractproperty
from functools import lru_cache
from typing import List, Iterator, Iterable, Optional, Union
from numbers import Number
import skimage
from pathlib import Path
import os

import numpy as np

import enum
from enum import IntEnum

import z5py
import h5py
import json
from ndstructs import Array5D, Image, Point5D, Shape5D, Slice5D
from ndstructs.utils import JsonSerializable
from .UnsupportedUrlException import UnsupportedUrlException


@enum.unique
class AddressMode(IntEnum):
    BLACK = 0
    MIRROR = enum.auto()
    WRAP = enum.auto()


class DataSource:
    @classmethod
    def create(cls, url: str, *, tile_shape: Optional[Shape5D] = None):
        for klass in [N5DataSource, H5DataSource, SkimageDataSource]:
            try:
                return klass(url, tile_shape=tile_shape)
            except UnsupportedUrlException as e:
                pass
        else:
            raise UnsupportedUrlException(url)

    def __init__(self, url: str, *, tile_shape: Shape5D, dtype, name: str = "", shape: Shape5D):
        self.url = url
        self.tile_shape = (tile_shape or Shape5D.hypercube(256)).to_slice_5d().clamped(shape.to_slice_5d()).shape
        self.dtype = dtype
        self.name = name or self.url.split("/")[-1]
        self.shape = shape
        self.roi = shape.to_slice_5d()

    @classmethod
    def from_json_data(cls, data: dict):
        tile_shape = Shape5D.from_json_data(data["tile_shape"]) if "tile_shape" in data else None
        return cls.create(url=data["url"], tile_shape=tile_shape)

    @property
    def json_data(self):
        return {
            **super().json_data,
            "url": self.url,
            "full_shape": self.full_shape.json_data,
            "tile_shape": self.tile_shape,
        }

    def __repr__(self):
        return super().__repr__() + f"({self.url.split('/')[-1]})"

    def __hash__(self):
        return hash((self.url, self.tile_shape))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.url == other.url and self.tile_shape == other.tile_shape

    @abstractmethod
    def _get_tile(self, tile: Slice5D) -> Array5D:
        pass

    def _allocate(self, slc: Slice5D, fill_value: int) -> Array5D:
        return Array5D.allocate(slc, dtype=self.dtype, value=fill_value)

    def retrieve(self, roi: Slice5D, address_mode: AddressMode = AddressMode.BLACK) -> Array5D:
        # FIXME: Remove address_mode or implement all variations and make feature extractors
        # use te correct one
        roi = roi.defined_with(self.shape)
        out = self._allocate(roi, fill_value=0)
        data_roi = roi.clamped(self.roi)
        for tile in data_roi.get_tiles(self.tile_shape):
            tile_data = self._get_tile(tile)
            out.set(tile_data, autocrop=True)
        return out  # TODO: make slice read-only


class N5DataSource(DataSource):
    def __init__(self, url: Union[Path, str], *, tile_shape: Optional[int] = None):
        url = str(url)
        if ".n5" not in url:
            raise UnsupportedUrlException(url)
        self.outer_path = url.split(".n5")[0] + ".n5"
        self.inner_path = url.split(".n5")[1]
        if not self.inner_path:
            raise ValueError(f"{url} does not have an inner path")
        self._file = z5py.File(self.outer_path, "r", use_zarr_format=False)
        self._dataset = self._file[self.inner_path]
        self._axiskeys = "".join(reversed(self._dataset.attrs["axes"])).lower()
        native_tile_shape = Shape5D(**{key: size for key, size in zip(self._axiskeys, self._dataset.chunks)})
        super().__init__(
            url,
            tile_shape=tile_shape or native_tile_shape,
            shape=Shape5D(**{key: size for key, size in zip(self._axiskeys, self._dataset.shape)}),
            dtype=self._dataset.dtype,
            name=self.outer_path.split("/")[-1] + self.inner_path,
        )

    def _get_tile(self, tile: Slice5D) -> Array5D:
        slices = tile.to_slices(self._axiskeys)
        raw = self._dataset[slices]
        return Array5D(raw, axiskeys=self._axiskeys, location=tile.start)


class ArrayDataSource(DataSource):
    """A DataSource backed by an Array5D"""

    def __init__(self, *, data: Array5D, tile_shape: Optional[Shape5D] = None):
        self._data = data
        super().__init__(
            url=f"[memory{id(data)}]", tile_shape=tile_shape, shape=self._data.shape, dtype=self._data.dtype
        )

    def _get_tile(self, tile: Slice5D) -> Array5D:
        return self._data.cut(tile, copy=True)

    def _allocate(self, slc: Slice5D, fill_value: int) -> Array5D:
        return self._data.__class__.allocate(slc, dtype=self.dtype, value=fill_value)


class MissingAxisKeysException(Exception):
    pass


class H5DataSource(DataSource):
    def __init__(self, url: str, *, tile_shape: Optional[Shape5D] = None, axiskeys: Optional[str] = ""):
        self._dataset = None
        try:
            self._dataset = self.openDataset(url)

            if axiskeys and len(axiskeys) != len(self._dataset.shape):
                raise ValueError("Mismatching axiskeys and dataset shape: {axiskeys} {self._dataset.shape}")

            self._axiskeys = axiskeys or self.getAxisKeys(self._dataset)
            if tile_shape is None and self._dataset.chunks:
                tile_shape = Shape5D(**{k: v for k, v in zip(self._axiskeys, self._dataset.chunks)})
            super().__init__(
                url=url,
                tile_shape=tile_shape,
                shape=Shape5D(**{key: size for key, size in zip(self._axiskeys, self._dataset.shape)}),
                dtype=self._dataset.dtype,
                name=self._dataset.file.filename.split("/")[-1] + self._dataset.name,
            )
        except Exception as e:
            if self._dataset:
                self._dataset.file.close()
            raise e

    def _get_tile(self, tile: Slice5D) -> Array5D:
        slices = tile.to_slices(self._axiskeys)
        raw = self._dataset[slices]
        return Array5D(raw, axiskeys=self._axiskeys, location=tile.start)

    @classmethod
    def openDataset(cls, path: Union[Path, str]) -> h5py.Dataset:
        path = Path(path).absolute()
        dataset_path_components = []
        while not path.is_file() and not path.is_dir():
            dataset_path_components.insert(0, path.name)
            path = path.parent
        if not path.is_file():
            raise UnsupportedUrlException(url)

        try:
            f = h5py.File(path, "r")
        except OSError:
            raise UnsupportedUrlException(url)

        try:
            inner_path = "/".join(dataset_path_components)
            dataset = f[inner_path]
            if not isinstance(dataset, h5py.Dataset):
                raise ValueError(f"{inner_path} is not a h5py.Dataset")
        except Exception as e:
            f.close()
            raise e

        return dataset

    @classmethod
    def getAxisKeys(cls, dataset: h5py.Dataset) -> str:
        dims_axiskeys = "".join([dim.label for dim in dataset.dims])
        if len(dims_axiskeys) != 0:
            if len(dims_axiskeys) != len(dataset.shape):
                raise ValueError("Axiskeys from 'dims' is inconsistent with shape: {dims_axiskeys} {dataset.shape}")
            return axiskeys

        if "axistags" in dataset.attrs:
            tag_dict = json.loads(dataset.attrs["axistags"])
            return "".join(tag["key"] for tag in tag_dict["axes"])

        raise MissingAxisKeysException("Cuold not find axistags for dataset {dataset} of {dataset.file.filename}")


class SkimageDataSource(ArrayDataSource):
    """A naive implementation of DataSource that can read images using skimage"""

    def __init__(self, url: str, *, tile_shape: Optional[Shape5D] = None):
        try:
            raw_data = skimage.io.imread(url)
        except ValueError:
            raise UnsupportedUrlException(url)
        axiskeys = "yxc"[: len(raw_data.shape)]
        data = Image(raw_data, axiskeys=axiskeys)
        super().__init__(data=data, tile_shape=tile_shape)
        self.url = url

    def _allocate(self, slc: Slice5D, fill_value: int) -> Image:
        return Image.allocate(slc, dtype=self.dtype, value=fill_value)
