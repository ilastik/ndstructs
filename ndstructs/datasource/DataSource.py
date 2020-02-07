import enum
import json
from abc import abstractmethod, ABC
from enum import IntEnum
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List, TypeVar, Callable, cast
from typing_extensions import Protocol

import h5py
import numpy as np
import skimage.io
import z5py

from ndstructs import Array5D, Shape5D, Slice5D
from ndstructs.utils import JsonSerializable

from .UnsupportedUrlException import UnsupportedUrlException


@enum.unique
class AddressMode(IntEnum):
    BLACK = 0
    MIRROR = enum.auto()
    WRAP = enum.auto()


# DS_CTOR = Callable[[str, Optional[Shape5D], str], "DataSource"]


class DS_CTOR(Protocol):
    def __call__(self, url: str, *, tile_shape: Optional[Shape5D] = None, axiskeys: str = "") -> "DataSource":
        ...


class DataSource(JsonSerializable, ABC):
    @classmethod
    def create(cls, url: str, *, tile_shape: Optional[Shape5D] = None, axiskeys: str = "") -> "DataSource":
        registry: List[DS_CTOR] = [N5DataSource, H5DataSource, SkimageDataSource]
        for klass in registry:
            try:
                return klass(url, tile_shape=tile_shape, axiskeys=axiskeys)
            except UnsupportedUrlException:
                pass
        raise UnsupportedUrlException(url)

    def __init__(
        self,
        url: str,
        *,
        tile_shape: Optional[Shape5D] = None,
        dtype: np.dtype,
        axiskeys: str,
        name: str = "",
        shape: Shape5D,
    ):
        self.url = url
        self.tile_shape = (tile_shape or Shape5D.hypercube(256)).to_slice_5d().clamped(shape.to_slice_5d()).shape
        self.dtype = dtype
        self.axiskeys = axiskeys
        self.name = name or self.url.split("/")[-1]
        self.shape = shape
        self.roi = shape.to_slice_5d()

    def __str__(self) -> str:
        return f"<{self.__class__.__name__} {self.shape} {self.url}>"

    @classmethod
    def from_json_data(cls, data: dict) -> "DataSource":
        tile_shape = Shape5D.from_json_data(data["tile_shape"]) if "tile_shape" in data else None
        return cls.create(url=data["url"], tile_shape=tile_shape)

    @property
    def json_data(self) -> Dict:
        return {
            "url": self.url,
            "tile_shape": self.tile_shape,
            "dtype": self.dtype.name,
            "name": self.name,
            "shape": self.shape,
            "roi": self.roi,
        }

    def __repr__(self) -> str:
        return super().__repr__() + f"({self.url.split('/')[-1]})"

    def __hash__(self) -> int:
        return hash((self.url, self.tile_shape))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.url == other.url and self.tile_shape == other.tile_shape

    @abstractmethod
    def _get_tile(self, tile: Slice5D) -> Array5D:
        pass

    def close(self) -> None:
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
    def __init__(self, url: Union[Path, str], *, tile_shape: Optional[Shape5D] = None, axiskeys: str = ""):
        url = str(url)
        if ".n5" not in url:
            raise UnsupportedUrlException(url)
        self.outer_path = url.split(".n5")[0] + ".n5"
        self.inner_path = url.split(".n5")[1]
        if not self.inner_path:
            raise ValueError(f"{url} does not have an inner path")
        self._file = z5py.File(self.outer_path, "r", use_zarr_format=False)
        self._dataset = self._file[self.inner_path]
        axiskeys = axiskeys or "".join(reversed(self._dataset.attrs["axes"])).lower()
        MismatchingAxisKeysException.ensure_matching(axiskeys, self._dataset.shape)
        native_tile_shape = Shape5D(**{key: size for key, size in zip(axiskeys, self._dataset.chunks)})
        super().__init__(
            url,
            tile_shape=tile_shape if tile_shape is not None else native_tile_shape,
            shape=Shape5D(**{key: size for key, size in zip(axiskeys, self._dataset.shape)}),
            dtype=self._dataset.dtype,
            axiskeys=axiskeys,
            name=self.outer_path.split("/")[-1] + self.inner_path,
        )

    def _get_tile(self, tile: Slice5D) -> Array5D:
        slices = tile.to_slices(self.axiskeys)
        raw = self._dataset[slices]
        return Array5D(raw, axiskeys=self.axiskeys, location=tile.start)


class ArrayDataSource(DataSource):
    """A DataSource backed by an Array5D"""

    def __init__(self, url: str = "", *, data: np.ndarray, axiskeys: str, tile_shape: Optional[Shape5D] = None):
        self._data = Array5D(data, axiskeys=axiskeys)
        super().__init__(
            url=url or f"[memory{id(data)}]",
            tile_shape=tile_shape,
            shape=self._data.shape,
            dtype=self._data.dtype,
            axiskeys=axiskeys,
        )

    def _get_tile(self, tile: Slice5D) -> Array5D:
        return self._data.cut(tile, copy=True)

    def _allocate(self, slc: Slice5D, fill_value: int) -> Array5D:
        return self._data.__class__.allocate(slc, dtype=self.dtype, value=fill_value)


class MissingAxisKeysException(Exception):
    pass


class MismatchingAxisKeysException(Exception):
    def __init__(self, axiskeys: str, shape: Tuple[int, ...]):
        super().__init__(f"Axiskeys {axiskeys} do not match encountered data {shape}")

    @classmethod
    def ensure_matching(cls, axiskeys: str, shape: Tuple[int, ...]) -> None:
        if len(axiskeys) != len(shape):
            raise cls(axiskeys=axiskeys, shape=shape)


class H5DataSource(DataSource):
    def __init__(self, url: str, *, tile_shape: Optional[Shape5D] = None, axiskeys: Optional[str] = ""):
        self._dataset: Optional[h5py.Dataset] = None
        try:
            self._dataset = self.openDataset(url)

            axiskeys = axiskeys or self.getAxisKeys(self._dataset)
            MismatchingAxisKeysException.ensure_matching(axiskeys, self._dataset.shape)

            if tile_shape is None and self._dataset.chunks:
                tile_shape = Shape5D(**{k: v for k, v in zip(axiskeys, self._dataset.chunks)})
            super().__init__(
                url=url,
                tile_shape=tile_shape,
                shape=Shape5D(**{key: size for key, size in zip(axiskeys, self._dataset.shape)}),
                dtype=self._dataset.dtype,
                axiskeys=axiskeys,
                name=self._dataset.file.filename.split("/")[-1] + self._dataset.name,
            )
        except Exception as e:
            if self._dataset:
                self._dataset.file.close()
            raise e

    def _get_tile(self, tile: Slice5D) -> Array5D:
        slices = tile.to_slices(self.axiskeys)
        raw = cast(h5py.Dataset, self._dataset)[slices]
        return Array5D(raw, axiskeys=self.axiskeys, location=tile.start)

    def close(self) -> None:
        self._dataset.file.close()  # type: ignore

    @classmethod
    def openDataset(cls, path: Union[Path, str]) -> h5py.Dataset:
        path = Path(path).absolute()
        dataset_path_components: List[str] = []
        while not path.is_file() and not path.is_dir():
            dataset_path_components.insert(0, path.name)
            path = path.parent
        if not path.is_file():
            raise UnsupportedUrlException(path.as_posix())

        try:
            f = h5py.File(path, "r")
        except OSError:
            raise UnsupportedUrlException(path)

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
            return dims_axiskeys

        if "axistags" in dataset.attrs:
            tag_dict = json.loads(dataset.attrs["axistags"])
            return "".join(tag["key"] for tag in tag_dict["axes"])

        raise MissingAxisKeysException("Cuold not find axistags for dataset {dataset} of {dataset.file.filename}")


class SkimageDataSource(ArrayDataSource):
    """A naive implementation of DataSource that can read images using skimage"""

    def __init__(self, url: str, *, tile_shape: Optional[Shape5D] = None, axiskeys: str = ""):
        try:
            raw_data = skimage.io.imread(url)
        except ValueError:
            raise UnsupportedUrlException(url)
        axiskeys = axiskeys or "yxc"[: len(raw_data.shape)]
        MismatchingAxisKeysException.ensure_matching(axiskeys, raw_data.shape)
        super().__init__(url=url, data=raw_data, axiskeys=axiskeys, tile_shape=tile_shape)
        self.url = url
