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
from fs.base import FS
from fs.osfs import OSFS


from ndstructs import Array5D, Shape5D, Slice5D, Point5D
from ndstructs.utils import JsonSerializable

from .UnsupportedUrlException import UnsupportedUrlException


@enum.unique
class AddressMode(IntEnum):
    BLACK = 0
    MIRROR = enum.auto()
    WRAP = enum.auto()


# DS_CTOR = Callable[[str, Optional[Shape5D], str], "DataSource"]


class DS_CTOR(Protocol):
    def __call__(self, path: Path, *, location: Point5D, filesystem: FS) -> "DataSource":
        ...


def guess_axiskeys(raw_shape: Tuple[int, ...]) -> str:
    guesses = {5: "tzyxc", 4: "zyxc", 3: "yxc", 2: "yx", 1: "x"}
    return guesses[len(raw_shape)]


class DataSource(JsonSerializable, ABC):
    REGISTRY: List[DS_CTOR] = []

    @classmethod
    def create(cls, path: Path, *, location: Point5D = Point5D.zero(), filesystem: Optional[FS] = None) -> "DataSource":
        filesystem = filesystem or OSFS(path.anchor)
        for klass in cls.REGISTRY:
            try:
                return klass(path, location=location, filesystem=filesystem)
            except UnsupportedUrlException:
                pass
        raise UnsupportedUrlException(path)

    def __init__(
        self,
        path: Path,
        *,
        tile_shape: Optional[Shape5D] = None,
        dtype: np.dtype,
        name: str = "",
        shape: Shape5D,
        location: Point5D = Point5D.zero(),
    ):
        self.path = path
        self.tile_shape = (tile_shape or Shape5D.hypercube(256)).to_slice_5d().clamped(shape.to_slice_5d()).shape
        self.dtype = dtype
        self.name = name or path.stem
        self.shape = shape
        self.roi = shape.to_slice_5d(offset=location)
        self.location = location

    def __str__(self) -> str:
        return f"<{self.__class__.__name__} {self.shape} {self.path}>"

    @classmethod
    def from_json_data(cls, data: dict) -> "DataSource":
        return cls.create(path=Path(data["path"]))

    @property
    def json_data(self) -> Dict:
        return {
            "path": self.path,
            "tile_shape": self.tile_shape,
            "dtype": self.dtype.name,
            "name": self.name,
            "shape": self.shape,
            "roi": self.roi,
        }

    def __repr__(self) -> str:
        return super().__repr__() + f"({self.name})"

    def __hash__(self) -> int:
        return hash((self.path, self.tile_shape))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.path == other.path and self.tile_shape == other.tile_shape

    @abstractmethod
    def _get_tile(self, tile: Slice5D) -> Array5D:
        pass

    def close(self) -> None:
        pass

    def _allocate(self, roi: Union[Shape5D, Slice5D], fill_value: int) -> Array5D:
        return Array5D.allocate(roi, dtype=self.dtype, value=fill_value)

    def retrieve(self, roi: Slice5D, address_mode: AddressMode = AddressMode.BLACK) -> Array5D:
        # FIXME: Remove address_mode or implement all variations and make feature extractors use the correct one
        out = self._allocate(roi.defined_with(self.shape).translated(-self.location), fill_value=0)
        local_data_roi = roi.clamped(self.roi).translated(-self.location)
        for tile in local_data_roi.get_tiles(self.tile_shape):
            tile_within_bounds = tile.clamped(self.shape)
            tile_data = self._get_tile(tile_within_bounds)
            out.set(tile_data, autocrop=True)
        out.setflags(write=False)
        return out.translated(self.location)


class H5DataSource(DataSource):
    def __init__(self, path: Path, *, location: Point5D = Point5D.zero(), filesystem: FS):
        self._dataset: Optional[h5py.Dataset] = None
        try:
            self._dataset = self.openDataset(path, filesystem=filesystem)
            self.axiskeys = self.getAxisKeys(self._dataset)
            tile_shape = Shape5D(**{k: v for k, v in zip(self.axiskeys, self._dataset.chunks)})
            super().__init__(
                path=path,
                tile_shape=tile_shape,
                shape=Shape5D(**{key: size for key, size in zip(self.axiskeys, self._dataset.shape)}),
                dtype=self._dataset.dtype,
                name=self._dataset.file.filename.split("/")[-1] + self._dataset.name,
                location=location,
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
    def openDataset(cls, path: Path, filesystem: FS) -> h5py.Dataset:
        dataset_path_components: List[str] = []
        while not path.is_file() and not path.is_dir():
            dataset_path_components.insert(0, path.name)
            path = path.parent
        if not path.is_file():
            raise UnsupportedUrlException(path.as_posix())

        try:
            binfile = filesystem.openbin(path.as_posix())
            f = h5py.File(binfile, "r")
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

        return guess_axiskeys(dataset.shape)


DataSource.REGISTRY.append(H5DataSource)


class ArrayDataSource(DataSource):
    """A DataSource backed by an Array5D"""

    def __init__(
        self,
        path: Optional[Path] = None,
        *,
        data: Array5D,
        tile_shape: Optional[Shape5D] = None,
        location: Point5D = Point5D.zero(),
    ):
        self._data = data
        super().__init__(
            path=path or Path(f"memory{id(data)}]"),
            shape=self._data.shape,
            dtype=self._data.dtype,
            tile_shape=tile_shape,
            location=location,
        )

    def _get_tile(self, tile: Slice5D) -> Array5D:
        return self._data.cut(tile, copy=True)

    def _allocate(self, roi: Union[Shape5D, Slice5D], fill_value: int) -> Array5D:
        return self._data.__class__.allocate(roi, dtype=self.dtype, value=fill_value)


class SkimageDataSource(ArrayDataSource):
    """A naive implementation of DataSource that can read images using skimage"""

    def __init__(self, path: Path, *, location: Point5D = Point5D.zero(), filesystem: FS):
        try:
            raw_data = skimage.io.imread(filesystem.openbin(path.as_posix()))
        except ValueError:
            raise UnsupportedUrlException(path)
        super().__init__(path=path, data=Array5D(raw_data, axiskeys="yxc"[: len(raw_data.shape)]), location=location)


DataSource.REGISTRY.append(SkimageDataSource)
