import json
import enum
from abc import abstractmethod, ABC
from enum import IntEnum
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List, Callable, cast, Any
from typing_extensions import Protocol

import h5py
import numpy as np
import skimage.io
from fs.base import FS
from fs.errors import ResourceNotFound
from fs.osfs import OSFS


from ndstructs import Array5D, Shape5D, Slice5D, Point5D
from ndstructs.utils import JsonSerializable, to_json_data

from .UnsupportedUrlException import UnsupportedUrlException

try:
    import ndstructs_datasource_cache
except ImportError:
    from functools import lru_cache

    ndstructs_datasource_cache = lru_cache(maxsize=4096)


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
        for klass in cls.REGISTRY if cls == DataSource else [cls]:
            try:
                return klass(path, location=location, filesystem=filesystem)
            except UnsupportedUrlException:
                pass
        raise UnsupportedUrlException(path)

    def __init__(
        self,
        url: str,
        *,
        tile_shape: Optional[Shape5D] = None,
        dtype: np.dtype,
        name: str = "",
        shape: Shape5D,
        location: Point5D = Point5D.zero(),
        axiskeys: str,
    ):
        self.url = url
        self.tile_shape = (tile_shape or Shape5D.hypercube(256)).to_slice_5d().clamped(shape.to_slice_5d()).shape
        self.dtype = dtype
        self.name = name or self.url.split("/")[-1]
        self.shape = shape
        self.roi = shape.to_slice_5d(offset=location)
        self.location = location
        self.axiskeys = axiskeys

    def __str__(self) -> str:
        return f"<{self.__class__.__name__} {self.shape} {self.url}>"

    def to_json_data(self, referencer: Callable[[Any], str] = lambda obj: None) -> Dict:
        return to_json_data(
            {
                "__class__": self.__class__.__name__,
                "__self__": referencer(self),
                "url": self.url,
                "tile_shape": self.tile_shape,
                "dtype": self.dtype.name,
                "name": self.name,
                "shape": self.shape,
                "roi": self.roi,
            }
        )

    def __repr__(self) -> str:
        return super().__repr__() + f"({self.name})"

    def __hash__(self) -> int:
        return hash((self.url, self.tile_shape))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.url == other.url and self.tile_shape == other.tile_shape

    @ndstructs_datasource_cache
    def get_tile(self, tile: Slice5D) -> Array5D:
        return self._get_tile(tile)

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
            tile_data = self.get_tile(tile_within_bounds)
            out.set(tile_data, autocrop=True)
        out.setflags(write=False)
        return out.translated(self.location)


class H5DataSource(DataSource):
    def __init__(self, path: Path, *, location: Point5D = Point5D.zero(), filesystem: FS):
        self._dataset: Optional[h5py.Dataset] = None
        try:
            self._dataset, outer_path, inner_path = self.openDataset(path, filesystem=filesystem)
            axiskeys = self.getAxisKeys(self._dataset)
            tile_shape = Shape5D.create(raw_shape=self._dataset.chunks or self._dataset.shape, axiskeys=axiskeys)
            super().__init__(
                url=filesystem.desc(outer_path.as_posix()) + "/" + inner_path.as_posix(),
                tile_shape=tile_shape,
                shape=Shape5D.create(raw_shape=self._dataset.shape, axiskeys=axiskeys),
                dtype=self._dataset.dtype,
                name=self._dataset.file.filename.split("/")[-1] + self._dataset.name,
                location=location,
                axiskeys=axiskeys,
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
    def openDataset(cls, path: Path, filesystem: FS) -> Tuple[h5py.Dataset, Path, Path]:
        outer_path = path
        dataset_path_components: List[str] = []
        while True:
            try:
                info = filesystem.getinfo(outer_path.as_posix())
                if not info.is_file:
                    raise UnsupportedUrlException(path.as_posix())
                break
            except ResourceNotFound as e:
                dataset_path_components.insert(0, outer_path.name)
                parent = outer_path.parent
                if parent == outer_path:
                    raise UnsupportedUrlException(path.as_posix())
                outer_path = parent

        try:
            binfile = filesystem.openbin(outer_path.as_posix())
            f = h5py.File(binfile, "r")
        except OSError as e:
            raise UnsupportedUrlException(path) from e

        try:
            inner_path = "/".join(dataset_path_components)
            dataset = f[inner_path]
            if not isinstance(dataset, h5py.Dataset):
                raise ValueError(f"{inner_path} is not a h5py.Dataset")
        except Exception as e:
            f.close()
            raise e

        return dataset, outer_path, Path(inner_path)

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
        url: str = "",
        *,
        data: np.ndarray,
        axiskeys: str,
        tile_shape: Optional[Shape5D] = None,
        location: Point5D = Point5D.zero(),
    ):
        self._data = Array5D(data, axiskeys=axiskeys)
        super().__init__(
            url=url or "memory://{id(data)}]",
            shape=self._data.shape,
            dtype=self._data.dtype,
            tile_shape=tile_shape,
            location=location,
            axiskeys=axiskeys,
        )

    @classmethod
    def from_array5d(cls, arr, *, tile_shape: Optional[Shape5D] = None, location: Point5D = Point5D.zero()):
        return cls(data=arr.raw(Point5D.LABELS), axiskeys=Point5D.LABELS, location=location, tile_shape=tile_shape)

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
        axiskeys = "yxc"[: len(raw_data.shape)]
        super().__init__(url=filesystem.desc(path.as_posix()), data=raw_data, axiskeys=axiskeys, location=location)


DataSource.REGISTRY.append(SkimageDataSource)
