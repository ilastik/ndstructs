import itertools
from typing import Iterator, List, Tuple, Iterable, Optional
import numpy as np
from skimage import measure as skmeasure
import skimage.io
import io
import os
import uuid

from .point5D import Point5D, Slice5D, Shape5D
from ndstructs.utils import JsonSerializable


class Array5D(JsonSerializable):
    """A wrapper around np.ndarray with labeled axes. Enforces 5D, even if some
    dimensions are of size 1. Sliceable with Slice5D's"""

    def __init__(self, arr: np.ndarray, axiskeys: str, location: Point5D = Point5D.zero()):
        assert len(arr.shape) == len(axiskeys)
        missing_keys = [key for key in Point5D.LABELS if key not in axiskeys]
        self._axiskeys = "".join(missing_keys) + axiskeys
        assert sorted(self._axiskeys) == sorted(Point5D.LABELS)
        slices = tuple([np.newaxis for key in missing_keys] + [...])
        self._data = arr[slices]
        self.location = location

    @classmethod
    def fromArray5D(cls, array: "Array5D"):
        return cls(array._data, array.axiskeys, array.location)

    @classmethod
    def from_json_data(cls, data: dict):
        return cls.from_file(io.BytesIO(data["arr"]), Point5D.from_json_data(data["location"]))

    @property
    def json_data(self):
        # FIXME
        raise NotImplemented("json_data")

    @classmethod
    def from_file(cls, filelike, location: Point5D = Point5D.zero()):
        data = skimage.io.imread(filelike)
        return cls(data, "yxc"[: len(data.shape)], location=location)

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.to_slice_5d()}>"

    @classmethod
    def allocate(cls, slc: Slice5D, dtype, axiskeys: str = Point5D.LABELS, value: int = None):
        assert sorted(axiskeys) == sorted(Point5D.LABELS)
        assert slc.is_defined()  # FIXME: Create DefinedSlice class?
        arr = np.empty(slc.shape.to_tuple(axiskeys), dtype=dtype)
        arr = cls(arr, axiskeys, location=slc.start)
        if value is not None:
            arr._data[...] = value
        return arr

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def axiskeys(self):
        return self._axiskeys

    @property
    def _shape(self) -> Tuple:
        return self._data.shape

    @property
    def shape(self) -> Shape5D:
        return Shape5D(**{key: value for key, value in zip(self.axiskeys, self._shape)})

    def iter_over(self, axis: str, step: int = 1) -> Iterator["Array5D"]:
        for slc in self.roi.iter_over(axis, step):
            yield self.cut(slc)

    def frames(self) -> Iterator["Array5D"]:
        return self.iter_over("t")

    def planes(self, key="z") -> Iterator["Array5D"]:
        return self.iter_over(key)

    def channels(self) -> Iterator["Array5D"]:
        return self.iter_over("c")

    def channel_stacks(self, step):
        return self.iter_over("c", step=step)

    def images(self, through_axis="z") -> Iterator["Image"]:
        for image_slice in self.roi.images(through_axis):
            yield Image.fromArray5D(self.cut(image_slice))

    def split(self, shape: Shape5D) -> Iterator["Array5D"]:
        for slc in self.roi.split(shape):
            yield self.cut(slc)

    def as_mask(self) -> "Array5D":
        return Array5D(self._data > 0, axiskeys=self.axiskeys)

    def sample_channels(self, mask: "ScalarData") -> "LinearData":
        """Extracts a 'list' of columnsi from self, one column for every True-valued
        element from the mask. Each element of a column represents the value in one
        of the channels of self. The expected raw shape of the output is therefore
        (N, c)
        where N is the number of True-valued elements in 'mask', and c is the number
        of channels in self."""
        assert self.shape.with_coord(c=1) == mask.shape
        assert mask.dtype == bool  # FIXME: create "Mask" type?

        # mask has singleton channel axis, so 'c' must be in the end to index self.raw
        sampling_axes = self.axiskeys.replace("c", self.axiskeys[-1])[:-1] + "c"

        raw_mask = mask.raw(sampling_axes.replace("c", ""))
        return StaticLine(self.raw(sampling_axes)[raw_mask], StaticLine.DEFAULT_AXES)

    def setflags(self, *, write: bool):
        self._data.setflags(write=write)

    def normalized(self, iteration_axes: str = "tzc") -> "Array5D":
        normalized = self.allocate(self.shape, self.dtype, self.axiskeys)
        axis_ranges = tuple(range(self.shape[key]) for key in iteration_axes)
        for indices in itertools.product(*axis_ranges):
            slc = Slice5D(**{k: v for k, v in zip(iteration_axes, indices)})
            source_slice = self.cut(slc).raw(self.axiskeys)
            dest_slice = normalized.cut(slc).raw(self.axiskeys)
            data_range = np.amax(source_slice) - np.amin(source_slice)
            if data_range != 0:
                dest_slice[...] = (source_slice / data_range * np.iinfo(self.dtype).max).astype(self.dtype)
            else:
                dest_slice[...] = source_slice
        return normalized

    def rebuild(self, arr: np.array, axiskeys: str, location: Point5D = None) -> "Array5D":
        location = self.location if location is None else location
        return self.__class__(arr, axiskeys, location)

    def translated(self, offset: Point5D):
        return self.rebuild(self._data, axiskeys=self._axiskeys, location=self.location + offset)

    def raw(self, axiskeys: str) -> np.ndarray:
        """Returns a raw view of the underlying np.ndarray, containing only the axes
        identified by and ordered like 'axiskeys'"""

        assert all(self.shape[axis] == 1 for axis in Point5D.LABELS if axis not in axiskeys)
        swapped = self.reordered(axiskeys)

        slices = tuple((slice(None) if k in axiskeys else 0) for k in swapped.axiskeys)
        return swapped._data[slices]

    def linear_raw(self):
        """Returns a raw view with one spatial dimension and one channel dimension"""
        new_shape = (int(self.shape.t * self.shape.volume), int(self.shape.c))
        return self.raw("txyzc").reshape(new_shape)

    def reordered(self, axiskeys: str):
        source_indices = [self.axiskeys.index(x) for x in axiskeys]
        dest_indices = sorted(source_indices)

        new_axes = ""
        requested_axis = list(axiskeys)
        for axis in self.axiskeys:
            if axis in axiskeys:
                new_axes += requested_axis.pop(0)
            else:
                new_axes += axis

        moved_arr = np.moveaxis(self._data, source=source_indices, destination=dest_indices)

        return self.rebuild(moved_arr, axiskeys=new_axes)

    def local_cut(self, roi: Slice5D, *, copy: bool = False) -> "Array5D":
        defined_roi = roi.defined_with(self.shape)
        slices = defined_roi.to_slices(self.axiskeys)
        if copy:
            cut_data = np.copy(self._data[slices])
        else:
            cut_data = self._data[slices]
        return self.rebuild(cut_data, self.axiskeys, location=self.location + defined_roi.start)

    def cut(self, roi: Slice5D, *, copy: bool = False) -> "Array5D":
        return self.local_cut(roi.translated(-self.location), copy=copy)  # TODO: define before translate?

    def clamped(self, roi: Slice5D) -> "Array5D":
        return self.cut(self.roi.clamped(roi))

    def to_slice_5d(self):
        return self.shape.to_slice_5d().translated(self.location)

    @property
    def roi(self):
        return self.to_slice_5d()

    def set(self, value: "Array5D", autocrop: bool = False):
        if autocrop:
            value_slc = value.roi.clamped(self.roi)
            value = value.cut(value_slc)
        self.cut(value.roi).raw(Point5D.LABELS)[...] = value.raw(Point5D.LABELS)

    def __eq__(self, other):
        if not isinstance(other, Array5D) or self.shape != other.shape:
            raise Exception(f"Comparing Array5D {self} with {other}")

        return np.all(self._data == other._data)

    def as_uint8(self, normalized=True):
        multi = 255 if normalized else 1
        return Array5D((self._data * multi).astype(np.uint8), axiskeys=self.axiskeys)

    def get_borders(self, thickness: Shape5D) -> Iterable["Array5D"]:
        for border_slc in self.roi.get_borders(thickness):
            yield self.cut(border_slc)

    def connected_components(self, background: int = 0, connectivity: Optional[int] = None) -> Iterable["Array5D"]:
        for frame in self.frames():
            for channel in self.channels():
                raw = self.raw(Point5D.SPATIAL_LABELS)
                labeled = skmeasure.label(raw, background=background, connectivity=connectivity)
                yield ScalarImage(labeled, axiskeys=Point5D.SPATIAL_LABELS, location=self.location)


class StaticData(Array5D):
    """An Array5D with a single time frame"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.shape.is_static


class ScalarData(Array5D):
    """An Array5D with a single channel"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.shape.is_scalar


class FlatData(Array5D):
    """An Array5D with less than 3 spacial dimensions having a size > 1"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.shape.is_flat


class LinearData(Array5D):
    """An Array5D with at most 1 spacial dimension having size > 1"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.shape.is_line
        line_axis = ""
        for axis in self.shape.present_spatial_axes.keys():
            line_axis = axis
            break
        self.line_axis = line_axis or "x"

    @property
    def length(self):
        return self.shape.volume


class Image(StaticData, FlatData):
    """An Array5D representing a 2D image"""

    def channels(self) -> Iterator["ScalarImage"]:
        for channel in super().channels():
            yield ScalarImage(channel._data, self.axiskeys)


class ScalarImage(Image, ScalarData):
    pass


class ScalarLine(LinearData, ScalarData):
    pass


class StaticLine(StaticData, LinearData):
    DEFAULT_AXES = "xc"

    def concatenate(self, *others: List["LinearData"]) -> "LinearData":
        raw_all = [self.linear_raw()] + [o.linear_raw() for o in others]
        data = np.concatenate(raw_all, axis=0)
        return self.rebuild(data, self.line_axis + "c")
