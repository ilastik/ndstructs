from typing import Iterator, Iterable, Optional, Union, TypeVar, Type, cast, Sequence
import numpy as np
from skimage import measure as skmeasure
import skimage.io
import io
import os
import uuid
from numbers import Number

from .point5D import Point5D, Interval5D, Shape5D, KeyMap, SPAN
from ndstructs.utils import JsonSerializable

Arr = TypeVar("Arr", bound="Array5D")

DTYPE = Union[
    Type[np.uint8],
    Type[np.uint16],
    Type[np.uint32],
    Type[np.uint64],
    Type[np.int8],
    Type[np.int16],
    Type[np.int32],
    Type[np.int64],
    Type[np.float16],
    Type[np.float32],
    Type[np.float64],
]


class All:
    pass


SPAN_OVERRIDE = Union[SPAN, All]


class Array5D(JsonSerializable):
    """A wrapper around np.ndarray with labeled axes. Enforces 5D, even if some
    dimensions are of size 1. Sliceable with Interval5D's"""

    LINEAR_RAW_AXISKEYS = "txyzc"

    def __init__(self, arr: np.ndarray, axiskeys: str, location: Point5D = Point5D.zero()):
        assert len(arr.shape) == len(axiskeys)
        missing_keys = [key for key in Point5D.LABELS if key not in axiskeys]
        self.axiskeys = "".join(missing_keys) + axiskeys
        assert sorted(self.axiskeys) == sorted(Point5D.LABELS)
        slices = tuple([np.newaxis for key in missing_keys] + [...])
        self._data = arr[slices]
        self.location = location
        self.shape = Shape5D(**{key: value for key, value in zip(self.axiskeys, self._data.shape)})
        self.dtype = arr.dtype

    def relabeled(self: Arr, keymap: KeyMap) -> Arr:
        new_location = self.location.relabeled(keymap)
        new_axiskeys = keymap.map_axiskeys(self.axiskeys)
        return self.rebuild(self.raw(self.axiskeys), axiskeys=new_axiskeys, location=new_location)

    @classmethod
    def fromArray5D(cls: Type[Arr], array: "Array5D", copy: bool = False) -> Arr:
        data = np.copy(array._data) if copy else array._data
        return cls(data, array.axiskeys, array.location)

    @classmethod
    def from_stack(cls: Type[Arr], stack: Sequence["Array5D"], stack_along: str) -> Arr:
        axiskeys = stack_along + "xyztc".replace(stack_along, "")

        raw_all = [a.raw(axiskeys) for a in stack]
        data = np.concatenate(raw_all, axis=0)
        return cls(data, axiskeys=axiskeys, location=stack[0].location)

    @classmethod
    def from_json_data(cls: Type[Arr], data: dict) -> Arr:
        raw_bytes = cast(io.IOBase, io.BytesIO(data["arr"]))
        return cls.from_file(raw_bytes, Point5D.from_json_data(data["location"]))

    def to_json_data(self) -> dict:
        # FIXME
        raise NotImplementedError("to_json_data")

    @classmethod
    def from_file(cls: Type[Arr], filelike: io.IOBase, location: Point5D = Point5D.zero()) -> Arr:
        data = skimage.io.imread(filelike)
        return cls(data, "yxc"[: len(data.shape)], location=location)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.interval}>"

    @classmethod
    def allocate(
        cls: Type[Arr], slc: Union[Interval5D, Shape5D], dtype: DTYPE, axiskeys: str = Point5D.LABELS, value: int = None
    ) -> Arr:
        slc = slc.to_interval5d() if isinstance(slc, Shape5D) else slc
        assert sorted(axiskeys) == sorted(Point5D.LABELS)
        assert slc.shape.hypervolume != float("inf")
        arr = np.empty(slc.shape.to_tuple(axiskeys), dtype=dtype)
        arr = cls(arr, axiskeys, location=slc.start)
        if value is not None:
            arr._data[...] = value
        return arr

    @classmethod
    def allocate_like(
        cls: Type[Arr], arr: "Array5D", dtype: Optional[DTYPE], axiskeys: str = "", value: int = None
    ) -> Arr:
        return cls.allocate(arr.interval, dtype=dtype or arr.dtype, axiskeys=axiskeys or arr.axiskeys, value=value)

    def split(self: Arr, shape: Shape5D) -> Iterator[Arr]:
        for slc in self.interval.split(shape):
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
        assert self.shape.updated(c=1) == mask.shape
        assert mask.dtype == bool  # FIXME: create "Mask" type?

        # mask has singleton channel axis, so 'c' must be in the end to index self.raw
        sampling_axes = self.axiskeys.replace("c", self.axiskeys[-1])[:-1] + "c"

        raw_mask = mask.raw(sampling_axes.replace("c", ""))
        return StaticLine(self.raw(sampling_axes)[raw_mask], StaticLine.DEFAULT_AXES)

    def unique_colors(self) -> "StaticLine":
        """Produces an array of shape
            Shape5D(x=number_of_unique_colors, c=self.shape.c)

        where each x element represents a unique combination across all channels of self
        """
        unique_colors = np.unique(self.linear_raw(), axis=0)
        return StaticLine(unique_colors, axiskeys="xc")

    def color_filtered(self, color: "StaticLine"):
        """Creates an array with shape self.shape where all data points are either equal to 'color' or zero otherwise"""
        if color.shape.c != self.shape.c:
            raise ValueError(f"Color {color} has wrong number of channels to filter {self}")
        raw_data = self.linear_raw()
        raw_color = color.linear_raw()
        raw_filtered = np.where(raw_data == raw_color, raw_data, np.zeros(raw_data.shape))
        filtered = Array5D.from_line(raw_filtered, shape=self.shape)
        return self.rebuild(filtered._data, axiskeys=filtered.axiskeys, location=self.location)

    def setflags(self, *, write: bool) -> None:
        self._data.setflags(write=write)

    def normalized(self: Arr, step: Optional[Shape5D] = None) -> Arr:
        step = step if step is not None else self.interval.updated(c=1, t=1).clamped(self.shape).shape
        normalized = self.allocate(self.shape, self.dtype, self.axiskeys)
        for source, dest in zip(normalized.split(step), self.split(step)):
            source_raw = source.raw(self.axiskeys)
            data_range = np.amax(source_raw) - np.amin(source_raw)
            dest_raw = dest.raw(self.axiskeys)
            if data_range != 0:
                dest_raw[...] = (source_raw / data_range * np.iinfo(self.dtype).max).astype(self.dtype)
            else:
                dest_raw[...] = source_raw
        return normalized

    def rebuild(self: Arr, arr: np.ndarray, *, axiskeys: str, location: Point5D = None) -> Arr:
        location = self.location if location is None else location
        return self.__class__(arr, axiskeys, location)

    def translated(self: Arr, offset: Point5D) -> Arr:
        return self.rebuild(self._data, axiskeys=self.axiskeys, location=self.location + offset)

    def raw(self, axiskeys: str) -> np.ndarray:
        """Returns a raw view of the underlying np.ndarray, containing only the axes
        identified by and ordered like 'axiskeys'"""

        assert all(self.shape[axis] == 1 for axis in Point5D.LABELS if axis not in axiskeys)
        swapped = self.reordered(axiskeys)

        slices = tuple((slice(None) if k in axiskeys else 0) for k in swapped.axiskeys)
        return swapped._data[slices]

    def linear_raw(self) -> np.ndarray:
        """Returns a raw view with one spatial dimension and one channel dimension"""
        new_shape = (int(self.shape.t * self.shape.volume), int(self.shape.c))
        return self.raw(self.LINEAR_RAW_AXISKEYS).reshape(new_shape)

    @classmethod
    def from_line(cls: Type[Arr], arr: np.ndarray, *, shape: Shape5D, location: Point5D = Point5D.zero()) -> Arr:
        reshaped_data = arr.reshape(shape.to_tuple(cls.LINEAR_RAW_AXISKEYS))
        return cls(reshaped_data, axiskeys=cls.LINEAR_RAW_AXISKEYS, location=location)

    def reordered(self: Arr, axiskeys: str) -> Arr:
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

    def local_cut(
        self: Arr,
        interval: Interval5D = None,
        *,
        x: Optional[SPAN_OVERRIDE] = None,
        y: Optional[SPAN_OVERRIDE] = None,
        z: Optional[SPAN_OVERRIDE] = None,
        t: Optional[SPAN_OVERRIDE] = None,
        c: Optional[SPAN_OVERRIDE] = None,
        copy: bool = False,
    ) -> Arr:
        local_interval = self.shape.to_interval5d()
        interval = (interval or local_interval).updated(
            x=local_interval.x if isinstance(x, All) else x,
            y=local_interval.y if isinstance(y, All) else y,
            z=local_interval.z if isinstance(z, All) else z,
            t=local_interval.t if isinstance(t, All) else t,
            c=local_interval.c if isinstance(c, All) else c,
        )
        slices = interval.to_slices(self.axiskeys)
        if any(slc.start < 0 for slc in slices):
            raise ValueError(f"Cant't cut locally with negative indices: {interval}")
        if copy:
            cut_data = np.copy(self._data[slices])
        else:
            cut_data = self._data[slices]
        return self.rebuild(cut_data, axiskeys=self.axiskeys, location=self.location + interval.start)

    def cut(
        self: Arr,
        interval: Interval5D = None,
        *,
        x: Optional[SPAN_OVERRIDE] = None,
        y: Optional[SPAN_OVERRIDE] = None,
        z: Optional[SPAN_OVERRIDE] = None,
        t: Optional[SPAN_OVERRIDE] = None,
        c: Optional[SPAN_OVERRIDE] = None,
        copy: bool = False,
    ) -> Arr:
        interval = (
            (interval or self.interval)
            .updated(
                x=self.interval.x if isinstance(x, All) else x,
                y=self.interval.y if isinstance(y, All) else y,
                z=self.interval.z if isinstance(z, All) else z,
                t=self.interval.t if isinstance(t, All) else t,
                c=self.interval.c if isinstance(c, All) else c,
            )
            .translated(-self.location)
        )
        return self.local_cut(interval, copy=copy)

    def duplicate(self: Arr) -> Arr:
        return self.cut(self.interval, copy=True)

    def clamped(
        self: Arr,
        interval: Union[Shape5D, Interval5D, None] = None,
        *,
        x: Optional[SPAN] = None,
        y: Optional[SPAN] = None,
        z: Optional[SPAN] = None,
        t: Optional[SPAN] = None,
        c: Optional[SPAN] = None,
    ) -> Arr:
        return self.cut(self.interval.clamped(interval, x=x, y=y, z=z, t=t, c=c))

    @property
    def interval(self) -> Interval5D:
        return self.shape.to_interval5d().translated(self.location)

    def set(self, value: "Array5D", autocrop: bool = False, mask_value: Optional[Number] = None) -> None:
        if autocrop:
            value_slc = value.interval.clamped(self.interval)
            value = value.cut(value_slc)
        self.cut(value.interval).localSet(value.translated(-self.location), mask_value=mask_value)

    def localSet(self, value: "Array5D", mask_value: Optional[Number] = None) -> None:
        self_raw = self.raw(Point5D.LABELS)
        value_raw = value.raw(Point5D.LABELS)
        if mask_value is None:
            self_raw[...] = value_raw
        else:
            self_raw[...] = np.where(value_raw != mask_value, value_raw, self_raw)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Array5D) or self.shape != other.shape:
            raise Exception(f"Comparing Array5D {self} with {other}")

        return np.all(self.raw(Point5D.LABELS) == other.raw(Point5D.LABELS))

    def as_uint8(self, normalized: bool = True) -> "Array5D":
        multi = 255 if normalized else 1
        return Array5D((self._data * multi).astype(np.uint8), axiskeys=self.axiskeys)

    def get_borders(self: Arr, thickness: Shape5D) -> Iterable[Arr]:
        for border_slc in self.interval.get_borders(thickness):
            yield self.cut(border_slc)

    def unique_border_colors(self, border_thickness: Optional[Shape5D] = None) -> "StaticLine":
        border_thickness = border_thickness or Shape5D.zero(**{key: 1 for key in "xyz" if self.shape[key] > 1})
        border_labels = StaticLine.empty(num_channels=self.shape.c)
        for border in self.get_borders(thickness=border_thickness):
            unique_labels = border.unique_colors()
            border_labels = border_labels.concatenate(unique_labels)
        return border_labels.unique_colors()

    def threshold(self: Arr, threshold: float) -> Arr:
        out = Array5D.allocate_like(self, dtype=np.bool)
        out_raw = out.raw(Point5D.LABELS)
        self_raw = self.raw(Point5D.LABELS)
        out_raw[self_raw >= threshold] = True
        out_raw[self_raw < threshold] = False
        return out

    def connected_components(self: Arr, background: int = 0, connectivity: str = "xyz") -> Arr:
        piece_shape = self.shape.updated(**{axis: 1 for axis in set("xyztc").difference(connectivity)})
        output = Array5D.allocate_like(self, dtype=np.int64)
        for piece in self.split(piece_shape):
            raw = piece.raw(connectivity)
            labeled_piece_raw = skmeasure.label(raw, background=background, connectivity=len(connectivity))
            labeled_piece_5d = Array5D(labeled_piece_raw, axiskeys=connectivity, location=piece.location)
            output.set(labeled_piece_5d)
        return output

    def paint_point(self, point: Point5D, value: Number, local: bool = False):
        point = point if local else point - self.location
        np_selection = tuple(int(v) for v in point.to_tuple(self.axiskeys))
        self._data[np_selection] = value

    def combine(self: Arr, others: Sequence[Arr]) -> Arr:
        out_roi = Interval5D.enclosing([self.interval] + [o.interval for o in others])
        out = self.allocate(slc=out_roi, dtype=self.dtype, axiskeys=self.axiskeys, value=0)
        out.set(self)
        for other in others:
            out.set(other)
        return out


class StaticData(Array5D):
    """An Array5D with a single time frame"""

    def __init__(self, arr: np.ndarray, axiskeys: str, location: Point5D = Point5D.zero()):
        super().__init__(arr=arr, axiskeys=axiskeys, location=location)
        assert self.shape.is_static


class ScalarData(Array5D):
    """An Array5D with a single channel"""

    def __init__(self, arr: np.ndarray, axiskeys: str, location: Point5D = Point5D.zero()):
        super().__init__(arr=arr, axiskeys=axiskeys, location=location)
        assert self.shape.is_scalar


class FlatData(Array5D):
    """An Array5D with less than 3 spacial dimensions having a size > 1"""

    def __init__(self, arr: np.ndarray, axiskeys: str, location: Point5D = Point5D.zero()):
        super().__init__(arr=arr, axiskeys=axiskeys, location=location)
        assert self.shape.is_flat


class LinearData(Array5D):
    """An Array5D with at most 1 spacial dimension having size > 1"""

    def __init__(self, arr: np.ndarray, axiskeys: str, location: Point5D = Point5D.zero()):
        super().__init__(arr=arr, axiskeys=axiskeys, location=location)
        assert self.shape.is_line
        line_axis = ""
        for axis in self.shape.present_spatial_axes.keys():
            line_axis = axis
            break
        self.line_axis = line_axis or "x"

    @property
    def length(self) -> float:
        return self.shape.volume

    @property
    def colors(self) -> Iterable["LinearData"]:
        return [LinearData(color, axiskeys="c") for color in self.linear_raw()]


class Image(StaticData, FlatData):
    """An Array5D representing a 2D image"""


class ScalarImage(Image, ScalarData):
    pass


class ScalarLine(LinearData, ScalarData):
    pass


class StaticLine(StaticData, LinearData):
    DEFAULT_AXES = "xc"

    @classmethod
    def empty(cls, num_channels: int, axiskeys: str = DEFAULT_AXES) -> "StaticLine":
        return StaticLine(np.zeros((0, num_channels)), axiskeys=axiskeys)

    def concatenate(self, *others: LinearData) -> "LinearData":
        raw_all = [self.linear_raw()] + [o.linear_raw() for o in others]
        data = np.concatenate(raw_all, axis=0)
        return self.rebuild(data, axiskeys=self.line_axis + "c")
