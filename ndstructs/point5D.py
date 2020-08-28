from collections import OrderedDict
from itertools import product
import functools
import operator
import numpy as np
from typing import Dict, Tuple, Iterator, List, Iterable, TypeVar, Type, Union, Optional, Callable, Any
from numbers import Number


from ndstructs.utils import JsonSerializable

PT = TypeVar("PT", bound="Point5D", covariant=True)
PT_OPERABLE = Union["Point5D", Number]

T = TypeVar("T")


class KeyMap:
    def __init__(self, x: str = "x", y: str = "y", z: str = "z", t: str = "t", c: str = "c"):
        self._map = {"x": x, "y": y, "z": z, "t": t, "c": c}
        assert set(self._map.values()) == set("xyztc")

    def items(self) -> Iterable[Tuple[str, str]]:
        yield from self._map.items()

    def map_axiskeys(self, axiskeys: str) -> str:
        return "".join(self._map[key] for key in axiskeys)

    def reversed(self) -> "KeyMap":
        return KeyMap(**{v: k for k, v in self._map.items()})


class Point5D(JsonSerializable):
    LABELS = "txyzc"
    SPATIAL_LABELS = "xyz"
    LABEL_MAP = {label: index for index, label in enumerate(LABELS)}
    DTYPE = np.float64
    INF = float("inf")
    NINF = -INF

    def __init__(self, *, t: float = 0, x: float = 0, y: float = 0, z: float = 0, c: float = 0):
        assert all(
            v in (self.INF, self.NINF) or int(v) == v for v in (t, c, x, y, z)
        ), f"Point5D accepts only ints or 'inf' {(t,c,x,y,z)}"
        self._coords = {"t": t, "c": c, "x": x, "y": y, "z": z}

    def __hash__(self) -> int:
        return hash(self.to_tuple(self.LABELS))

    @classmethod
    def from_tuple(cls: Type[PT], tup: Tuple[float, ...], labels: str) -> PT:
        assert len(tup) == len(labels)
        return cls(**{label: value for label, value in zip(labels, tup)})

    @classmethod
    def from_np(cls: Type[PT], arr: np.ndarray, labels: str) -> PT:
        return cls.from_tuple(tuple(float(e) for e in arr), labels)

    def to_tuple(self, axis_order: str, type_converter: Callable[[float], T] = lambda x: float(x)) -> Tuple[T, ...]:
        return tuple(type_converter(self._coords[label]) for label in axis_order)

    def to_dict(self) -> Dict[str, float]:
        return self._coords.copy()

    def to_np(self, axis_order: str = LABELS) -> np.ndarray:
        return np.asarray(self.to_tuple(axis_order))

    def __repr__(self) -> str:
        contents = ",".join((f"{label}:{val}" for label, val in self._coords.items()))
        return f"{self.__class__.__name__}({contents})"

    @staticmethod
    def inf(*, t: float = None, x: float = None, y: float = None, z: float = None, c: float = None) -> "Point5D":
        return Point5D(
            t=Point5D.INF if t is None else t,
            x=Point5D.INF if x is None else x,
            y=Point5D.INF if y is None else y,
            z=Point5D.INF if z is None else z,
            c=Point5D.INF if c is None else c,
        )

    @staticmethod
    def ninf(*, t: float = None, x: float = None, y: float = None, z: float = None, c: float = None) -> "Point5D":
        return Point5D(
            t=Point5D.NINF if t is None else t,
            x=Point5D.NINF if x is None else x,
            y=Point5D.NINF if y is None else y,
            z=Point5D.NINF if z is None else z,
            c=Point5D.NINF if c is None else c,
        )

    @staticmethod
    def zero(*, t: float = 0, x: float = 0, y: float = 0, z: float = 0, c: float = 0) -> "Point5D":
        return Point5D(t=t or 0, x=x or 0, y=y or 0, z=z or 0, c=c or 0)

    @staticmethod
    def one(*, t: float = 1, x: float = 1, y: float = 1, z: float = 1, c: float = 1) -> "Point5D":
        return Point5D(t=t, x=x, y=y, z=z, c=c)

    def __getitem__(self, key: str) -> float:
        return self._coords[key]

    @property
    def t(self) -> float:
        return self["t"]

    @property
    def x(self) -> float:
        return self["x"]

    @property
    def y(self) -> float:
        return self["y"]

    @property
    def z(self) -> float:
        return self["z"]

    @property
    def c(self) -> float:
        return self["c"]

    def with_coord(
        self: PT,
        *,
        t: Optional[float] = None,
        c: Optional[float] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
    ) -> PT:
        params = self.to_dict()
        params["t"] = t if t is not None else params["t"]
        params["c"] = c if c is not None else params["c"]
        params["x"] = x if x is not None else params["x"]
        params["y"] = y if y is not None else params["y"]
        params["z"] = z if z is not None else params["z"]
        return self.__class__(**params)

    def __np_op(self: PT, other: PT_OPERABLE, op: str) -> PT:
        if isinstance(other, Point5D):
            raw_value = other.to_np(self.LABELS)
        else:
            raw_value = other
        raw = getattr(self.to_np(self.LABELS), op)(raw_value)
        return self.from_np(raw, self.LABELS)

    def _compare(self, other: PT_OPERABLE, op: str) -> bool:
        return all(self.__np_op(other, op).to_tuple(self.LABELS))

    def __gt__(self: PT, other: PT_OPERABLE) -> bool:
        return self._compare(other, "__gt__")

    def __ge__(self: PT, other: PT_OPERABLE) -> bool:
        return self._compare(other, "__ge__")

    def __lt__(self: PT, other: PT_OPERABLE) -> bool:
        return self._compare(other, "__lt__")

    def __le__(self: PT, other: PT_OPERABLE) -> bool:
        return self._compare(other, "__le__")

    def __eq__(self: PT, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._compare(other, "__eq__")

    def __ne__(self: PT, other: object) -> bool:
        return not self.__eq__(other)

    def __sub__(self: PT, other: PT_OPERABLE) -> PT:
        return self.__np_op(other, "__sub__")

    def __neg__(self) -> "Point5D":
        raw = self.to_np(self.LABELS)
        return Point5D.from_np(-raw, self.LABELS)

    def __mod__(self: PT, other: PT_OPERABLE) -> PT:
        return self.__np_op(other, "__mod__")

    def __add__(self: PT, other: PT_OPERABLE) -> PT:
        return self.__np_op(other, "__add__")

    def __floordiv__(self: PT, other: PT_OPERABLE) -> PT:
        return self.__np_op(other, "__floordiv__")

    def __mul__(self: PT, other: PT_OPERABLE) -> PT:
        return self.__np_op(other, "__mul__")

    def clamped(self: PT, minimum: "Point5D" = None, maximum: "Point5D" = None) -> PT:
        minimum = minimum or Point5D.ninf()
        maximum = maximum or Point5D.inf()
        result = np.maximum(self.to_np(self.LABELS), minimum.to_np(self.LABELS))
        result = np.minimum(result, maximum.to_np(self.LABELS))
        return self.__class__(**{label: val for label, val in zip(self.LABELS, result)})

    def as_shape(self) -> "Shape5D":
        return Shape5D(**self.to_dict())

    @classmethod
    def as_ceil(cls: Type[PT], arr: np.ndarray, axis_order: str = LABELS) -> PT:
        raw = np.ceil(arr)
        return cls.from_np(raw, axis_order)

    @classmethod
    def as_floor(cls: Type[PT], arr: np.ndarray, axis_order: str = LABELS) -> PT:
        raw = np.floor(arr)
        return cls.from_np(raw, axis_order)

    def relabeled(self: PT, keymap: KeyMap) -> PT:
        params = {target_key: self[src_key] for src_key, target_key in keymap.items()}
        return self.with_coord(**params)

    def interpolate_until(self, endpoint: "Point5D") -> Iterable["Point5D"]:
        start = self.to_np(self.LABELS)
        end = endpoint.to_np(self.LABELS)
        delta = end - start
        steps = np.max(np.absolute(delta))
        if steps == 0:
            yield self
            return
        increment = delta / steps
        for i in range(int(steps)):
            yield Point5D.from_np(np.around(start + (increment * i)), labels=self.LABELS)
        yield endpoint

    @staticmethod
    def min_coords(points: Iterable["Point5D"]) -> "Point5D":
        return Point5D.zero(**{key: min(vox[key] for vox in points) for key in Point5D.LABELS})

    @staticmethod
    def max_coords(points: Iterable["Point5D"]) -> "Point5D":
        return Point5D.zero(**{key: max(vox[key] for vox in points) for key in Point5D.LABELS})


class MismatchingAxiskeysException(Exception):
    @classmethod
    def ensure_matching(cls, raw_shape: Tuple[int, ...], axiskeys: str):
        if len(raw_shape) != len(axiskeys):
            raise cls(f"Shape {raw_shape} does not fit axiskeys {axiskeys}")


class Shape5D(Point5D):
    def __init__(cls, *, t: float = 1, x: float = 1, y: float = 1, z: float = 1, c: float = 1):
        super().__init__(t=t, x=x, y=y, z=z, c=c)

    @classmethod
    def create(cls, *, raw_shape: Tuple[int, ...], axiskeys: str) -> "Shape5D":
        MismatchingAxiskeysException.ensure_matching(raw_shape, axiskeys)
        return cls(**dict(zip(axiskeys, raw_shape)))

    @classmethod
    def hypercube(cls, length: int) -> "Shape5D":
        return cls(t=length, x=length, y=length, z=length, c=length)

    def __repr__(self) -> str:
        contents = ",".join((f"{label}:{val}" for label, val in self._coords.items() if val != 1))
        return f"{self.__class__.__name__}({contents or 1})"

    def to_tuple(self, axis_order: str) -> Tuple[float, ...]:
        return tuple(int(v) for v in super().to_tuple(axis_order))

    @property
    def spatial_axes(self) -> Dict[str, float]:
        return {k: self._coords[k] for k in self.SPATIAL_LABELS}

    @property
    def missing_spatial_axes(self) -> Dict[str, float]:
        return {k: v for k, v in self.spatial_axes.items() if v == 1}

    @property
    def present_spatial_axes(self) -> Dict[str, float]:
        return {k: v for k, v in self.spatial_axes.items() if k not in self.missing_spatial_axes}

    @property
    def is_static(self) -> bool:
        return self.t == 1

    @property
    def is_flat(self) -> bool:
        return len(self.present_spatial_axes) <= 2

    @property
    def is_line(self) -> bool:
        return len(self.present_spatial_axes) <= 1

    @property
    def is_scalar(self) -> bool:
        return self.c == 1

    @property
    def volume(self) -> float:
        return self.x * self.y * self.z

    @property
    def hypervolume(self) -> float:
        return functools.reduce(operator.mul, self.to_tuple(Point5D.LABELS))

    def to_slice_5d(self, offset: Point5D = Point5D.zero()) -> "Slice5D":
        return Slice5D.create_from_start_stop(offset, self + offset)

    @classmethod
    def from_point(cls: Type[PT], point: Point5D) -> PT:
        return cls(**{k: v or 1 for k, v in point.to_dict().items()})


SLC = TypeVar("SLC", bound="Slice5D", covariant=True)
SLC_PARAM = Union[slice, float]


class Slice5D(JsonSerializable):
    """A labeled 5D slice"""

    DTYPE = np.int64

    @classmethod
    def ensure_slice(cls, value: SLC_PARAM) -> slice:
        if isinstance(value, slice):
            return value
        i = int(value)
        return slice(value, i + 1)

    def __init__(
        self,
        *,
        t: SLC_PARAM = slice(None),
        c: SLC_PARAM = slice(None),
        x: SLC_PARAM = slice(None),
        y: SLC_PARAM = slice(None),
        z: SLC_PARAM = slice(None),
    ):
        self._slices = {
            "t": self.ensure_slice(t),
            "c": self.ensure_slice(c),
            "x": self.ensure_slice(x),
            "y": self.ensure_slice(y),
            "z": self.ensure_slice(z),
        }

        self.start = Point5D.ninf(**{label: slc.start for label, slc in self._slices.items()})
        self.stop = Point5D.inf(**{label: slc.stop for label, slc in self._slices.items()})

    @staticmethod
    def zero(*, t: SLC_PARAM = 0, c: SLC_PARAM = 0, x: SLC_PARAM = 0, y: SLC_PARAM = 0, z: SLC_PARAM = 0) -> "Slice5D":
        """Creates a slice with coords defaulting to slice(0, 1), except where otherwise specified"""
        return Slice5D(t=t, c=c, x=x, y=y, z=z)

    def relabeled(self: SLC, keymap: KeyMap) -> SLC:
        params = {target_key: self[src_key] for src_key, target_key in keymap.items()}
        return self.with_coord(**params)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Slice5D):
            return False
        return self.start == other.start and self.stop == other.stop

    def __hash__(self) -> int:
        return hash((self.start, self.stop))

    def contains(self, other: "Slice5D") -> bool:
        assert other.is_defined()
        return self.start <= other.start and self.stop >= other.stop

    def is_defined(self) -> bool:
        if any(slc.stop is None for slc in self._slices.values()):
            return False
        if any(slc.start is None for slc in self._slices.values()):
            return False
        return True

    def defined_with(self: SLC, limits: Union[Shape5D, "Slice5D"]) -> SLC:
        """Slice5D can have slices which are open to interpretation, like slice(None). This method
        forces those slices expand into their interpretation within an array of shape 'shape'"""
        limits_slice = limits if isinstance(limits, Slice5D) else limits.to_slice_5d()
        assert limits_slice.is_defined()
        params = {}
        for key in Point5D.LABELS:
            this_slc = self[key]
            limit_slc = limits_slice[key]

            start = limit_slc.start if this_slc.start is None else this_slc.start
            stop = limit_slc.stop if this_slc.stop is None else this_slc.stop
            params[key] = slice(start, stop)
        return self.with_coord(**params)

    def to_dict(self) -> Dict[str, slice]:
        return self._slices.copy()

    @staticmethod
    def all(
        t: SLC_PARAM = slice(None),
        c: SLC_PARAM = slice(None),
        x: SLC_PARAM = slice(None),
        y: SLC_PARAM = slice(None),
        z: SLC_PARAM = slice(None),
    ) -> "Slice5D":
        return Slice5D(t=t, c=c, x=x, y=y, z=z)

    @classmethod
    def make_slices(cls, start: Point5D, stop: Point5D) -> Dict[str, slice]:
        slices = {}
        for label in Point5D.LABELS:
            slice_start = None if start[label] == Point5D.NINF else start[label]
            slice_stop = None if stop[label] == Point5D.INF else stop[label]
            slices[label] = slice(slice_start, slice_stop)
        return slices

    @staticmethod
    def create_from_start_stop(start: Point5D, stop: Point5D) -> "Slice5D":
        return Slice5D(**Slice5D.make_slices(start, stop))

    @staticmethod
    def from_json_data(data: dict) -> "Slice5D":
        start = Point5D.from_json_data(data["start"])
        stop = Point5D.from_json_data(data["stop"])
        return Slice5D.create_from_start_stop(start, stop)

    def to_json_data(self, referencer: Callable[[Any], str] = lambda obj: None) -> dict:
        return {"start": self.start.to_json_data(), "stop": self.stop.to_json_data()}

    def from_start_stop(self: SLC, start: Point5D, stop: Point5D) -> SLC:
        slices = self.make_slices(start, stop)
        return self.with_coord(**slices)

    def _ranges(self, block_shape: Shape5D) -> Iterator[List[float]]:
        starts = self.start.to_np(Point5D.LABELS)
        ends = self.stop.to_np(Point5D.LABELS)
        steps = block_shape.to_np(Point5D.LABELS)
        for start, end, step in zip(starts, ends, steps):
            yield list(np.arange(start, end, step))

    def split(self: SLC, block_shape: Shape5D) -> Iterator[SLC]:
        assert self.is_defined()
        for begin_tuple in product(*self._ranges(block_shape)):
            start = Point5D.from_tuple(begin_tuple, Point5D.LABELS)
            stop = (start + block_shape).clamped(maximum=self.stop)
            yield self.from_start_stop(start, stop)

    def get_tiles(self: SLC, tile_shape: Shape5D) -> Iterator[SLC]:
        assert self.is_defined()
        start = Point5D.as_floor(self.start.to_np() / tile_shape.to_np()) * tile_shape
        stop = Point5D.as_ceil(self.stop.to_np() / tile_shape.to_np()) * tile_shape
        return self.from_start_stop(start, stop).split(tile_shape)

    @property
    def t(self) -> slice:
        return self._slices["t"]

    @property
    def c(self) -> slice:
        return self._slices["c"]

    @property
    def x(self) -> slice:
        return self._slices["x"]

    @property
    def y(self) -> slice:
        return self._slices["y"]

    @property
    def z(self) -> slice:
        return self._slices["z"]

    def __getitem__(self, key: str) -> slice:
        return self._slices[key]

    def with_coord(
        self: SLC,
        *,
        t: Optional[SLC_PARAM] = None,
        c: Optional[SLC_PARAM] = None,
        x: Optional[SLC_PARAM] = None,
        y: Optional[SLC_PARAM] = None,
        z: Optional[SLC_PARAM] = None,
    ) -> SLC:
        params = {}
        params["t"] = self.t if t is None else t
        params["c"] = self.c if c is None else c
        params["x"] = self.x if x is None else x
        params["y"] = self.y if y is None else y
        params["z"] = self.z if z is None else z
        return self.__class__(**params)

    def with_full_c(self: SLC) -> SLC:
        return self.with_coord(c=slice(None))

    @property
    def shape(self) -> Shape5D:
        assert self.is_defined()
        return Shape5D(**(self.stop - self.start).to_dict())

    def clamped(self: SLC, roi: Union[Shape5D, "Slice5D"]) -> SLC:
        slc = roi if isinstance(roi, Slice5D) else roi.to_slice_5d()
        return self.from_start_stop(self.start.clamped(slc.start, slc.stop), self.stop.clamped(slc.start, slc.stop))

    def enlarged(self: SLC, radius: Point5D) -> SLC:
        start = self.start - radius
        stop = self.stop + radius
        return self.from_start_stop(start, stop)

    def translated(self: SLC, offset: Point5D) -> SLC:
        return self.from_start_stop(self.start + offset, self.stop + offset)

    def to_slices(self, axis_order: str = Point5D.LABELS) -> Tuple[slice, ...]:
        slices = []
        for axis in axis_order:
            slc = self._slices[axis]
            start = slc.start if slc.start is None else int(slc.start)
            stop = slc.stop if slc.stop is None else int(slc.stop)
            slices.append(slice(start, stop))
        return tuple(slices)

    def to_tuple(self, axis_order: str) -> Tuple[float, ...]:
        assert self.is_defined()
        return (self.start.to_np(axis_order), self.stop.to_np(axis_order))

    def to_ilastik_cutout_subregion(self, axiskeys: str) -> str:
        start = [slc.start for slc in self.to_slices(axiskeys)]
        stop = [slc.stop for slc in self.to_slices(axiskeys)]
        return str([tuple(start), tuple(stop)])

    def __repr__(self) -> str:
        slice_reprs = []
        starts = self.start.to_tuple(Point5D.LABELS)
        stops = self.stop.to_tuple(Point5D.LABELS)
        for label, start, stop in zip(Point5D.LABELS, starts, stops):
            if start == Point5D.NINF and stop == Point5D.INF:
                continue
            if stop - start == 1:
                label_repr = str(int(start))
            else:
                start_str = int(start) if start != Point5D.NINF else start
                stop_str = int(stop) if stop != Point5D.INF else stop
                label_repr = f"{start_str}_{stop_str}"
            slice_reprs.append(f"{label}:{label_repr}")
        return ",".join(slice_reprs)

    def get_borders(self: SLC, thickness: Shape5D) -> Iterable[SLC]:
        assert self.shape >= thickness
        for axis, axis_thickness in thickness.to_dict().items():
            if axis_thickness == 0:
                continue
            slc = self[axis]
            yield self.with_coord(**{axis: slice(slc.start, slc.start + axis_thickness)})
            if self.shape[axis] > thickness[axis]:
                yield self.with_coord(**{axis: slice(slc.stop - axis_thickness, slc.stop)})

    def mod_tile(self: SLC, tile_shape: Shape5D) -> SLC:
        assert self.is_defined()
        assert self.shape <= tile_shape
        offset = self.start - (self.start % tile_shape)
        return self.from_start_stop(self.start - offset, self.stop - offset)

    def get_neighboring_tiles(self: SLC, tile_shape: Shape5D) -> Iterator[SLC]:
        assert self.is_defined()
        assert self.shape <= tile_shape
        for axis in Point5D.LABELS:
            for axis_offset in (tile_shape[axis], -tile_shape[axis]):
                offset = Point5D.zero(**{axis: axis_offset})
                yield self.translated(offset)

    @staticmethod
    def enclosing(points: Iterable[Union[Point5D, "Slice5D"]]) -> "Slice5D":
        all_points = []
        for p in points:
            if isinstance(p, Point5D):
                all_points.append(p)
            else:
                all_points += [p.start, p.stop - 1]
        if not all_points:
            return Slice5D.from_start_stop(Point5D.zero(), Point5D.zero())
        start = Point5D.min_coords(all_points)
        stop = Point5D.max_coords(all_points) + 1
        return Slice5D.create_from_start_stop(start=start, stop=stop)
