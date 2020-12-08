import itertools
import functools
from ndstructs.utils.JsonSerializable import Referencer
import operator
import numpy as np
from typing import Dict, Tuple, Iterator, List, Iterable, TypeVar, Type, Union, Optional, Callable, Any
from numbers import Number


from ndstructs.utils import JsonSerializable, Dereferencer, Referencer

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


INF = float("inf")
NINF = -INF


class Point5D(JsonSerializable):
    LABELS = "txyzc"
    SPATIAL_LABELS = "xyz"
    LABEL_MAP = {label: index for index, label in enumerate(LABELS)}
    DTYPE = np.float64

    def __init__(self, *, t: int = 0, x: int = 0, y: int = 0, z: int = 0, c: int = 0):
        self.x = x
        self.y = y
        self.z = z
        self.t = t
        self.c = c

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
        return tuple(type_converter(self[label]) for label in axis_order)

    def to_dict(self) -> Dict[str, float]:
        return {k: self[k] for k in self.LABELS}

    def to_np(self, axis_order: str = LABELS) -> np.ndarray:
        return np.asarray(self.to_tuple(axis_order))

    def __repr__(self) -> str:
        contents = ",".join((f"{label}:{val}" for label, val in self.to_dict().items()))
        return f"{self.__class__.__name__}({contents})"

    @staticmethod
    def zero(*, t: int = 0, x: int = 0, y: int = 0, z: int = 0, c: int = 0) -> "Point5D":
        return Point5D(t=t, x=x, y=y, z=z, c=c)

    @staticmethod
    def one(*, t: int = 1, x: int = 1, y: int = 1, z: int = 1, c: int = 1) -> "Point5D":
        return Point5D(t=t, x=x, y=y, z=z, c=c)

    def __getitem__(self, key: str) -> int:
        if key == "x":
            return self.x
        if key == "y":
            return self.y
        if key == "z":
            return self.z
        if key == "t":
            return self.t
        if key == "c":
            return self.c
        raise KeyError(key)

    def with_coord(
        self: PT,
        *,
        t: Optional[int] = None,
        c: Optional[int] = None,
        x: Optional[int] = None,
        y: Optional[int] = None,
        z: Optional[int] = None,
    ) -> PT:
        return self.__class__(
            t=t if t is not None else self.t,
            c=c if c is not None else self.c,
            x=x if x is not None else self.x,
            y=y if y is not None else self.y,
            z=z if z is not None else self.z,
        )

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
        minimum = minimum or self
        maximum = maximum or self
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
    def __init__(self, *, t: int = 1, x: int = 1, y: int = 1, z: int = 1, c: int = 1):
        assert all(coord >= 0 for coord in (x, y, z, t, c))
        super().__init__(t=t, x=x, y=y, z=z, c=c)

    @classmethod
    def create(cls, *, raw_shape: Tuple[int, ...], axiskeys: str) -> "Shape5D":
        MismatchingAxiskeysException.ensure_matching(raw_shape, axiskeys)
        return cls(**dict(zip(axiskeys, raw_shape)))

    @classmethod
    def hypercube(cls, length: int) -> "Shape5D":
        return cls(t=length, x=length, y=length, z=length, c=length)

    def __repr__(self) -> str:
        contents = ",".join((f"{label}:{val}" for label, val in self.to_dict().items() if val != 1))
        return f"{self.__class__.__name__}({contents or 1})"

    @property
    def spatial_axes(self) -> Dict[str, int]:
        return {k: self[k] for k in self.SPATIAL_LABELS}

    @property
    def missing_spatial_axes(self) -> Dict[str, int]:
        return {k: v for k, v in self.spatial_axes.items() if v == 1}

    @property
    def present_spatial_axes(self) -> Dict[str, int]:
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

    def to_slice_5d(self, offset: Point5D = Point5D.zero()) -> "Interval5D":
        return Interval5D.create_from_start_stop(offset, self + offset)

    @classmethod
    def from_point(cls: Type[PT], point: Point5D) -> PT:
        return cls(**{k: v or 1 for k, v in point.to_dict().items()})


INTERVALABLE = Union["Interval", int, None, Tuple[Optional[int], Optional[int]]]


class Interval:
    """A contiguous interval in space of indicies between start (inclusive) and stop (exclusive)"""

    def __init__(self, start: int = 0, stop: Optional[int] = None):
        self.start = start
        self.stop = stop
        assert self.stop >= self.start

    @classmethod
    def create(cls, value: INTERVALABLE) -> "Interval":
        if isinstance(value, int):
            return Interval(value, value + 1)
        if isinstance(value, Interval):
            return value
        if isinstance(value, tuple):
            return Interval(value[0] or 0, value[1])
        return Interval.all()

    def __eq__(self, other: INTERVALABLE) -> bool:
        other_interval = Interval.create(other)
        return self.start == other_interval.start and self.stop == other_interval.stop

    def __hash__(self) -> int:
        return hash((self.start, self.stop))

    @classmethod
    def all(cls) -> "Interval":
        return cls()

    @classmethod
    def zero(cls) -> "Interval":
        return cls(start=0, stop=1)

    def to_slice(self) -> slice:
        return slice(self.start, self.stop)

    def is_defined(self) -> bool:
        return self.stop != None

    def defined_with(self, limit: "Interval") -> "Interval":
        assert limit.is_defined()
        return Interval(self.start, self.stop if self.stop != INF else limit.stop)

    def contains(self, other: "Interval") -> bool:
        if self.stop == None or other.stop == None:
            return False
        return self.start <= other.start and self.stop >= other.stop

    def split(self, step: int, clamp: bool = True) -> Iterable["Interval"]:
        start = self.start
        while self.stop == None or start < self.stop:
            stop = start + step
            piece = Interval(start, stop)
            if clamp:
                piece = piece.clamped(self)
            yield piece
            start = stop

    def get_tiles(self, tile_side: int, clamp: bool) -> Iterable["Interval"]:
        start = (self.start // tile_side) * tile_side
        return Interval(start, self.stop).split(tile_side, clamp=clamp)

    def clamped(self, limits: "Interval") -> "Interval":
        if limits.stop == None:
            stop = self.stop
        elif self.stop == None:
            stop = limits.stop
        else:
            stop = min(self.stop, limits.stop)
        return Interval(max(self.start, limits.start), stop)

    def enlarged(self, radius: int) -> "Interval":
        return Interval(self.start - radius, None if self.stop == None else self.stop + radius)

    def translated(self, offset: int) -> "Interval":
        return Interval(self.start + offset, None if self.stop == None else self.stop + offset)


INTERVAL_5D = TypeVar("INTERVAL_5D", bound="Interval5D", covariant=True)


class Interval5D(JsonSerializable):
    """A labeled 5D interval"""

    def __init__(
        self,
        *,
        t: INTERVALABLE = Interval.all(),
        c: INTERVALABLE = Interval.all(),
        x: INTERVALABLE = Interval.all(),
        y: INTERVALABLE = Interval.all(),
        z: INTERVALABLE = Interval.all(),
    ):
        self.x = Interval.create(x)
        self.y = Interval.create(y)
        self.z = Interval.create(z)
        self.t = Interval.create(t)
        self.c = Interval.create(c)
        self.start = Point5D(x=self.x.start, y=self.y.start, z=self.z.start, t=self.t.start, c=self.c.start)

    def get_stop(self) -> Optional[Point5D]:
        x = self.x.stop
        y = self.y.stop
        z = self.z.stop
        t = self.t.stop
        c = self.c.stop
        if x is None or y is None or z is None or t is None or c is None:
            return None
        return Point5D(x=x, y=y, z=z, t=t, c=c)

    @staticmethod
    def zero(
        *,
        t: INTERVALABLE = Interval.zero(),
        c: INTERVALABLE = Interval.zero(),
        x: INTERVALABLE = Interval.zero(),
        y: INTERVALABLE = Interval.zero(),
        z: INTERVALABLE = Interval.zero(),
    ) -> "Interval5D":
        """Creates a slice with coords defaulting to slice(0, 1), except where otherwise specified"""
        return Interval5D(t=t, c=c, x=x, y=y, z=z)

    def relabeled(self: INTERVAL_5D, keymap: KeyMap) -> INTERVAL_5D:
        params = {target_key: self[src_key] for src_key, target_key in keymap.items()}
        return self.with_coord(**params)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Interval5D):
            return False
        return self.to_tuple(Point5D.LABELS) == other.to_tuple(Point5D.LABELS)

    def __hash__(self) -> int:
        return hash(self.to_tuple(Point5D.LABELS))

    def contains(self, other: "Interval5D") -> bool:
        return all(self[k].contains(other[k]) for k in Point5D.LABELS)

    def is_defined(self) -> bool:
        return all(i.is_defined() for i in self.to_tuple(Point5D.LABELS))

    def defined_with(self: INTERVAL_5D, limits: Union[Shape5D, "Interval5D"]) -> INTERVAL_5D:
        """Interval5D can have intervals which are open to interpretation, like Interval(0, None). This method
        forces those slices expand into their interpretation within the boundaries of 'limits'"""
        limits_interval = limits if isinstance(limits, Interval5D) else limits.to_slice_5d()
        return self.with_coord(**{k: self[k].defined_with(limits_interval[k]) for k in Point5D.LABELS})

    def to_dict(self) -> Dict[str, Interval]:
        return {k: self[k] for k in Point5D.LABELS}

    @staticmethod
    def all(
        t: Interval = Interval.all(),
        c: Interval = Interval.all(),
        x: Interval = Interval.all(),
        y: Interval = Interval.all(),
        z: Interval = Interval.all(),
    ) -> "Interval5D":
        return Interval5D(t=t, c=c, x=x, y=y, z=z)

    @classmethod
    def make_intervals(cls, start: Point5D, stop: Point5D) -> Dict[str, Interval]:
        return {k: Interval(int(start[k]), int(stop[k])) for k in Point5D.LABELS}

    @staticmethod
    def create_from_start_stop(start: Point5D, stop: Point5D) -> "Interval5D":
        return Interval5D(**Interval5D.make_intervals(start, stop))

    @staticmethod
    def from_json_data(data: dict, dereferencer: Optional[Dereferencer] = None) -> "Interval5D":
        start = Point5D.from_json_data(data["start"])
        stop = Point5D.from_json_data(data["stop"])
        return Interval5D.create_from_start_stop(start, stop)

    def to_json_data(self, referencer: Referencer = lambda obj: None) -> dict:
        self_tuple = self.to_tuple(Point5D.LABELS)
        return {"start": self_tuple[0], "stop": self_tuple[1]}

    def from_start_stop(self: INTERVAL_5D, start: Point5D, stop: Point5D) -> INTERVAL_5D:
        slices = self.make_intervals(start, stop)
        return self.with_coord(**slices)

    def split(self: INTERVAL_5D, block_shape: Shape5D) -> Iterator[INTERVAL_5D]:
        """Splits self into multiple Interval5D instances, starting from self.start. Every piece shall have
        shape == block_shape excedpt for the last one, which will be clamped to self.stop"""

        yield from itertools.product([self[k].split(int(block_shape[k])) for k in Point5D.LABELS])

    def get_tiles(self: INTERVAL_5D, tile_shape: Shape5D, clamp: bool) -> Iterator[INTERVAL_5D]:
        """Gets all tiles that would cover the entirety of self. Tiles that overflow self can be clamped
        by setting `clamp` to True"""

        yield from itertools.product([self[k].get_tiles(int(tile_shape[k]), clamp=clamp) for k in Point5D.LABELS])

    def __getitem__(self, key: str) -> Interval:
        if key == "x":
            return self.x
        if key == "y":
            return self.y
        if key == "z":
            return self.z
        if key == "t":
            return self.t
        if key == "c":
            return self.c
        raise KeyError(key)

    # override this in subclasses so that it returns an instance of self.__class__
    def with_coord(
        self: INTERVAL_5D,
        *,
        t: INTERVALABLE = None,
        c: INTERVALABLE = None,
        x: INTERVALABLE = None,
        y: INTERVALABLE = None,
        z: INTERVALABLE = None,
    ) -> INTERVAL_5D:
        return self.__class__(
            t=self.t if t is None else t,
            c=self.c if c is None else c,
            x=self.x if x is None else x,
            y=self.y if y is None else y,
            z=self.z if z is None else z,
        )

    def with_full_c(self: INTERVAL_5D) -> INTERVAL_5D:
        return self.with_coord(c=Interval.all())

    def clamped(self: INTERVAL_5D, roi: Union[Shape5D, "Interval5D"]) -> INTERVAL_5D:
        interv = roi if isinstance(roi, Interval5D) else roi.to_slice_5d()
        return self.with_coord(**{k: self[k].clamped(interv[k]) for k in Point5D.LABELS})

    def enlarged(self: INTERVAL_5D, radius: Point5D) -> INTERVAL_5D:
        return self.with_coord(**{k: self[k].enlarged(int(radius[k])) for k in Point5D.LABELS})

    def translated(self: INTERVAL_5D, offset: Point5D) -> INTERVAL_5D:
        return self.with_coord(**{k: self[k].translated(offset[k]) for k in Point5D.LABELS})

    def to_slices(self, axis_order: str = Point5D.LABELS) -> Tuple[slice, ...]:
        return tuple(self[axis].to_slice() for axis in axis_order)

    def to_tuple(self, axis_order: str) -> Tuple[Interval, ...]:
        return tuple(self[k] for k in axis_order)

    def to_start_stop(self, axis_order: str) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        start = tuple(self[k].start for k in axis_order)
        stop = tuple(self[k].stop for k in axis_order)
        return (start, stop)

    def to_ilastik_cutout_subregion(self, axis_order: str) -> str:
        return str(list(self.to_start_stop(axis_order=axis_order)))

    def __repr__(self) -> str:
        interval_reprs = ", ".join(
            f"{k}:{self[k].start}_{self[k].stop}" for k in Point5D.LABELS if self[k] != Interval.all()
        )
        return f"{self.__class__.__name__}({interval_reprs})"

    def get_borders(self: INTERVAL_5D, thickness: Shape5D) -> Iterable[INTERVAL_5D]:
        """Returns subslices of self, such that these subslices are at the borders
        of self (i.e.: touching the start or end of self)

        No axis of thickness should exceed self.shape[axis], since the subslices must be contained in self
        Axis where thickness[axis] == 0 will produce no borders:
            slc.get_borders(Interval5D.zero(x=1, y=1)) will produce 4 borders (left, right, top, bottom)
        If, for any axis, thickness[axis] == self.shape[axis], then there will be duplicated borders in the output
        """
        thickness_interval = thickness.to_slice_5d(offset=self.start)
        assert all(self[k].contains(thickness_interval[k]) for k in Point5D.LABELS)
        # FIXME: I haven't ported this yet!!!!!
        for axis, axis_thickness in thickness.to_dict().items():
            if axis_thickness == 0:
                continue
            slc = self[axis]
            yield self.with_coord(**{axis: slice(slc.start, slc.start + axis_thickness)})
            yield self.with_coord(**{axis: slice(slc.stop - axis_thickness, slc.stop)})

    def mod_tile(self: INTERVAL_5D, tile_shape: Shape5D) -> INTERVAL_5D:
        assert self.shape <= tile_shape
        offset = self.start - (self.start % tile_shape)
        return self.from_start_stop(self.start - offset, self.stop - offset)

    def get_neighboring_tiles(self: INTERVAL_5D, tile_shape: Shape5D) -> Iterator[INTERVAL_5D]:
        assert self.shape <= tile_shape
        for axis in Point5D.LABELS:
            for axis_offset in (tile_shape[axis], -tile_shape[axis]):
                offset = Point5D.zero(**{axis: axis_offset})
                yield self.translated(offset)

    def get_neighbor_tile_adjacent_to(
        self: INTERVAL_5D, *, anchor: "Interval5D", tile_shape: Shape5D
    ) -> Optional[INTERVAL_5D]:
        assert self.contains(anchor)

        direction_axis: Optional[str] = None
        for axis in Point5D.LABELS:
            if anchor[axis] != self[axis]:
                if direction_axis:
                    raise ValueError(f"Bad anchor for slice {self}: {anchor}")
                direction_axis = axis

        if direction_axis is None:
            raise ValueError(f"Bad anchor for slice {self}: {anchor}")

        # a neighbor has all but one coords equal
        offset = Point5D.zero(**{direction_axis: tile_shape[direction_axis]})

        if anchor[direction_axis].stop == self[direction_axis].stop:
            if self.shape != tile_shape:  # Getting a further tile from a partial tile
                return None
            return self.translated(offset)
        if anchor[direction_axis].start == self[direction_axis].start:
            if self.start - offset < Point5D.zero():  # no negative neighbors
                return None
            return self.translated(-offset)

        raise ValueError(f"Bad anchor for slice {self}: {anchor}")

    @staticmethod
    def enclosing(points: Iterable[Union[Point5D, "Interval5D"]]) -> "Interval5D":
        all_points = []
        for p in points:
            if isinstance(p, Point5D):
                all_points.append(p)
            else:
                all_points += [p.start, p.stop - Point5D.one()]
        if not all_points:
            return Interval5D.create_from_start_stop(Point5D.zero(), Point5D.zero())
        start = Point5D.min_coords(all_points)
        stop = Point5D.max_coords(all_points) + Point5D.one()
        return Interval5D.create_from_start_stop(start=start, stop=stop)
