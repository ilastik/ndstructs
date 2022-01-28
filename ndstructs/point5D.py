# pyright: strict

import itertools
import functools
import operator
from typing import ClassVar, Dict, Mapping, Tuple, Iterator, List, Iterable, TypeVar, Type, Union, Optional, cast
from typing_extensions import Final
from math import floor, ceil

import numpy as np

from ndstructs.utils.json_serializable import JsonObject, ensureJsonInt, JsonValue, ensureJsonObject

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


PT = TypeVar("PT", bound="Point5D")
PT_OPERABLE = Union["Point5D", int]


class Point5D:
    LABELS : Final[str] = "txyzc"  # if you change this order, also change self._array order
    SPATIAL_LABELS : Final[str] = "xyz"
    LABEL_MAP: ClassVar[Mapping[str, int]] = {label: index for index, label in enumerate(LABELS)}

    x: int
    y: int
    z: int
    t: int
    c: int

    def __init__(self, *, t: int = 0, x: int = 0, y: int = 0, z: int = 0, c: int = 0):
        self.x = x
        self.y = y
        self.z = z
        self.t = t
        self.c = c
        self._array = np.asarray([t, x, y, z, c])

    @classmethod
    def from_json_value(cls: Type[PT], data: JsonValue) -> PT:
        data_dict = ensureJsonObject(data)
        params = {k: ensureJsonInt(data_dict[k]) for k in Point5D.LABELS if k in data_dict}
        return cls(**params)

    def to_json_value(self) -> JsonObject:
        return self.to_dict()

    def __hash__(self) -> int:
        return hash(self.to_tuple(self.LABELS))

    @classmethod
    def from_tuple(cls: Type[PT], tup: Tuple[int, ...], labels: str) -> PT:
        if len(tup) != len(labels):
            raise ValueError(f"Mismatched args: {tup} , {labels}")
        return cls(**{label: value for label, value in zip(labels, tup)})

    @classmethod
    def from_np(cls: Type[PT], arr: np.ndarray, labels: str) -> PT:
        return cls.from_tuple(tuple(int(e) for e in arr), labels)

    def to_tuple(self, axis_order: str) -> Tuple[int, ...]:
        return tuple(self[label] for label in axis_order)

    def to_dict(self) -> Dict[str, int]:
        return {k: self[k] for k in self.LABELS}

    def to_np(self, axis_order: str = LABELS) -> np.ndarray:
        return np.asarray(self.to_tuple(axis_order))

    def __repr__(self) -> str:
        contents = ",".join((f"{label}:{val}" for label, val in self.to_dict().items()))
        return f"{self.__class__.__name__}({contents})"

    @classmethod
    def zero(cls: Type[PT], *, t: int = 0, x: int = 0, y: int = 0, z: int = 0, c: int = 0) -> PT:
        return cls(t=t, x=x, y=y, z=z, c=c)

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

    def updated(
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

    def __gt__(self, other: PT_OPERABLE) -> bool:
        return self._compare(other, "__gt__")

    def __ge__(self, other: PT_OPERABLE) -> bool:
        return self._compare(other, "__ge__")

    def __lt__(self, other: PT_OPERABLE) -> bool:
        return self._compare(other, "__lt__")

    def __le__(self, other: PT_OPERABLE) -> bool:
        return self._compare(other, "__le__")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._compare(other, "__eq__")

    def __ne__(self, other: object) -> bool:
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

    def clamped(self: PT, minimum: Optional["Point5D"] = None, maximum: Optional["Point5D"] = None) -> PT:
        result = self.to_np(self.LABELS)
        if minimum is not None:
            result = np.maximum(self.to_np(self.LABELS), minimum.to_np(self.LABELS))
        if maximum is not None:
            result = np.minimum(result, maximum.to_np(self.LABELS))
        return self.from_np(cast(np.ndarray, result), labels=self.LABELS)

    def as_shape(self) -> "Shape5D":
        return Shape5D(**self.to_dict())

    def relabeled(self: PT, keymap: KeyMap) -> PT:
        params = {target_key: self[src_key] for src_key, target_key in keymap.items()}
        return self.updated(**params)

    def interpolate_until(self, endpoint: "Point5D") -> Iterable["Point5D"]:
        start = self.to_np(self.LABELS)
        end = endpoint.to_np(self.LABELS)
        delta : np.ndarray = end - start
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
    def volume(self) -> int:
        return self.x * self.y * self.z

    @property
    def hypervolume(self) -> int:
        return functools.reduce(operator.mul, self.to_tuple(Point5D.LABELS))

    def to_interval5d(self, offset: Point5D = Point5D.zero()) -> "Interval5D":
        return Interval5D.create_from_start_stop(offset, self + offset)

    @classmethod
    def from_point(cls: Type[PT], point: Point5D) -> PT:
        return cls(**{k: v or 1 for k, v in point.to_dict().items()})


INTERVAL = Tuple[int, int]
SPAN = Union[int, INTERVAL]

INTERVAL_5D = TypeVar("INTERVAL_5D", bound="Interval5D")


class Interval5D:
    """A labeled 5D interval"""

    x : INTERVAL
    y : INTERVAL
    z : INTERVAL
    t : INTERVAL
    c : INTERVAL
    start: Point5D
    stop: Point5D

    def __init__(self, *, t: SPAN, c: SPAN, x: SPAN, y: SPAN, z: SPAN):
        self.x = (x, x + 1) if isinstance(x, int) else x
        self.y = (y, y + 1) if isinstance(y, int) else y
        self.z = (z, z + 1) if isinstance(z, int) else z
        self.t = (t, t + 1) if isinstance(t, int) else t
        self.c = (c, c + 1) if isinstance(c, int) else c
        if any(interval[0] > interval[1] for interval in (self.x, self.y, self.z, self.t, self.c)):
            raise ValueError(f"Intervals must have start <= stop")
        self.start = Point5D(x=self.x[0], y=self.y[0], z=self.z[0], t=self.t[0], c=self.c[0])
        self.stop = Point5D(x=self.x[1], y=self.y[1], z=self.z[1], t=self.t[1], c=self.c[1])

    @staticmethod
    def zero(*, t: SPAN = 0, c: SPAN = 0, x: SPAN = 0, y: SPAN = 0, z: SPAN = 0) -> "Interval5D":
        """Creates a slice with coords defaulting to slice(0, 1), except where otherwise specified"""
        return Interval5D(t=t, c=c, x=x, y=y, z=z)

    def relabeled(self: INTERVAL_5D, keymap: KeyMap) -> INTERVAL_5D:
        params = {target_key: self[src_key] for src_key, target_key in keymap.items()}
        return self.updated(**params)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Interval5D):
            return False
        return self.start == other.start and self.stop == other.stop

    def __hash__(self) -> int:
        return hash(self.to_tuple(Point5D.LABELS))

    def contains(self, other: "Interval5D") -> bool:
        return self.start <= other.start and self.stop >= other.stop

    def to_dict(self) -> Dict[str, INTERVAL]:
        return {k: self[k] for k in Point5D.LABELS}

    @classmethod
    def make_intervals(cls, start: Point5D, stop: Point5D) -> Dict[str, INTERVAL]:
        return {k: (start[k], stop[k]) for k in Point5D.LABELS}

    @staticmethod
    def create_from_start_stop(start: Point5D, stop: Point5D) -> "Interval5D":
        return Interval5D(**Interval5D.make_intervals(start, stop))

    @staticmethod
    def from_json_value(data: JsonValue) -> "Interval5D":
        data_dict = ensureJsonObject(data)
        start = Point5D.from_json_value(data_dict["start"])
        stop = Point5D.from_json_value(data_dict["stop"])
        return Interval5D.create_from_start_stop(start, stop)

    def to_json_value(self) -> JsonObject:
        return {"start": self.start.to_json_value(), "stop": self.stop.to_json_value()}

    def from_start_stop(self: INTERVAL_5D, start: Point5D, stop: Point5D) -> INTERVAL_5D:
        slices = self.make_intervals(start, stop)
        return self.updated(**slices)

    def _ranges(self, block_shape: Shape5D) -> Iterator[List[int]]:
        starts = self.start.to_np(Point5D.LABELS)
        ends = self.stop.to_np(Point5D.LABELS)
        steps = block_shape.to_np(Point5D.LABELS)

        for start, end, step in zip(starts, ends, steps): #type: ignore
            yield list(np.arange(start, end, step))

    def split(self: INTERVAL_5D, block_shape: Shape5D) -> Iterator[INTERVAL_5D]:
        """Splits self into multiple Interval5D instances, starting from self.start. Every piece shall have
        shape == block_shape excedpt for the last one, which will be clamped to self.stop"""
        for begin_tuple in itertools.product(*self._ranges(block_shape)):
            start = Point5D.from_tuple(begin_tuple, Point5D.LABELS)
            stop = (start + block_shape).clamped(maximum=self.stop)
            yield self.from_start_stop(start, stop)

    def enlarge_to_tiles(self: INTERVAL_5D, tile_shape: Shape5D, tiles_origin: Point5D) -> INTERVAL_5D:
        """Enlarges self until it becomes a multiple of tile_shape."""
        aligned = self.translated(-tiles_origin)
        start = Point5D(**{k: floor(aligned.start[k] / tile_shape[k]) * tile_shape[k] for k in Point5D.LABELS})
        stop = Point5D(**{k: ceil(aligned.stop[k] / tile_shape[k]) * tile_shape[k] for k in Point5D.LABELS})
        return self.from_start_stop(start, stop).translated(tiles_origin)

    def is_tile(self, *, tile_shape: Shape5D, full_interval: "Interval5D", clamped: bool) -> bool:
        """Checks if self is a tile of full_interval.

        Tiles start at a multiple of tile_shape, as counted from full_interval.start.
        Each coordinate of self.stop must be either:
            - a multiple of tile_shape, as counted from full_interval or
            - the coresponding value in full_interval.stop if clamped == True
        """
        enlarged = self.enlarge_to_tiles(tile_shape=tile_shape, tiles_origin=full_interval.start)
        if enlarged.shape > tile_shape or self.shape.hypervolume == 0:
            return False
        if self.start != enlarged.start:
            return False
        if not clamped:
            return self.stop == enlarged.stop
        else:
            return all(self.stop[k] in (enlarged.stop[k], full_interval.stop[k]) for k in Point5D.LABELS)

    def get_tiles(self: INTERVAL_5D, tile_shape: Shape5D, tiles_origin: Point5D) -> Iterator[INTERVAL_5D]:
        """Gets all tiles that would cover the entirety of self."""
        yield from self.enlarge_to_tiles(tile_shape=tile_shape, tiles_origin=tiles_origin).split(tile_shape)

    def __getitem__(self, key: str) -> INTERVAL:
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
    def updated(
        self: INTERVAL_5D,
        *,
        t: Optional[SPAN] = None,
        c: Optional[SPAN] = None,
        x: Optional[SPAN] = None,
        y: Optional[SPAN] = None,
        z: Optional[SPAN] = None,
    ) -> INTERVAL_5D:
        return self.__class__(
            t=self.t if t is None else t,
            c=self.c if c is None else c,
            x=self.x if x is None else x,
            y=self.y if y is None else y,
            z=self.z if z is None else z,
        )

    @property
    def shape(self) -> Shape5D:
        return Shape5D(**(self.stop - self.start).to_dict())

    def clamped(
        self: INTERVAL_5D,
        limits: Union[Shape5D, "Interval5D", None] = None,
        *,
        x: Optional[SPAN] = None,
        y: Optional[SPAN] = None,
        z: Optional[SPAN] = None,
        t: Optional[SPAN] = None,
        c: Optional[SPAN] = None,
    ) -> INTERVAL_5D:
        limits = limits or self
        limits_interval = limits if isinstance(limits, Interval5D) else limits.to_interval5d()
        updated_limits = limits_interval.updated(x=x, y=y, z=z, t=t, c=c)
        return self.from_start_stop(
            self.start.clamped(updated_limits.start, updated_limits.stop),
            self.stop.clamped(updated_limits.start, updated_limits.stop),
        )

    def enlarged(self: INTERVAL_5D, radius: Point5D) -> INTERVAL_5D:
        return self.from_start_stop(self.start - radius, self.stop + radius)

    def translated(self: INTERVAL_5D, offset: Point5D) -> INTERVAL_5D:
        return self.from_start_stop(self.start + offset, self.stop + offset)

    def to_slices(self, axis_order: str = Point5D.LABELS) -> Tuple[slice, ...]:
        return tuple(slice(self[k][0], self[k][1]) for k in axis_order)

    def to_tuple(self, axis_order: str) -> Tuple[INTERVAL, ...]:
        return tuple(self[k] for k in axis_order)

    def to_start_stop_tuple(self, axis_order: str) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        return (self.start.to_tuple(axis_order), self.stop.to_tuple(axis_order))

    def to_ilastik_cutout_subregion(self, axis_order: str) -> str:
        return str(list(self.to_start_stop_tuple(axis_order=axis_order)))

    def __repr__(self) -> str:
        reprs: List[str] = []
        for k, span in self.to_dict().items():
            if span[1] - span[0] == 1:
                if span[0] != 0:
                    reprs.append(f"{k}:{span[0]}")
            else:
                reprs.append(f"{k}:{span[0]}_{span[1]}")
        spans = ", ".join(reprs)
        return self.__class__.__name__ + f"({spans})"

    def get_borders(self: INTERVAL_5D, thickness: Shape5D) -> Iterable[INTERVAL_5D]:
        """Returns subslices of self, such that these subslices are at the borders
        of self (i.e.: touching the start or end of self)

        No axis of thickness should exceed self.shape[axis], since the subslices must be contained in self
        Axis where thickness[axis] == 0 will produce no borders:
            slc.get_borders(Interval5D.zero(x=1, y=1)) will produce 4 borders (left, right, top, bottom)
        If, for any axis, thickness[axis] == self.shape[axis], then there will be duplicated borders in the output
        """
        thickness_interval = thickness.to_interval5d(offset=self.start)
        if not self.contains(thickness_interval):
            raise ValueError(f"Bad thickness {thickness} for interval {self}")
        # FIXME: I haven't ported this yet!!!!!
        for axis, axis_thickness in thickness.to_dict().items():
            if axis_thickness == 0:
                continue
            span = self[axis]
            yield self.updated(**{axis: (span[0], span[0] + axis_thickness)})
            yield self.updated(**{axis: (span[1] - axis_thickness, span[1])})

    def get_neighboring_tiles(self: INTERVAL_5D, tile_shape: Shape5D) -> Iterator[INTERVAL_5D]:
        for axis in Point5D.LABELS:
            for axis_offset in (tile_shape[axis], -tile_shape[axis]):
                offset = Point5D.zero(**{axis: axis_offset})
                yield self.translated(offset)

    def get_neighbor_tile_adjacent_to(
        self: INTERVAL_5D, *, anchor: "Interval5D", tile_shape: Shape5D
    ) -> Optional[INTERVAL_5D]:
        if not self.contains(anchor):
            raise ValueError(f"Anchor {anchor} is not contained within {self}")

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

        if anchor[direction_axis][1] == self[direction_axis][1]:
            if self.shape != tile_shape:  # Getting a further tile from a partial tile
                return None
            return self.translated(offset)
        if anchor[direction_axis][0] == self[direction_axis][0]:
            if self.start - offset < Point5D.zero():  # no negative neighbors
                return None
            return self.translated(-offset)

        raise ValueError(f"Bad anchor for slice {self}: {anchor}")

    @staticmethod
    def enclosing(points: Iterable[Union[Point5D, "Interval5D"]]) -> "Interval5D":
        all_points: List[Point5D] = []
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
