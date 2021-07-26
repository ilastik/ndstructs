from ndstructs import Point5D, KeyMap
import pytest
import json


inf = Point5D.one() * 999999
ninf = Point5D.one() * -999999


def test_labeled_coords_constructor_property_assignment():
    p = Point5D(x=1, y=2, z=3, t=4, c=5)
    assert p.x == 1
    assert p.y == 2
    assert p.z == 3
    assert p.t == 4
    assert p.c == 5


def test_zero_factory_method_defaults_coords_to_zero():
    p = Point5D.zero(x=123, c=456)
    assert p.x == 123
    assert p.y == 0
    assert p.z == 0
    assert p.t == 0
    assert p.c == 456


def test_one_factory_method_defaults_coords_to_one():
    p = Point5D.one(x=123, c=456)
    assert p.x == 123
    assert p.y == 1
    assert p.z == 1
    assert p.t == 1
    assert p.c == 456


def test_to_tuple_respects_given_axis_order():
    p = Point5D(x=1, y=2, z=3, t=4, c=5)
    assert p.to_tuple("xyz") == (1, 2, 3) == (p.x, p.y, p.z)
    assert p.to_tuple("x") == (1,) == (p.x,)
    assert p.to_tuple("yzxct") == (2, 3, 1, 5, 4) == (p.y, p.z, p.x, p.c, p.t)
    assert p.to_tuple("txcyz") == (4, 1, 5, 2, 3) == (p.t, p.x, p.c, p.y, p.z)


def test_to_dict_consistent():
    p = Point5D(x=1, y=2, z=3, t=4, c=5)
    assert p.to_dict() == {"x": 1, "y": 2, "z": 3, "t": 4, "c": 5}


def test_updated_modifies_coords_and_keeps_original_intact():
    p = Point5D(x=1, y=2, z=3, t=4, c=5)
    assert p.updated(z=1000).z == 1000
    assert p.updated(x=99, y=88, c=77).to_tuple("xyctz") == (99, 88, 77, 4, 3)
    assert p.to_tuple("xyztc") == (1, 2, 3, 4, 5)


def test_clamped_keeps_values_within_limits():
    p = Point5D(x=100, y=200, z=300, t=400, c=500)
    assert p.clamped(maximum=inf.updated(y=50, c=600)).to_tuple("yc") == (50, 500)
    assert p.clamped(minimum=ninf.updated(y=300, x=90)).to_tuple("yx") == (300, 100)

    min_pt = Point5D(x=10, y=20, z=30, t=40, c=1000)
    assert p.clamped(minimum=min_pt).to_tuple("xyztc") == (100, 200, 300, 400, 1000)

    max_pt = Point5D(x=1, y=2, z=3, t=4, c=1000)
    assert p.clamped(maximum=max_pt).to_tuple("xyztc") == (1, 2, 3, 4, 500)

    clamped_pt = p.clamped(minimum=ninf.updated(x=20, t=50), maximum=inf.updated(x=120, t=500))
    assert clamped_pt.to_tuple("xt") == (100, 400)


def test_point_equality():
    assert Point5D.zero() == Point5D(x=0, y=0, z=0, t=0, c=0)
    assert Point5D(x=1, y=2, z=3, t=4, c=5) == Point5D(x=1, y=2, z=3, t=4, c=5)
    assert Point5D.zero() != Point5D(x=1, y=0, z=0, t=0, c=0)


def test_point_arithmetic():
    p = Point5D(x=100, y=200, z=300, t=400, c=500)
    assert p + Point5D.zero(x=100) == Point5D(x=200, y=200, z=300, t=400, c=500)
    assert p + Point5D(x=1, y=2, z=3, t=4, c=5) == Point5D(x=101, y=202, z=303, t=404, c=505)

    other = Point5D(x=1, y=2, z=3, t=4, c=5)
    for op in ("__add__", "__sub__", "__mul__", "__floordiv__"):
        p_as_np = p.to_np(Point5D.LABELS)
        np_result = getattr(p_as_np, op)(other.to_np(Point5D.LABELS))
        assert all(getattr(p, op)(other).to_np(Point5D.LABELS) == np_result)


def test_point_relabeling_swap():
    p = Point5D(x=100, y=200, z=300, t=400, c=500)
    keymap = KeyMap(x="y", y="x")
    assert p.relabeled(keymap) == Point5D(y=100, x=200, z=300, t=400, c=500)


def test_point_relabeling_shift():
    p = Point5D(x=100, y=200, z=300, t=400, c=500)
    keymap = KeyMap(x="y", y="z", z="x")
    assert p.relabeled(keymap) == Point5D(y=100, z=200, x=300, t=400, c=500)


def test_point_relabeling_bad_map():
    with pytest.raises(AssertionError):
        keymap = KeyMap(x="z")
    with pytest.raises(AssertionError):
        keymap = KeyMap(x="z")


def test_point_interpolation():
    start = Point5D.zero()
    end = Point5D(x=3, y=7)
    print("")
    print(list(start.interpolate_until(endpoint=end)))

def test_json_serialization():
    point = Point5D(x=1, y=2, z= 3, t=4, c=5)
    assert Point5D.from_json_value(json.loads(json.dumps(point.to_json_value()))) == point
