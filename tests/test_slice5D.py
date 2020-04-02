from ndstructs import Point5D, Shape5D, Slice5D, KeyMap
import numpy


def test_all_constructor():
    slc = Slice5D.all(x=3, z=slice(10, 20))
    assert slc.to_slices("xyztc") == (slice(3, 4), slice(None), slice(10, 20), slice(None), slice(None))


def test_from_start_stop():
    start = Point5D(x=10, y=20, z=30, t=40, c=50)
    stop = start + 10
    slc = Slice5D.create_from_start_stop(start, stop)
    assert slc == Slice5D(x=slice(10, 20), y=slice(20, 30), z=slice(30, 40), t=slice(40, 50), c=slice(50, 60))


def test_slice_translation():
    slc = Slice5D(x=slice(10, 100), y=slice(20, 200))
    translated_slc = slc.translated(Point5D(x=1, y=2, z=3, t=4, c=5))
    assert translated_slc == Slice5D(x=slice(11, 101), y=slice(22, 202))

    slc = Slice5D(x=slice(10, 100), y=slice(20, 200), z=0, t=0, c=0)
    translated_slc = slc.translated(Point5D(x=-1, y=-2, z=-3, t=-4, c=-5000))
    assert translated_slc == Slice5D(
        x=slice(9, 99), y=slice(18, 198), z=slice(-3, -2), t=slice(-4, -3), c=slice(-5000, -4999)
    )


def test_slice_enlarge():
    slc = Slice5D(x=slice(10, 100), y=slice(20, 200))
    enlarged = slc.enlarged(radius=Point5D(x=1, y=2, z=3, t=4, c=5))
    assert enlarged == Slice5D(x=slice(9, 101), y=slice(18, 202))

    slc = Slice5D(x=slice(10, 100), y=slice(20, 200), z=0, t=0, c=0)
    enlarged = slc.enlarged(radius=Point5D(x=1, y=2, z=3, t=4, c=5))
    assert enlarged == Slice5D(x=slice(9, 101), y=slice(18, 202), z=slice(-3, 4), t=slice(-4, 5), c=slice(-5, 6))


def test_slice_contains_smaller_slice():
    outer_slice = Slice5D(x=slice(10, 100), y=slice(20, 200))
    inner_slice = Slice5D(x=slice(20, 50), y=slice(30, 40), z=0, t=0, c=0)
    assert outer_slice.contains(inner_slice)


def test_slice_does_not_contain_translated_slice():
    slc = Slice5D(x=slice(10, 100), y=slice(20, 200), z=0, t=0, c=0)
    translated_slc = slc.translated(Point5D.zero(x=10))
    assert not slc.contains(translated_slc)


def test_slice_clamp():
    outer = Slice5D(x=slice(10, 100), y=slice(20, 200))
    inner = Slice5D(x=slice(20, 50), y=slice(30, 40), z=0, t=0, c=0)
    assert outer.clamped(inner) == inner
    assert inner.clamped(outer) == inner

    intersecting_outer = Slice5D(x=slice(50, 200), y=slice(30, 900))
    assert intersecting_outer.clamped(outer) == Slice5D(x=slice(50, 100), y=slice(30, 200))

    intersecting_outer = Slice5D(x=slice(-100, 50), y=slice(10, 100))
    assert intersecting_outer.clamped(outer) == Slice5D(x=slice(10, 50), y=slice(20, 100))

    outside_outer = Slice5D(x=slice(200, 300), y=slice(400, 500))
    assert outside_outer.clamped(outer).defined_with(Shape5D()).shape.volume == 0


def test_slice_defined_with():
    slc = Slice5D(x=slice(10, 20))

    assert slc.defined_with(Shape5D(x=100, y=15, z=17)) == Slice5D.zero(x=slice(10, 20), y=slice(0, 15), z=slice(0, 17))

    assert slc.defined_with(Slice5D.zero(x=slice(1, 3), y=slice(10, 20))) == Slice5D.zero(
        x=slice(10, 20), y=slice(10, 20)
    )


def test_to_slices():
    slc = Slice5D(x=1, y=2, z=slice(10, 20))
    assert slc.to_slices("xyztc") == (slice(1, 2), slice(2, 3), slice(10, 20), slice(None), slice(None))
    assert slc.to_slices("ytzcx") == (slice(2, 3), slice(None), slice(10, 20), slice(None), slice(1, 2))


def test_with_coord():
    slc = Slice5D(x=0, y=1, z=2, t=3, c=4)
    assert slc.with_coord(z=slice(10, 20)).to_slices("xyztc") == (
        slice(0, 1),
        slice(1, 2),
        slice(10, 20),
        slice(3, 4),
        slice(4, 5),
    )
    assert slc.with_coord(x=123).to_slices("xyztc") == (
        slice(123, 124),
        slice(1, 2),
        slice(2, 3),
        slice(3, 4),
        slice(4, 5),
    )


def test_split_when_slice_is_multiple_of_block_shape():
    slc = Slice5D.zero(x=slice(100, 200), y=slice(200, 300))
    pieces = list(slc.split(Shape5D(x=50, y=50)))
    assert Slice5D.zero(x=slice(100, 150), y=slice(200, 250)) in pieces
    assert Slice5D.zero(x=slice(100, 150), y=slice(250, 300)) in pieces
    assert Slice5D.zero(x=slice(150, 200), y=slice(200, 250)) in pieces
    assert Slice5D.zero(x=slice(150, 200), y=slice(250, 300)) in pieces
    assert len(pieces) == 4


def test_split_when_slice_is_NOT_multiple_of_block_shape():
    slc = Slice5D.zero(x=slice(100, 210), y=slice(200, 320))
    pieces = list(slc.split(Shape5D(x=50, y=50)))
    assert Slice5D.zero(x=slice(100, 150), y=slice(200, 250)) in pieces
    assert Slice5D.zero(x=slice(100, 150), y=slice(250, 300)) in pieces
    assert Slice5D.zero(x=slice(100, 150), y=slice(300, 320)) in pieces

    assert Slice5D.zero(x=slice(150, 200), y=slice(200, 250)) in pieces
    assert Slice5D.zero(x=slice(150, 200), y=slice(250, 300)) in pieces
    assert Slice5D.zero(x=slice(150, 200), y=slice(300, 320)) in pieces

    assert Slice5D.zero(x=slice(200, 210), y=slice(200, 250)) in pieces
    assert Slice5D.zero(x=slice(200, 210), y=slice(250, 300)) in pieces
    assert Slice5D.zero(x=slice(200, 210), y=slice(300, 320)) in pieces
    assert len(pieces) == 9


def test_get_tiles_when_slice_is_multiple_of_tile():
    slc = Slice5D.zero(x=slice(100, 200), y=slice(200, 300))
    tiles = list(slc.get_tiles(Shape5D(x=50, y=50)))
    assert Slice5D.zero(x=slice(100, 150), y=slice(200, 250)) in tiles
    assert Slice5D.zero(x=slice(100, 150), y=slice(250, 300)) in tiles
    assert Slice5D.zero(x=slice(150, 200), y=slice(200, 250)) in tiles
    assert Slice5D.zero(x=slice(150, 200), y=slice(250, 300)) in tiles
    assert len(tiles) == 4


def test_get_tiles_when_slice_is_NOT_multiple_of_tile():
    slc = Slice5D.zero(x=slice(90, 210), y=slice(200, 320), z=slice(10, 20))
    pieces = list(slc.get_tiles(Shape5D(x=50, y=50, z=10)))

    assert Slice5D.zero(x=slice(50, 100), y=slice(200, 250), z=slice(10, 20)) in pieces
    assert Slice5D.zero(x=slice(50, 100), y=slice(250, 300), z=slice(10, 20)) in pieces
    assert Slice5D.zero(x=slice(50, 100), y=slice(300, 350), z=slice(10, 20)) in pieces

    assert Slice5D.zero(x=slice(100, 150), y=slice(200, 250), z=slice(10, 20)) in pieces
    assert Slice5D.zero(x=slice(100, 150), y=slice(250, 300), z=slice(10, 20)) in pieces
    assert Slice5D.zero(x=slice(100, 150), y=slice(300, 350), z=slice(10, 20)) in pieces

    assert Slice5D.zero(x=slice(150, 200), y=slice(200, 250), z=slice(10, 20)) in pieces
    assert Slice5D.zero(x=slice(150, 200), y=slice(250, 300), z=slice(10, 20)) in pieces
    assert Slice5D.zero(x=slice(150, 200), y=slice(300, 350), z=slice(10, 20)) in pieces

    assert Slice5D.zero(x=slice(200, 250), y=slice(200, 250), z=slice(10, 20)) in pieces
    assert Slice5D.zero(x=slice(200, 250), y=slice(250, 300), z=slice(10, 20)) in pieces
    assert Slice5D.zero(x=slice(200, 250), y=slice(300, 350), z=slice(10, 20)) in pieces
    assert len(pieces) == 12


def test_get_borders():
    slc = Slice5D.zero(x=slice(100, 200), y=slice(300, 400), c=slice(0, 4))
    thickness = Shape5D.zero(x=1, y=1)
    expected_borders = {
        slc.with_coord(x=slice(100, 101)),
        slc.with_coord(y=slice(300, 301)),
        slc.with_coord(x=slice(199, 200)),
        slc.with_coord(y=slice(399, 400)),
    }
    for border_slc in slc.get_borders(thickness):
        expected_borders.remove(border_slc)
    assert len(expected_borders) == 0

    thickness = Shape5D.zero(x=10, y=20)
    expected_thick_borders = {
        slc.with_coord(x=slice(100, 110)),
        slc.with_coord(x=slice(190, 200)),
        slc.with_coord(y=slice(300, 320)),
        slc.with_coord(y=slice(380, 400)),
    }
    for border_slc in slc.get_borders(thickness=thickness):
        expected_thick_borders.remove(border_slc)
    assert len(expected_thick_borders) == 0

    z2_slc = Slice5D.zero(x=slice(100, 200), y=slice(300, 400), z=slice(8, 10))
    thickness = Shape5D.zero(x=10, z=2)
    expected_z2_borders = {
        z2_slc.with_coord(x=slice(100, 110)),
        z2_slc.with_coord(x=slice(190, 200)),
        z2_slc.with_coord(z=slice(8, 10)),
    }
    for border_slc in z2_slc.get_borders(thickness=thickness):
        expected_z2_borders.remove(border_slc)
    assert len(expected_z2_borders) == 0


def test_slice_relabeling_swap():
    slc = Slice5D(x=100, y=200, z=300, t=400, c=500)
    keymap = KeyMap(x="y", y="x")
    assert slc.relabeled(keymap) == Slice5D(y=100, x=200, z=300, t=400, c=500)


def test_slice_relabeling_shift():
    slc = Slice5D(x=100, y=200, z=300, t=400, c=500)
    keymap = KeyMap(x="y", y="z", z="x")
    assert slc.relabeled(keymap) == Slice5D(y=100, z=200, x=300, t=400, c=500)


def test_slice_enclosing():
    p1 = Point5D.zero(x=-13, y=40)
    p2 = Point5D.zero(z=-1, c=6)
    p3 = Point5D.zero(t=3, x=4)
    p4 = Point5D.zero(t=100, y=400)

    expected_slice = Slice5D(x=slice(-13, 4 + 1), y=slice(40, 400 + 1), z=slice(-1, -1 + 1), c=slice(6, 6 + 1))
    assert Slice5D.enclosing([p1, p2, p3, p4])
