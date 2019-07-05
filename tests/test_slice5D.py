from ndstructs import Point5D, Shape5D, Slice5D
import numpy


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
