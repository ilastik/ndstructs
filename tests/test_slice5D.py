from ndstructs import Point5D, Shape5D, Interval5D, KeyMap
import random
import json


def test_from_start_stop():
    start = Point5D(x=10, y=20, z=30, t=40, c=50)
    stop = start + 10
    slc = Interval5D.create_from_start_stop(start, stop)
    assert slc == Interval5D(x=(10, 20), y=(20, 30), z=(30, 40), t=(40, 50), c=(50, 60))


def test_slice_translation():
    slc = Interval5D.zero(x=(10, 100), y=(20, 200))
    translated_slc = slc.translated(Point5D(x=1, y=2, z=3, t=4, c=5))
    assert translated_slc == Interval5D(x=(11, 101), y=(22, 202), z=(3, 4), t=(4, 5), c=(5, 6))

    slc = Interval5D(x=(10, 100), y=(20, 200), z=0, t=0, c=0)
    translated_slc = slc.translated(Point5D(x=-1, y=-2, z=-3, t=-4, c=-5000))
    assert translated_slc == Interval5D(x=(9, 99), y=(18, 198), z=(-3, -2), t=(-4, -3), c=(-5000, -4999))


def test_slice_enlarge():
    slc = Interval5D.zero(x=(10, 100), y=(20, 200))
    enlarged = slc.enlarged(radius=Point5D(x=1, y=2, z=3, t=4, c=5))
    assert enlarged == Interval5D(x=(9, 101), y=(18, 202), z=(-3, 4), t=(-4, 5), c=(-5, 6))

    slc2 = Interval5D(x=(10, 100), y=(20, 200), z=0, t=0, c=0)
    enlarged2 = slc2.enlarged(radius=Point5D(x=1, y=2, z=3, t=4, c=5))
    assert enlarged2 == Interval5D(x=(9, 101), y=(18, 202), z=(-3, 4), t=(-4, 5), c=(-5, 6))


def test_slice_contains_smaller_slice():
    outer_slice = Interval5D.zero(x=(10, 100), y=(20, 200))
    inner_slice = Interval5D(x=(20, 50), y=(30, 40), z=0, t=0, c=0)
    assert outer_slice.contains(inner_slice)


def test_slice_does_not_contain_translated_slice():
    slc = Interval5D(x=(10, 100), y=(20, 200), z=0, t=0, c=0)
    translated_slc = slc.translated(Point5D.zero(x=10))
    assert not slc.contains(translated_slc)


def test_slice_clamp():
    outer = Interval5D.zero(x=(10, 100), y=(20, 200))
    inner = Interval5D.zero(x=(20, 50), y=(30, 40))
    assert outer.clamped(inner) == inner
    assert inner.clamped(outer) == inner

    intersecting_outer = Interval5D.zero(x=(50, 200), y=(30, 900))
    assert intersecting_outer.clamped(outer) == Interval5D.zero(x=(50, 100), y=(30, 200))

    intersecting_outer = Interval5D.zero(x=(-100, 50), y=(10, 100))
    assert intersecting_outer.clamped(outer) == Interval5D.zero(x=(10, 50), y=(20, 100))

    outside_outer = Interval5D.zero(x=(200, 300), y=(400, 500))
    a = outside_outer.clamped(outer)
    assert a.shape.volume == 0


def test_to_slices():
    slc = Interval5D.zero(x=1, y=2, z=(10, 20))
    assert slc.to_slices("xyztc") == (slice(1, 2), slice(2, 3), slice(10, 20), slice(0, 1), slice(0, 1))
    assert slc.to_slices("ytzcx") == (slice(2, 3), slice(0, 1), slice(10, 20), slice(0, 1), slice(1, 2))


def test_with_coord():
    slc = Interval5D(x=0, y=1, z=2, t=3, c=4)
    assert slc.updated(z=(10, 20)).to_slices("xyztc") == (
        slice(0, 1),
        slice(1, 2),
        slice(10, 20),
        slice(3, 4),
        slice(4, 5),
    )
    assert slc.updated(x=123).to_slices("xyztc") == (
        slice(123, 124),
        slice(1, 2),
        slice(2, 3),
        slice(3, 4),
        slice(4, 5),
    )


def test_split_when_slice_is_multiple_of_block_shape():
    slc = Interval5D.zero(x=(100, 200), y=(200, 300))
    pieces = list(slc.split(Shape5D(x=50, y=50)))
    assert Interval5D.zero(x=(100, 150), y=(200, 250)) in pieces
    assert Interval5D.zero(x=(100, 150), y=(250, 300)) in pieces
    assert Interval5D.zero(x=(150, 200), y=(200, 250)) in pieces
    assert Interval5D.zero(x=(150, 200), y=(250, 300)) in pieces
    assert len(pieces) == 4


def test_split_when_slice_is_NOT_multiple_of_block_shape():
    slc = Interval5D.zero(x=(100, 210), y=(200, 320))
    pieces = list(slc.split(Shape5D(x=50, y=50)))
    assert Interval5D.zero(x=(100, 150), y=(200, 250)) in pieces
    assert Interval5D.zero(x=(100, 150), y=(250, 300)) in pieces
    assert Interval5D.zero(x=(100, 150), y=(300, 320)) in pieces

    assert Interval5D.zero(x=(150, 200), y=(200, 250)) in pieces
    assert Interval5D.zero(x=(150, 200), y=(250, 300)) in pieces
    assert Interval5D.zero(x=(150, 200), y=(300, 320)) in pieces

    assert Interval5D.zero(x=(200, 210), y=(200, 250)) in pieces
    assert Interval5D.zero(x=(200, 210), y=(250, 300)) in pieces
    assert Interval5D.zero(x=(200, 210), y=(300, 320)) in pieces
    assert len(pieces) == 9


def test_interval_short_on_the_right_side_expands_to_tiles():
    enlarged = Interval5D.zero(
        y=(100, 200 - 7),
        z=(200, 300 - 7)
    ).enlarge_to_tiles(
        tile_shape=Shape5D(y=100, z=100),
        tiles_origin=Point5D.zero()
    )
    assert enlarged == Interval5D.zero(y=(100, 200), z=(200, 300))


def test_interval_slightly_big_on_the_left_expands_to_tiles():
    enlarged = Interval5D.zero(
        y=(0 + 99, 150),
        z=(200, 230)
    ).enlarge_to_tiles(
        tile_shape=Shape5D(y=100, z=100),
        tiles_origin=Point5D.zero()
    )
    assert enlarged == Interval5D.zero(y=(0, 200), z=(200, 300))


def test_misaligned_interval_expands_to_tiles_with_non_zero_origin():
    enlarged = Interval5D.zero(
        y=(117, 217), # aligned to origin.y = 17
        z=(200, 230), # aligned to origin.z = 0
    ).enlarge_to_tiles(
        tiles_origin=Point5D.zero(y=17, z=0),
        tile_shape=Shape5D(y=100, z=100)
    )
    assert enlarged == Interval5D.zero(y=(117, 217), z=(200, 300))


def test_small_interval_expands_to_misaligned_tiles():
    enlarged = Interval5D.zero(
        y=(117 + 10, 217 - 10),
        z=(200, 230)
    ).enlarge_to_tiles(
        tiles_origin=Point5D.zero(y=17, z=0),
        tile_shape=Shape5D(y=100, z=100)
    )
    assert enlarged == Interval5D.zero(
        y=(117, 217),
        z=(200, 300)
    )


def test_get_tiles_when_slice_is_multiple_of_tile():
    slc = Interval5D.zero(x=(100, 200), y=(200, 300))
    tiles = list(slc.get_tiles(tile_shape=Shape5D(x=50, y=50), tiles_origin=Point5D.zero()))
    assert Interval5D.zero(x=(100, 150), y=(200, 250)) in tiles
    assert Interval5D.zero(x=(100, 150), y=(250, 300)) in tiles
    assert Interval5D.zero(x=(150, 200), y=(200, 250)) in tiles
    assert Interval5D.zero(x=(150, 200), y=(250, 300)) in tiles
    assert len(tiles) == 4


def test_get_tiles_when_slice_is_NOT_multiple_of_tile():
    slc = Interval5D.zero(x=(90, 210), y=(200, 320), z=(10, 20))
    pieces = list(slc.get_tiles(tile_shape=Shape5D(x=50, y=50, z=10), tiles_origin=Point5D.zero()))

    assert Interval5D.zero(x=(50, 100), y=(200, 250), z=(10, 20)) in pieces
    assert Interval5D.zero(x=(50, 100), y=(250, 300), z=(10, 20)) in pieces
    assert Interval5D.zero(x=(50, 100), y=(300, 350), z=(10, 20)) in pieces

    assert Interval5D.zero(x=(100, 150), y=(200, 250), z=(10, 20)) in pieces
    assert Interval5D.zero(x=(100, 150), y=(250, 300), z=(10, 20)) in pieces
    assert Interval5D.zero(x=(100, 150), y=(300, 350), z=(10, 20)) in pieces

    assert Interval5D.zero(x=(150, 200), y=(200, 250), z=(10, 20)) in pieces
    assert Interval5D.zero(x=(150, 200), y=(250, 300), z=(10, 20)) in pieces
    assert Interval5D.zero(x=(150, 200), y=(300, 350), z=(10, 20)) in pieces

    assert Interval5D.zero(x=(200, 250), y=(200, 250), z=(10, 20)) in pieces
    assert Interval5D.zero(x=(200, 250), y=(250, 300), z=(10, 20)) in pieces
    assert Interval5D.zero(x=(200, 250), y=(300, 350), z=(10, 20)) in pieces
    assert len(pieces) == 12


def test_get_tiles_with_offset_origin():
    tiles = list(Interval5D.zero(
        x=(0 + 7, 30 + 7), y=(100 + 3, 130 + 3) #multiple of 10-sided tiles, offset by 3
    ).get_tiles(
        tiles_origin=Point5D.zero(x=7, y=3),
        tile_shape=Shape5D(x=10, y=10),
    ))

    assert len(tiles) == 9

    assert Interval5D.zero(x=(7, 17), y=(103, 113)) in tiles
    assert Interval5D.zero(x=(17, 27), y=(103, 113)) in tiles
    assert Interval5D.zero(x=(27, 37), y=(103, 113)) in tiles

    assert Interval5D.zero(x=(7, 17), y=(113, 123)) in tiles
    assert Interval5D.zero(x=(17, 27), y=(113, 123)) in tiles
    assert Interval5D.zero(x=(27, 37), y=(113, 123)) in tiles

    assert Interval5D.zero(x=(7, 17), y=(123, 133)) in tiles
    assert Interval5D.zero(x=(17, 27), y=(123, 133)) in tiles
    assert Interval5D.zero(x=(27, 37), y=(123, 133)) in tiles


def test_get_tiles_with_offset_origin_and_clamped_interval():
    tiles = list(Interval5D.zero(
        x=(0 + 7, 30 + 7), y=(100 + 3, 130 + 0) # y falls short of being a mutliple of 10-sided tile
    ).get_tiles(
        tiles_origin=Point5D.zero(x=7, y=3),
        tile_shape=Shape5D(x=10, y=10),
    ))

    assert len(tiles) == 9

    assert Interval5D.zero(x=(7, 17), y=(103, 113)) in tiles
    assert Interval5D.zero(x=(17, 27), y=(103, 113)) in tiles
    assert Interval5D.zero(x=(27, 37), y=(103, 113)) in tiles

    assert Interval5D.zero(x=(7, 17), y=(113, 123)) in tiles
    assert Interval5D.zero(x=(17, 27), y=(113, 123)) in tiles
    assert Interval5D.zero(x=(27, 37), y=(113, 123)) in tiles

    assert Interval5D.zero(x=(7, 17), y=(123, 133)) in tiles
    assert Interval5D.zero(x=(17, 27), y=(123, 133)) in tiles
    assert Interval5D.zero(x=(27, 37), y=(123, 133)) in tiles


def test_get_borders():
    slc = Interval5D.zero(x=(100, 200), y=(300, 400), c=(0, 4))
    thickness = Shape5D.zero(x=1, y=1)
    expected_borders = {
        slc.updated(x=(100, 101)),
        slc.updated(y=(300, 301)),
        slc.updated(x=(199, 200)),
        slc.updated(y=(399, 400)),
    }
    assert expected_borders == set(slc.get_borders(thickness))
    assert len(list(slc.get_borders(thickness))) == 4

    thickness = Shape5D.zero(x=10, y=20)
    expected_thick_borders = {
        slc.updated(x=(100, 110)),
        slc.updated(x=(190, 200)),
        slc.updated(y=(300, 320)),
        slc.updated(y=(380, 400)),
    }
    assert expected_thick_borders == set(slc.get_borders(thickness=thickness))
    assert len(list(slc.get_borders(thickness=thickness))) == 4

    z2_slc = Interval5D.zero(x=(100, 200), y=(300, 400), z=(8, 10))
    thickness = Shape5D.zero(x=10, z=2)
    expected_z2_borders = {z2_slc.updated(x=(100, 110)), z2_slc.updated(x=(190, 200)), z2_slc.updated(z=(8, 10))}
    assert expected_z2_borders == set(z2_slc.get_borders(thickness=thickness))
    assert len(list(z2_slc.get_borders(thickness=thickness))) == 4


def test_get_neighbor_tile_adjacent_to():
    source_tile = Interval5D(x=(100, 200), y=(300, 400), c=(0, 3), z=1, t=1)

    right_border = source_tile.updated(x=(199, 200))
    right_neighbor = source_tile.get_neighbor_tile_adjacent_to(anchor=right_border, tile_shape=source_tile.shape)
    assert right_neighbor == source_tile.updated(x=(200, 300))

    left_border = source_tile.updated(x=(100, 101))
    left_neighbor = source_tile.get_neighbor_tile_adjacent_to(anchor=left_border, tile_shape=source_tile.shape)
    assert left_neighbor == source_tile.updated(x=(0, 100))

    top_border = source_tile.updated(y=(399, 400))
    top_neighbor = source_tile.get_neighbor_tile_adjacent_to(anchor=top_border, tile_shape=source_tile.shape)
    assert top_neighbor == source_tile.updated(y=(400, 500))

    bottom_border = source_tile.updated(y=(300, 301))
    bottom_neighbor = source_tile.get_neighbor_tile_adjacent_to(anchor=bottom_border, tile_shape=source_tile.shape)
    assert bottom_neighbor == source_tile.updated(y=(200, 300))

    partial_tile = Interval5D(x=(100, 200), y=(400, 470), c=(0, 3), z=1, t=1)

    right_border = partial_tile.updated(x=(199, 200))
    assert partial_tile.get_neighbor_tile_adjacent_to(anchor=right_border, tile_shape=source_tile.shape) == None

    left_border = partial_tile.updated(x=(100, 101))
    left_neighbor = partial_tile.get_neighbor_tile_adjacent_to(anchor=left_border, tile_shape=source_tile.shape)
    assert left_neighbor == partial_tile.updated(x=(0, 100))


def test_slice_relabeling_swap():
    slc = Interval5D(x=100, y=200, z=300, t=400, c=500)
    keymap = KeyMap(x="y", y="x")
    assert slc.relabeled(keymap) == Interval5D(y=100, x=200, z=300, t=400, c=500)


def test_slice_relabeling_shift():
    slc = Interval5D(x=100, y=200, z=300, t=400, c=500)
    keymap = KeyMap(x="y", y="z", z="x")
    assert slc.relabeled(keymap) == Interval5D(y=100, z=200, x=300, t=400, c=500)


def test_slice_enclosing():
    p1 = Point5D(x=-13, y=  40, z= 0, t=0,   c=0)
    p2 = Point5D(x=  0, y=  60, z=-1, t=0,   c=6)
    p3 = Point5D(x=  4, y=  70, z= 0, t=3,   c=0)
    p4 = Point5D(x=  0, y= 800, z= 0, t=100, c=0)

    expected_slice = Interval5D(x=(-13, 4 + 1), y=(40, 800 + 1), z=(-1, 0 + 1), t=(0, 100+1), c=(0, 6 + 1))
    print(f"\nexpected_slice: {expected_slice}")
    print(f"enclosing: {Interval5D.enclosing([p1, p2, p3, p4])}")

    assert Interval5D.enclosing([p1, p2, p3, p4]) == expected_slice

    assert Interval5D.enclosing([p2]).start == p2

def test_is_tile():
    # normal tile, contained within full_interval
    assert Interval5D.zero(x=(110, 120), y=(310, 320)).is_tile(
        tile_shape=Shape5D(x=10, y=10),
        full_interval=Interval5D.zero(x=(100, 200), y=(300, 400)),
        clamped=False
    )

    # tile clipped on x
    assert Interval5D.zero(x=(200, 207), y=(300, 310)).is_tile(
        tile_shape=Shape5D(x=10, y=10),
        full_interval=Interval5D.zero(x=(100, 207), y=(300, 402)),
        clamped=True
    )

    # tile that is too short on x
    assert not Interval5D.zero(x=(200, 206), y=(300, 310)).is_tile(
        tile_shape=Shape5D(x=10, y=10),
        full_interval=Interval5D.zero(x=(100, 207), y=(300, 402)),
        clamped=True
    )

    #tile is clamped on x
    assert Interval5D.zero(x=(200, 210), y=(300, 400)).is_tile(
        tile_shape=Shape5D(x=100, y=100),
        full_interval=Interval5D.zero(x=(100, 210), y=(0, 1000)),
        clamped=True
    )

def test_json_serialization():
    interval = Interval5D(x=(100, 200), y=(200, 300), z=(400, 500), t=(600, 700), c=(700, 800))
    assert Interval5D.from_json_value(json.loads(json.dumps(interval.to_json_value()))) == interval

def test_counting_num_tiles():
    for _ in range(5):
        x_start = random.randint(10, 20)
        y_start = random.randint(10, 20)
        z_start = random.randint(10, 20)
        t_start = random.randint(10, 20)
        c_start = random.randint(10, 20)

        interval = Interval5D(
            x=(x_start, x_start + random.randint(1, 10)),
            y=(y_start, y_start + random.randint(1, 10)),
            z=(z_start, z_start + random.randint(1, 10)),
            t=(t_start, t_start + random.randint(1, 10)),
            c=(c_start, c_start + random.randint(1, 10)),
        )

        tile_shape = Shape5D(
            x=5, y=5, z=5, t=5, c=5
        )
        counted_tiles = sum(1 for _ in interval.split(block_shape=tile_shape))
        assert counted_tiles == interval.get_num_tiles(tile_shape=tile_shape)