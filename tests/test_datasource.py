import pytest
import os
import tempfile
import numpy as np
from ndstructs import Shape5D, Slice5D, Array5D
from ndstructs.datasource import DataSource, PilDataSource, N5DataSource
from PIL import Image as PilImage
import z5py
import shutil

# fmt: off
raw = np.asarray([
    [1,  2,  3,  4,  5],
    [6,  7,  8,  9,  10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
]).astype(np.uint8)

raw_0_2x0_2y = np.asarray([
    [1,2],
    [6,7]
])

raw_0_2x2_4y = np.asarray([
    [11,12],
    [16,17]
])

raw_2_4x0_2y = np.asarray([
    [3,4],
    [8,9]
])

raw_2_4x2_4y = expected_raw = np.asarray([
    [13,14],
    [18,19]
])

raw_4_5x0_2y = np.asarray([
    [5],
    [10]
])

raw_4_5x2_4y = np.asarray([
    [15],
    [20]
])
# fmt: on


def get_png_image(array):
    pil_image = PilImage.fromarray(array)
    _, png_path = tempfile.mkstemp()
    with open(png_path, "wb") as png_file:
        pil_image.save(png_file, "png")
    return png_path


@pytest.fixture
def png_image() -> str:
    png_path = get_png_image(raw)
    yield png_path
    os.remove(png_path)


@pytest.fixture
def raw_as_n5(tmp_path):
    raw_n5_path = tmp_path / "raw.n5"
    f = z5py.File(raw_n5_path, use_zarr_format=False)
    dataset = f.create_dataset("data", shape=raw.shape, chunks=(2, 2), dtype=raw.dtype)
    dataset[...] = raw
    dataset.attrs["axes"] = list(reversed(["y", "x"]))
    f.close()
    yield f
    shutil.rmtree(raw_n5_path)


def tile_equals(tile: DataSource, axiskeys: str, raw: np.ndarray):
    return (tile.retrieve().raw(axiskeys) == raw).all()


def test_n5_datasource(raw_as_n5):
    path = raw_as_n5.path / "data"
    ds = N5DataSource(path)
    assert ds.roi == Shape5D(y=4, x=5).to_slice_5d()
    assert ds.full_shape == Shape5D(y=4, x=5)
    assert ds.tile_shape == Shape5D(y=2, x=2)

    piece = ds.clamped(Slice5D(x=slice(0, 3), y=slice(0, 2)))
    expected_raw_piece = np.asarray([[1, 2, 3], [6, 7, 8]]).astype(np.uint8)
    assert tile_equals(piece, "yx", expected_raw_piece)


def test_pil_datasource_tiles(png_image: str):
    ds = PilDataSource(png_image)
    num_checked_tiles = 0
    for tile in ds.get_tiles(Shape5D(x=2, y=2)):
        if tile == Slice5D.zero(x=slice(0, 2), y=slice(0, 2)):
            expected_raw = raw_0_2x0_2y
        elif tile == Slice5D.zero(x=slice(0, 2), y=slice(2, 4)):
            expected_raw = raw_0_2x2_4y
        elif tile == Slice5D.zero(x=slice(2, 4), y=slice(0, 2)):
            expected_raw = raw_2_4x0_2y
        elif tile == Slice5D.zero(x=slice(2, 4), y=slice(2, 4)):
            expected_raw = raw_2_4x2_4y
        elif tile == Slice5D.zero(x=slice(4, 5), y=slice(0, 2)):
            expected_raw = raw_4_5x0_2y
        elif tile == Slice5D.zero(x=slice(4, 5), y=slice(2, 4)):
            expected_raw = raw_4_5x2_4y
        else:
            raise Exception(f"Unexpected tile {tile}")
        assert (tile.retrieve().raw("yx") == expected_raw).all()
        num_checked_tiles += 1
    assert num_checked_tiles == 6


def test_neighboring_tiles():
    # fmt: off
    arr = Array5D(np.asarray([
        [10, 11, 12,   20, 21, 22,   30],
        [13, 14, 15,   23, 24, 25,   33],
        [16, 17, 18,   26, 27, 28,   36],

        [40, 41, 42,   50, 51, 52,   60],
        [43, 44, 45,   53, 54, 55,   63],
        [46, 47, 48,   56, 57, 58,   66],

        [70, 71, 72,   80, 81, 82,   90],
        [73, 74, 75,   83, 84, 85,   93],
        [76, 77, 78,   86, 87, 88,   96],

        [0,   1,  2,    3,  4,  5,    6]], dtype=np.uint8), axiskeys="yx")

    ds = PilDataSource(get_png_image(arr.raw('yx')))

    fifties_slice = ds.clamped(Slice5D(x=slice(3, 6), y=slice(3, 6)))
    expected_fifties_slice = Array5D(np.asarray([
        [50, 51, 52],
        [53, 54, 55],
        [56, 57, 58]
    ]), axiskeys="yx")
    # fmt: on

    top_slice = ds.resize(Slice5D(x=slice(3, 6), y=slice(0, 3)))
    bottom_slice = ds.resize(Slice5D(x=slice(3, 6), y=slice(6, 9)))

    right_slice = ds.resize(Slice5D(x=slice(6, 7), y=slice(3, 6)))
    left_slice = ds.resize(Slice5D(x=slice(0, 3), y=slice(3, 6)))

    # fmt: off
    fifties_neighbor_data = {
        top_slice: Array5D(np.asarray([
            [20, 21, 22],
            [23, 24, 25],
            [26, 27, 28]
        ]), axiskeys="yx"),

        right_slice: Array5D(np.asarray([
            [60],
            [63],
            [66]
        ]), axiskeys="yx"),

        bottom_slice: Array5D(np.asarray([
            [80, 81, 82],
            [83, 84, 85],
            [86, 87, 88]
        ]), axiskeys="yx"),

        left_slice: Array5D(np.asarray([
            [40, 41, 42],
            [43, 44, 45],
            [46, 47, 48]
        ]), axiskeys="yx"),
    }

    expected_fifties_neighbors = {
    }
    # fmt: on

    assert (fifties_slice.retrieve().raw("yx") == expected_fifties_slice.raw("yx")).all()

    for neighbor in fifties_slice.get_neighboring_tiles(tile_shape=Shape5D(x=3, y=3)):
        try:
            expected_slice = fifties_neighbor_data.pop(neighbor)
            print("\nFound neighbor ", neighbor)
            assert (expected_slice.raw("yx") == neighbor.retrieve().raw("yx")).all()
        except KeyError:
            print(f"\nWas searching for ", neighbor, "\n")
            for k in fifties_neighbor_data.keys():
                print("--->>> ", k)
    assert len(fifties_neighbor_data) == 0
