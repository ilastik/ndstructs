import pytest
import os
import tempfile
import numpy as np
from ndstructs import Shape5D, Slice5D
from ndstructs.datasource import DataSource, PilDataSource
from PIL import Image as PilImage

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


@pytest.fixture
def png_image() -> str:
    pil_image = PilImage.fromarray(raw)
    _, png_path = tempfile.mkstemp()
    with open(png_path, "wb") as png_file:
        pil_image.save(png_file, "png")
    yield png_path
    os.remove(png_path)


def tile_equals(tile: DataSource, axiskeys: str, raw: np.ndarray):
    return (tile.retrieve().raw(axiskeys) == raw).all()


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
