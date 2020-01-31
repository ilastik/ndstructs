import pytest
import os
import tempfile
import numpy as np
from ndstructs import Shape5D, Slice5D, Array5D
from ndstructs.datasource import DataSource, SkimageDataSource, N5DataSource, H5DataSource, SequenceDataSource
from ndstructs.datasource import BackedSlice5D, MismatchingAxisKeysException
import z5py
import h5py
import json
import shutil
import skimage.io

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


def create_png(array: Array5D):
    png_path = tempfile.mkstemp()[1] + ".png"
    skimage.io.imsave(png_path, array.raw("yxc"))
    return png_path


def create_n5(array: Array5D, axiskeys: str = "xyztc"):
    path = tempfile.mkstemp()[1] + ".n5"
    f = z5py.File(path, use_zarr_format=False)
    ds = f.create_dataset(
        "data", shape=array.shape.to_tuple(axiskeys), chunks=(10,) * len(axiskeys), dtype=array.dtype.name
    )

    ds[...] = array.raw(axiskeys)
    ds.attrs["axes"] = list(reversed(list(axiskeys)))
    return path + "/data"


def create_h5(array: Array5D, axiskeys_style: str, chunk_shape: Shape5D = None, axiskeys="xyztc"):
    chunk_shape = chunk_shape.to_tuple(axiskeys) if chunk_shape else (10,) * len(axiskeys)

    path = tempfile.mkstemp()[1] + ".h5"
    f = h5py.File(path, "w")
    ds = f.create_dataset("data", shape=array.shape.to_tuple(axiskeys), chunks=chunk_shape, dtype=array.dtype.name)

    ds[...] = array.raw(axiskeys)
    if axiskeys_style == "dims":
        for key, dim in zip(axiskeys, ds.dims):
            dim.label = key
    elif axiskeys_style == "vigra":
        tagged_dict = {"axes": []}
        for key in axiskeys:
            tagged_dict["axes"].append({"key": key, "typeFlags": 2, "resolution": 0, "description": ""})
        ds.attrs["axistags"] = json.dumps(tagged_dict)
    else:
        raise Exception(f"Bad axiskeys_style: {axiskeys_style}")

    return path + "/data"


@pytest.fixture
def png_image() -> str:
    png_path = create_png(Array5D(raw, axiskeys="yx"))
    yield png_path
    os.remove(png_path)


@pytest.fixture
def raw_as_n5(tmp_path):
    raw_n5_path = tmp_path / "raw.n5"
    f = z5py.File(str(raw_n5_path), use_zarr_format=False)
    dataset = f.create_dataset("data", shape=raw.shape, chunks=(2, 2), dtype=raw.dtype)
    dataset[...] = raw
    dataset.attrs["axes"] = list(reversed(["y", "x"]))
    f.close()
    yield (f, raw_n5_path)
    shutil.rmtree(raw_n5_path)


def tile_equals(tile: DataSource, axiskeys: str, raw: np.ndarray):
    return (tile.retrieve().raw(axiskeys) == raw).all()


def test_n5_datasource(raw_as_n5):
    n5_file, path = raw_as_n5
    ds = N5DataSource(path / "data")
    assert ds.shape == Shape5D(y=4, x=5)
    assert ds.tile_shape == Shape5D(y=2, x=2)

    piece = BackedSlice5D(ds).clamped(Slice5D(x=slice(0, 3), y=slice(0, 2)))
    expected_raw_piece = np.asarray([[1, 2, 3], [6, 7, 8]]).astype(np.uint8)
    assert tile_equals(piece, "yx", expected_raw_piece)


def test_h5_datasource():
    data_2d = Array5D(np.arange(100).reshape(10, 10), axiskeys="yx")
    h5_path = create_h5(data_2d, axiskeys_style="vigra", chunk_shape=Shape5D(x=3, y=3))
    bs = BackedSlice5D(H5DataSource(h5_path))
    assert bs.full_shape == data_2d.shape
    assert bs.tile_shape == Shape5D(x=3, y=3)

    slc = Slice5D(x=slice(0, 3), y=slice(0, 2))
    assert (bs.clamped(slc).retrieve().raw("yx") == data_2d.cut(slc).raw("yx")).all()

    data_3d = Array5D(np.arange(10 * 10 * 10).reshape(10, 10, 10), axiskeys="zyx")
    h5_path = create_h5(data_3d, axiskeys_style="vigra", chunk_shape=Shape5D(x=3, y=3))
    bs = BackedSlice5D(DataSource.create(h5_path))
    assert bs.full_shape == data_3d.shape
    assert bs.tile_shape == Shape5D(x=3, y=3)

    slc = Slice5D(x=slice(0, 3), y=slice(0, 2), z=3)
    assert (bs.clamped(slc).retrieve().raw("yxz") == data_3d.cut(slc).raw("yxz")).all()


def test_skimage_datasource_tiles(png_image: str):
    bs = BackedSlice5D(SkimageDataSource(png_image))
    num_checked_tiles = 0
    for tile in bs.get_tiles(Shape5D(x=2, y=2)):
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

    ds = SkimageDataSource(create_png(arr))

    fifties_slice = BackedSlice5D(ds).clamped(Slice5D(x=slice(3, 6), y=slice(3, 6)))
    expected_fifties_slice = Array5D(np.asarray([
        [50, 51, 52],
        [53, 54, 55],
        [56, 57, 58]
    ]), axiskeys="yx")
    # fmt: on

    top_slice = BackedSlice5D(ds, x=slice(3, 6), y=slice(0, 3))
    bottom_slice = BackedSlice5D(ds, x=slice(3, 6), y=slice(6, 9))

    right_slice = BackedSlice5D(ds, x=slice(6, 7), y=slice(3, 6))
    left_slice = BackedSlice5D(ds, x=slice(0, 3), y=slice(3, 6))

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


def test_sequence_datasource():
    # fmt: off
    img1_data = Array5D(np.asarray([
       [[100, 101, 102, 103, 104],
        [105, 106, 107, 108, 109],
        [110, 111, 112, 113, 114],
        [115, 116, 117, 118, 119]],

       [[120, 121, 122, 123, 124],
        [125, 126, 127, 128, 129],
        [130, 131, 132, 133, 134],
        [135, 136, 137, 138, 139]],

       [[140, 141, 142, 143, 144],
        [145, 146, 147, 148, 149],
        [150, 151, 152, 153, 154],
        [155, 156, 157, 158, 159]]
    ]), axiskeys="cyx")

    img2_data = Array5D(np.asarray([
       [[200, 201, 202, 203, 204],
        [205, 206, 207, 208, 209],
        [210, 211, 212, 213, 214],
        [215, 216, 217, 218, 219]],

       [[220, 221, 222, 223, 224],
        [225, 226, 227, 228, 229],
        [230, 231, 232, 233, 234],
        [235, 236, 237, 238, 239]],

       [[240, 241, 242, 243, 244],
        [245, 246, 247, 248, 249],
        [250, 251, 252, 253, 254],
        [255, 256, 257, 258, 259]]
    ]), axiskeys="cyx")

    img3_data = Array5D(np.asarray([
       [[300, 301, 302, 303, 304],
        [305, 306, 307, 308, 309],
        [310, 311, 312, 313, 314],
        [315, 316, 317, 318, 319]],

       [[320, 321, 322, 323, 324],
        [325, 326, 327, 328, 329],
        [330, 331, 332, 333, 334],
        [335, 336, 337, 338, 339]],

       [[340, 341, 342, 343, 344],
        [345, 346, 347, 348, 349],
        [350, 351, 352, 353, 354],
        [355, 356, 357, 358, 359]]
    ]), axiskeys="cyx")

    expected_x_2_4__y_1_3 = Array5D(np.asarray([
      [[[107, 108],
        [112, 113]],

       [[127, 128],
        [132, 133]],

       [[147, 148],
        [152, 153]]],


      [[[207, 208],
        [212, 213]],

       [[227, 228],
        [232, 233]],

       [[247, 248],
        [252, 253]]],


      [[[307, 308],
        [312, 313]],

       [[327, 328],
        [332, 333]],

       [[347, 348],
        [352, 353]]],
    ]), axiskeys="zcyx")
    # fmt: on

    urls = [
        create_n5(img1_data, axiskeys="cyx"),
        create_n5(img2_data, axiskeys="cyx"),
        create_n5(img3_data, axiskeys="cyx"),
    ]
    combined_url = os.path.pathsep.join(urls)

    seq_ds = SequenceDataSource(combined_url, stack_axis="z")
    data = BackedSlice5D(seq_ds, x=slice(2, 4), y=slice(1, 3)).retrieve()
    assert (expected_x_2_4__y_1_3.raw("xyzc") == data.raw("xyzc")).all()

    seq_ds = SequenceDataSource(combined_url, stack_axis="z", tile_shape=Shape5D(x=2, y=3, z=2))
    data = BackedSlice5D(seq_ds, x=slice(2, 4), y=slice(1, 3)).retrieve()
    assert (expected_x_2_4__y_1_3.raw("xyzc") == data.raw("xyzc")).all()

    seq_ds = SequenceDataSource(combined_url, stack_axis="c")
    expected_c = sum([img1_data.shape.c, img2_data.shape.c, img3_data.shape.c])
    assert seq_ds.shape == img1_data.shape.with_coord(c=expected_c)

    seq_ds = SequenceDataSource(combined_url, stack_axis="c", slice_axiskeys="zyx")
    data = BackedSlice5D(seq_ds, x=slice(2, 4), y=slice(1, 3)).retrieve()
    assert (expected_x_2_4__y_1_3.raw("xyzc") == data.raw("xycz")).all()


def test_datasource_axiskeys():
    data = Array5D(np.arange(200).astype(np.uint8).reshape(20, 10), "xy")
    assert data.shape == Shape5D(x=20, y=10)

    png_path = create_png(data)
    n5_path = create_n5(data, axiskeys="xy")
    h5_path = create_h5(data, axiskeys_style="dims", axiskeys="xy")

    for p in [png_path, n5_path, h5_path]:
        print(p)
        with pytest.raises(MismatchingAxisKeysException):
            ds = DataSource.create(p, axiskeys="xyz")

    ds = DataSource.create(png_path, axiskeys="zy")
    assert ds.shape == Shape5D(z=data.shape.y, y=data.shape.x)

    ds = DataSource.create(n5_path, axiskeys="zx")
    assert ds.shape == Shape5D(z=data.shape.x, x=data.shape.y)

    ds = DataSource.create(h5_path, axiskeys="zx")
    assert ds.shape == Shape5D(z=data.shape.x, x=data.shape.y)
