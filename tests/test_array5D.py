from ndstructs import Point5D, Shape5D, Slice5D, Array5D
import numpy


def test_creation():
    raw = numpy.random.rand(10, 20, 30)
    arr = Array5D(raw, "xyz")
    assert arr.shape == Shape5D(x=10, y=20, z=30)


def test_allocation():
    arr = Array5D.allocate(Slice5D.zero(x=slice(100, 200), y=slice(200, 300)), numpy.uint8)
    assert arr.shape == Shape5D(x=100, y=100)
    assert arr.location == Point5D.zero(x=100, y=200)

    arr = Array5D.allocate(Slice5D.zero(x=slice(-100, 200), y=slice(200, 300)), numpy.uint8)
    assert arr.shape == Shape5D(x=300, y=100)
    assert arr.location == Point5D.zero(x=-100, y=200)


def test_raw():
    raw = numpy.random.rand(10, 20, 30)
    arr = Array5D(raw, "xyz")
    assert raw.shape == (10, 20, 30)
    assert (arr.raw("xyz") == raw).all()

    # fmt: off
    raw2 = numpy.asarray([[1, 2, 3],
                          [4, 5, 6]])

    expected_raw2_yx = numpy.asarray([
        [1, 4],
        [2, 5],
        [3, 6]
    ])

    raw3 = numpy.asarray([
        [[1  ,2,  3],
         [4,  5,  6]],

        [[7,  8,  9],
         [10, 11, 12]]
    ])

    expected_raw3_zxy = numpy.asarray([
        [[1, 4], [7, 10]],
        [[2, 5], [8, 11]],
        [[3, 6], [9, 12]]
    ])
    # fmt: on

    arr2 = Array5D(raw2, "xy")
    assert (arr2.raw("yx") == expected_raw2_yx).all()

    arr3 = Array5D(raw3, "xyz")
    zxy_raw = arr3.raw("zxy")
    assert (zxy_raw == expected_raw3_zxy).all()


def test_cut():
    # fmt: off
    raw = numpy.asarray([
        [1,  2,  3,  4,  5],
        [6,  7,  8,  9,  10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
    ])
    expected_piece = numpy.asarray([
        [2,  3],
        [7,  8],
        [12, 13],
        [17, 18]
    ])
    expected_global_sub_piece = numpy.asarray([
        [3],
        [8],
        [13],
        [18]
    ])
    # fmt: on
    arr = Array5D(raw, "zy")
    piece = arr.cut(Slice5D(y=slice(1, 3)))
    assert (piece.raw("zy") == expected_piece).all()
    assert piece.location == Point5D.zero(y=1)

    global_sub_piece = piece.cut(Slice5D(y=2))
    assert (global_sub_piece.raw("zy") == expected_global_sub_piece).all()

    local_sub_piece = piece.local_cut(Slice5D(y=1))
    assert (local_sub_piece.raw("zy") == global_sub_piece.raw("zy")).all()


def test_setting_rois():
    # fmt: off
    raw = numpy.asarray([
        [[1,   2,   3,   4,   5 ],
         [6,   7,   8,   9,   10],
         [11,  12,  13,  14,  15],
         [16,  17,  18,  19,  20]],

        [[-1,  -2,  -3,  -4,  -5 ],
         [-6,  -7,  -8,  -9,  -10],
         [-11, -12, -13, -14, -15],
         [-16, -17, -18, -19, -20]],
    ])

    piece = Array5D(numpy.asarray([
        [100, 200],
        [300, 400]
    ]), "yx", location=Point5D.zero(x=2, y=1, c=1))

    expected_cyx_raw_with_piece = numpy.asarray([
        [[1,   2,   3,   4,   5 ],
         [6,   7,   8,   9,   10],
         [11,  12,  13,  14,  15],
         [16,  17,  18,  19,  20]],

        [[-1,  -2,  -3,  -4,  -5 ],
         [-6,  -7,  100, 200,  -10],
         [-11, -12, 300, 400, -15],
         [-16, -17, -18, -19, -20]],
    ])

    extrapolating_piece = Array5D(numpy.asarray([
        [111, 222, 333],
        [444, 555, 6661]
    ]), "yx", location=Point5D.zero(x=3, y=2, c=0))

    expected_cyx_raw_with_extrapolating_piece = numpy.asarray([
        [[1,   2,   3,   4,    5 ],
         [6,   7,   8,   9,    10],
         [11,  12,  13,  111,  222],
         [16,  17,  18,  444,  555]],

        [[-1,  -2,  -3,  -4,   -5 ],
         [-6,  -7,  100, 200,  -10],
         [-11, -12, 300, 400,  -15],
         [-16, -17, -18, -19,  -20]],
    ])
    # fmt: on
    arr = Array5D(raw, "cyx")
    arr.set(piece)
    assert (arr.raw("cyx") == expected_cyx_raw_with_piece).all()

    arr.set(extrapolating_piece, autocrop=True)
    assert (arr.raw("cyx") == expected_cyx_raw_with_extrapolating_piece).all()


def test_clamping():
    # fmt: off
    raw = numpy.asarray([
        [[1,   2,   3,   4,   5 ],
         [6,   7,   8,   9,   10],
         [11,  12,  13,  14,  15],
         [16,  17,  18,  19,  20]],

        [[-1,  -2,  -3,  -4,  -5 ],
         [-6,  -7,  -8,  -9,  -10],
         [-11, -12, -13, -14, -15],
         [-16, -17, -18, -19, -20]],
    ])
    expected_clamped_array = numpy.asarray([
        [-7,  -8,  -9],
        [-12, -13, -14]
    ])
    # fmt: on
    arr = Array5D(raw, "zyx")
    clamped_raw = arr.clamped(Slice5D(z=1, x=slice(1, 4), y=slice(1, 3))).raw("zyx")
    assert (clamped_raw == expected_clamped_array).all()


def test_sample_channels():
    # fmt: off
    arr = Array5D(numpy.asarray([
        [[1,   2,   3,   4,   5 ],
         [6,   7,   8,   9,   10],
         [11,  12,  13,  14,  15],
         [16,  17,  18,  19,  20]],

        [[-1,  -2,  -3,  -4,  -5 ],
         [-6,  -7,  -8,  -9,  -10],
         [-11, -12, -13, -14, -15],
         [-16, -17, -18, -19, -20]],

        [[10,  20,  30,  40,  50],
         [11,  21,  31,  41,  51],
         [12,  22,  32,  42,  52],
         [13,  23,  33,  43,  53]],
    ]), "cyx")

    mask = Array5D(numpy.asarray([
        [1,  1,  1,  0,  0],
        [0,  0,  1,  0,  0],
        [0,  0,  1,  0,  0],
        [0,  0,  1,  1,  1],
    ]), "yx").as_mask()

    expected_raw_samples = numpy.asarray([
        [1, -1, 10], [2,-2,20], [3,  -3,  30],
                                [8,  -8,  31],
                                [13, -13, 32],
                                [18, -18, 33], [19, -19, 43], [20, -20, 53]
    ])
    # fmt: on

    samples = arr.sample_channels(mask)
    assert (samples.linear_raw() == expected_raw_samples).all()
