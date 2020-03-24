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


def test_get_borders():
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

    expected_thin_borders = {
        "left_border": Array5D(numpy.asarray([
            [[1 ],
             [6 ],
             [11],
             [16]],

            [[-1 ],
             [-6 ],
             [-11],
             [-16]],

            [[10],
             [11],
             [12],
             [13]],
        ]), "cyx"),

        "top_border": Array5D(numpy.asarray([
            [[1,   2,   3,   4,   5 ]],

            [[-1,  -2,  -3,  -4,  -5 ]],

            [[10,  20,  30,  40,  50]]
        ]), "cyx"),

        "right_border": Array5D(numpy.asarray([
            [[5 ],
             [10],
             [15],
             [20]],

            [[-5 ],
             [-10],
             [-15],
             [-20]],

            [[50],
             [51],
             [52],
             [53]],
        ]), "cyx"),

        "bottom_border": Array5D(numpy.asarray([
            [[16,  17,  18,  19,  20]],

            [[-16, -17, -18, -19, -20]],

            [[13,  23,  33,  43,  53]],
        ]), "cyx")
    }
    # fmt: on
    for border_data in arr.get_borders(thickness=Shape5D.zero(x=1, y=1)):
        for expected_border in expected_thin_borders.values():
            if (border_data.raw("cyx") == expected_border.raw("cyx")).all():
                break
        else:
            raise Exception(f"Could not find this border in the expected set:\n{border_data.raw('cyx')}")


def test_connected_components():
    # fmt: off
    arr = Array5D(numpy.asarray([
        [7, 7, 0, 0, 0, 0],
        [7, 7, 0, 0, 0, 0],
        [7, 0, 0, 0, 0, 0],
        [0, 0, 0, 3, 0, 0],
        [0, 0, 3, 3, 3, 0],
        [0, 0, 0, 3, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]]), axiskeys="yx")

    expected = Array5D(numpy.asarray([
        [1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 0, 0],
        [0, 0, 2, 2, 2, 0],
        [0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]]), axiskeys="yx")
    # fmt: on

    labeled = list(arr.connected_components())
    assert len(labeled) == 1
    assert (labeled[0].raw("yx") == expected.raw("yx")).all()


def test_color_filter():
    # fmt: off
    arr = Array5D(numpy.asarray([
        [[100,  20, 30, 100],
         [ 11,  21, 31,  41],
         [ 12,  22, 32,  42],
         [ 13,  23, 33,  43]],

        [[200, 24, 34, 200],
         [ 15, 25, 35,  45],
         [ 16, 26, 36,  46],
         [ 17, 27, 37,  47]]
    ]), axiskeys="cyx")

    color = Array5D(numpy.asarray([100, 200]), axiskeys="c")

    expected_color_filtered = Array5D(numpy.asarray([
        [[100,  0,  0, 100],
         [ 0,   0,  0,   0],
         [ 0,   0,  0,   0],
         [ 0,   0,  0,   0]],

        [[200,  0,  0, 200],
         [  0,  0,  0,   0],
         [  0,  0,  0,   0],
         [  0,  0,  0,   0]]
    ]), axiskeys="cyx")
    # fmt: on

    filtered = arr.color_filtered(color=color)
    assert filtered == expected_color_filtered


def test_unique_colors():
    # fmt: off
    img_c_as_first_axis = Array5D(numpy.asarray([
        [[100,  0,  0,  100],
         [ 0,   17,  0,   0],
         [ 0,   0,  17,   0],
         [ 0,   0,  0,    0]],

        [[200,   0,   0, 200],
         [  0,  40,   0,   0],
         [  0,   0,  40,   0],
         [  0,   0,   0,   0]]
    ]), axiskeys="cyx")

    img_c_as_last_axis = Array5D(numpy.asarray([
        [[100,  200],
         [ 0,     0],
         [ 0,     0],
         [ 0,     0]],

        [[100, 200],
         [ 17,  40],
         [  0,   0],
         [  0,   0]]
    ]), axiskeys="yxc")
    # fmt: on

    for img in (img_c_as_first_axis, img_c_as_last_axis):
        unique_colors = [list(color) for color in img.unique_colors().linear_raw()]
        for expected_color in [[0, 0], [17, 40], [100, 200]]:
            unique_colors.pop(unique_colors.index(expected_color))
        assert len(unique_colors) == 0


def test_paint_point():
    # fmt: off
    img = Array5D(numpy.asarray([
        [[100,   0,   0,  100],
         [ 0,   17,   0,    0],
         [ 0,    0,  17,    0],
         [ 0,    0,   0,    0]],

        [[200,   0,   0, 200],
         [  0,  40,   0,   0],
         [  0,   0,  40,   0],
         [  0,   0,   0,   0]]
    ]), axiskeys="cyx")
    # fmt: on

    # fmt: off
    expected_painted = Array5D(
        numpy.asarray(
            [
                [[107, 0, 0, 100], [0, 17, 0, 0], [0, 0, 17, 0], [0, 0, 0, 0]],
                [[200, 0, 0, 200], [0, 40, 123, 0], [0, 0, 40, 0], [0, 0, 0, 0]],
            ]
        ),
        axiskeys="cyx",
    )
    # fmt: on

    img.paint_point(Point5D.zero(c=0, y=0, x=0), value=107)
    img.paint_point(Point5D.zero(c=1, y=1, x=2), value=123)
    assert img == expected_painted
