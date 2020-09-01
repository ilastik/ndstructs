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

    # fmt: off
    zcyx = Array5D(numpy.asarray(
        [[[[ 0,  1,  2],
           [ 3,  4,  5],
           [ 6,  7,  8],
           [ 9, 10, 11]],

          [[12, 13, 14],
           [15, 16, 17],
           [18, 19, 20],
           [21, 22, 23]]],


         [[[24, 25, 26],
           [27, 28, 29],
           [30, 31, 32],
           [33, 34, 35]],

          [[36, 37, 38],
           [39, 40, 41],
           [42, 43, 44],
           [45, 46, 47]]],


         [[[48, 49, 50],
           [51, 52, 53],
           [54, 55, 56],
           [57, 58, 59]],

          [[60, 61, 62],
           [63, 64, 65],
           [66, 67, 68],
           [69, 70, 71]]]]),
        axiskeys="zcyx")

    expected_raw_xycz = numpy.asarray(
      [[[[ 0, 24, 48],
         [12, 36, 60]],

        [[ 3, 27, 51],
         [15, 39, 63]],

        [[ 6, 30, 54],
         [18, 42, 66]],

        [[ 9, 33, 57],
         [21, 45, 69]]],


       [[[ 1, 25, 49],
         [13, 37, 61]],

        [[ 4, 28, 52],
         [16, 40, 64]],

        [[ 7, 31, 55],
         [19, 43, 67]],

        [[10, 34, 58],
         [22, 46, 70]]],


       [[[ 2, 26, 50],
         [14, 38, 62]],

        [[ 5, 29, 53],
         [17, 41, 65]],

        [[ 8, 32, 56],
         [20, 44, 68]],

        [[11, 35, 59],
         [23, 47, 71]]]])
    # fmt: on

    assert (zcyx.raw("xycz") == expected_raw_xycz).all()


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

    labeled = arr.connected_components()
    assert (labeled.raw("yx") == expected.raw("yx")).all()


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


def test_unique_border_colors():
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
        [0, 0, 5, 5, 0, 0]]), axiskeys="yx")
    # fmt: on

    border_colors = arr.unique_border_colors()
    assert border_colors.shape == Shape5D(x=len([7, 5, 0]))

    raw_colors = border_colors.raw("x")
    assert 7 in raw_colors
    assert 5 in raw_colors
    assert 0 in raw_colors

    # fmt: off
    arr_zyx = Array5D(numpy.asarray([
        [[7, 7, 0, 0, 0, 0],
         [7, 7, 0, 0, 0, 0],
         [7, 0, 0, 0, 0, 0],
         [0, 0, 0, 3, 0, 0],
         [0, 0, 3, 3, 3, 0],
         [0, 0, 0, 3, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 5, 5, 0, 0]],

        [[0, 0, 0, 2, 2, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 9, 0],
         [0, 0, 0, 0, 9, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]],
    ]), axiskeys="zyx")
    # fmt: on

    # import pydevd; pydevd.settrace()
    # get borders as if this was two separate plaes, as opposed to a single 3d block
    border_colors = arr_zyx.unique_border_colors(border_thickness=Shape5D.zero(x=1, y=1))
    print("===>>>>>", border_colors.raw("x"))
    assert border_colors.shape == Shape5D(x=len([7, 5, 0, 2]))

    raw_colors = border_colors.raw("x")
    assert 7 in raw_colors
    assert 5 in raw_colors
    assert 0 in raw_colors
    assert 2 in border_colors._data


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
