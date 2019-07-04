from ndstructs import Point5D, Shape5D, Slice5D, Array5D
import numpy


def test_creation():
    raw = numpy.random.rand(10,20,30)
    arr = Array5D(raw, 'xyz')
    assert arr.shape == Shape5D(x=10, y=20, z=30)

def test_allocation():
    arr = Array5D.allocate(Slice5D.zero(x=slice(100, 200), y=slice(200,300)), numpy.uint8)
    assert arr.shape == Shape5D(x=100, y=100)
    assert arr.location == Point5D.zero(x=100, y=200)

    arr = Array5D.allocate(Slice5D.zero(x=slice(-100, 200), y=slice(200,300)), numpy.uint8)
    assert arr.shape == Shape5D(x=300, y=100)
    assert arr.location == Point5D.zero(x=-100, y=200)

def test_raw():
    raw = numpy.random.rand(10,20,30)
    arr = Array5D(raw, 'xyz')
    assert raw.shape == (10,20,30)
    assert (arr.raw('xyz') == raw).all()

    raw2 = numpy.asarray([[1,2,3],
                          [4,5,6]])
    arr2 = Array5D(raw2, 'xy')
    assert (arr2.raw('yx') == numpy.asarray([[1, 4],
                                             [2, 5],
                                             [3, 6]])).all()

    raw3 = numpy.asarray([
        [[1  ,2,  3],
         [4,  5,  6]],

        [[7,  8,  9],
         [10, 11, 12]]
    ])
    arr3 = Array5D(raw3, 'xyz')
    zxy_raw = arr3.raw('zxy')
    zxy_manual = numpy.asarray([
        [[1,4], [7,10]],
        [[2,5], [8,11]],
        [[3,6], [9,12]]
    ])
    assert (zxy_raw == zxy_manual).all()

def test_cut():
    raw = numpy.asarray([
        [1,  2,  3,  4,  5],
        [6,  7,  8,  9,  10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
    ])
    arr = Array5D(raw, 'zy')
    piece = arr.cut(Slice5D(y=slice(1,3)))
    assert (piece.raw('zy') == numpy.asarray([
        [2,  3],
        [7,  8],
        [12, 13],
        [17, 18]
    ])).all()

    assert piece.location == Point5D.zero(y=1)

    global_sub_piece = piece.cut(Slice5D(y=2))
    assert (global_sub_piece.raw('zy') == numpy.asarray([
        [3],
        [8],
        [13],
        [18]
    ])).all()

    local_sub_piece = piece.local_cut(Slice5D(y=1))
    assert (local_sub_piece.raw('zy') == global_sub_piece.raw('zy')).all()

def test_setting_rois():
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
    arr = Array5D(raw, 'cyx')

    piece = Array5D(numpy.asarray([
        [100, 200],
        [300, 400]
    ]), 'yx', location=Point5D.zero(x=2, y=1, c=1))

    arr.set(piece)
    expected_cyx_raw = numpy.asarray([
        [[1,   2,   3,   4,   5 ],
         [6,   7,   8,   9,   10],
         [11,  12,  13,  14,  15],
         [16,  17,  18,  19,  20]],

        [[-1,  -2,  -3,  -4,  -5 ],
         [-6,  -7,  100, 200,  -10],
         [-11, -12, 300, 400, -15],
         [-16, -17, -18, -19, -20]],
    ])

    assert (arr.raw('cyx') == expected_cyx_raw).all()

    extrapolating_piece = Array5D(numpy.asarray([
        [111, 222, 333],
        [444, 555, 6661]
    ]), 'yx', location=Point5D.zero(x=3, y=2, c=0))


    arr.set(extrapolating_piece, autocrop=True)
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

    assert (arr.raw('cyx') == expected_cyx_raw_with_extrapolating_piece).all()

def test_clamping():
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
    arr = Array5D(raw, 'zyx')
    clamped_raw = arr.clamped(Slice5D(z=1, x=slice(1,4), y=slice(1,3))).raw('zyx')
    assert (clamped_raw == numpy.asarray([
        [-7,  -8,  -9],
        [-12, -13, -14]
    ])).all()

def test_sample_channels():
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
    ]), 'cyx')

    mask = Array5D(numpy.asarray([
        [1,  1,  1,  0,  0],
        [0,  0,  1,  0,  0],
        [0,  0,  1,  0,  0],
        [0,  0,  1,  1,  1],
    ]), 'yx').as_mask()

    samples = arr.sample_channels(mask)
    expected_raw_samples = numpy.asarray([
        [1, -1, 10], [2,-2,20], [3,  -3,  30],
                                [8,  -8,  31],
                                [13, -13, 32],
                                [18, -18, 33], [19, -19, 43], [20, -20, 53]
    ])

    assert (samples.linear_raw() == expected_raw_samples).all()
