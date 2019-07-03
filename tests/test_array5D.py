from ndstructs import Point5D, Shape5D, Slice5D, Array5D
import numpy


def test_creation():
    raw = numpy.random.rand(10,20,30)
    arr = Array5D(raw, 'xyz')
    assert arr.shape == Shape5D(x=10, y=20, z=30)

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
