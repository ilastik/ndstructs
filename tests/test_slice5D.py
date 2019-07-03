from ndstructs import Point5D, Shape5D, Slice5D
import numpy

def test_slice_translation():
    slc = Slice5D(x=slice(10,100), y=slice(20,200))
    translated_slc = slc.translated(Point5D(x=1,y=2,z=3,t=4,c=5))
    assert translated_slc == Slice5D(x=slice(11,101), y=slice(22,202))


    slc = Slice5D(x=slice(10,100), y=slice(20,200), z=0, t=0, c=0)
    translated_slc = slc.translated(Point5D(x=-1,y=-2,z=-3,t=-4,c=-5000))
    assert translated_slc == Slice5D(x=slice(9,99), y=slice(18,198), z=slice(-3,-2),
                                     t=slice(-4,-3), c=slice(-5000,-4999))

def test_slice_enlarge():
    slc = Slice5D(x=slice(10,100), y=slice(20,200))
    enlarged = slc.enlarged(radius=Point5D(x=1,y=2,z=3,t=4,c=5))
    assert enlarged == Slice5D(x=slice(9,101), y=slice(18,202))

    slc = Slice5D(x=slice(10,100), y=slice(20,200), z=0, t=0, c=0)
    enlarged = slc.enlarged(radius=Point5D(x=1,y=2,z=3,t=4,c=5))
    assert enlarged == Slice5D(x=slice(9,101), y=slice(18,202), z=slice(-3, 4),
                               t=slice(-4,5), c=slice(-5,6))

def test_slice_contains_smaller_slice():
    outer_slice = Slice5D(x=slice(10,100), y=slice(20,200))
    inner_slice = Slice5D(x=slice(20, 50), y=slice(30, 40), z=0, t=0, c=0)
    assert outer_slice.contains(inner_slice)

def test_slice_does_not_contain_translated_slice():
    slc = Slice5D(x=slice(10,100), y=slice(20,200), z=0, t=0, c=0)
    translated_slc = slc.translated(Point5D.zero(x=10))
    assert not slc.contains(translated_slc)

def test_slice_clamp():
    outer = Slice5D(x=slice(10,100), y=slice(20,200))
    inner = Slice5D(x=slice(20, 50), y=slice(30, 40), z=0, t=0, c=0)
    assert outer.clamped(inner) == inner
    assert inner.clamped(outer) == inner

    intersecting_outer = Slice5D(x=slice(50,200), y=slice(30, 900))
    assert intersecting_outer.clamped(outer) == Slice5D(x=slice(50, 100), y=slice(30, 200))

    intersecting_outer = Slice5D(x=slice(-100,50), y=slice(10, 100))
    assert intersecting_outer.clamped(outer) == Slice5D(x=slice(10, 50), y=slice(20, 100))

    outside_outer = Slice5D(x=slice(200,300), y=slice(400,500))
    assert outside_outer.clamped(outer).defined_with(Shape5D()).shape.volume == 0
