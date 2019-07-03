from ndstructs import Point5D, Shape5D, Slice5D
import numpy

def test_slice_contains_smaller_slice():
    outer_slice = Slice5D(x=slice(10,100), y=slice(20,200))
    inner_slice = Slice5D(x=slice(20, 50), y=slice(30, 40), z=0, t=0, c=0)
    assert outer_slice.contains(inner_slice)

def test_slice_does_not_contain_translated_slice():
    slc = Slice5D(x=slice(10,100), y=slice(20,200), z=0, t=0, c=0)
    translated_slc = slc.translated(Point5D.zero(x=10))
    assert not slc.contains(translated_slc)
