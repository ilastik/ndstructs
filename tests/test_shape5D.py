from ndstructs import Point5D, Shape5D
import numpy
import pytest


def test_shape_coords_have_sensible_defaults():
    assert Shape5D(x=123, y=456) == Shape5D(x=123, y=456, z=1, t=1, c=1)
