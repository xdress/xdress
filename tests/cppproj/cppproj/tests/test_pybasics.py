import numpy as np

from nose.tools import assert_equal, assert_true
from numpy.testing import assert_array_equal, assert_array_almost_equal

from cppproj import pybasics

def test_func4():
    exp = 42
    obs = pybasics.func4(42)
    assert_equal(exp, obs)
    
