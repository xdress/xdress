import numpy as np

from nose.tools import assert_equal, assert_true
from numpy.testing import assert_array_equal, assert_array_almost_equal

from cppproj import basics

def test_a_better_name():
    exp = np.zeros(5, float)
    obs = basics.a_better_name(3.0, range(5))
    assert_array_almost_equal(exp, obs)

def test_func1():
    assert_true(basics.func1({1: 10.0}, {2: 42.0}))

def test_func2():
    # FIXME
    #exp = 0.0
    #obs = basics.func2([13], [14])[0][0]
    exp = []
    obs = basics.func2([], [])
    assert_equal(len(exp), len(obs))

def test_func3():
    exp = -1
    obs = basics.func3("Waka Jawaka", ["Vows are apoken"], 42)
    assert_equal(exp, obs)
    
