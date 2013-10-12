import numpy as np

from nose.tools import assert_equal, assert_true, assert_false
from numpy.testing import assert_array_equal, assert_array_almost_equal

from cproj import basics

def test_a_better_name():
    exp = 1
    obs = basics.a_better_name(3.0, 5)
    assert_array_almost_equal(exp, obs)

def test_func1():
    exp = 100
    obs = basics.func1(10, 42.0)
    assert_equal(exp, obs)

def test_func2():
    exp = "Doing it right."
    obs = basics.func2()
    assert_equal(len(exp), len(obs))

def test_func3():
    exp = -1
    obs = basics.func3("Waka Jawaka", ["Vows are apoken"], 42)
    assert_equal(exp, obs)
    
def test_voided():
    assert_true(basics.voided() is None)


