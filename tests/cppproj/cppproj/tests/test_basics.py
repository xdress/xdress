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
    
def test_voided():
    assert_true(basics.voided() is None)


def test_findmin_int_float():
    exp = 42
    obs = basics.findmin_int_float(65, 42.0)
    assert_equal(exp, obs)

    exp = 42
    obs = basics.findmin[int, float](42, 65.0)
    assert_equal(exp, obs)

def test_tclass1double():
    x = basics.TClass1Double()
    y = basics.TClass1[float]()
    z = basics.TClass1['float64']()

def test_regmin_int_int():
    exp = 42
    obs = basics.regmin_int_int(65, 42)
    assert_equal(exp, obs)

    exp = 42
    obs = basics.regmin[int, int](42, 65)
    assert_equal(exp, obs)

def test_tc1floater():
    x = basics.TC1Floater()

def test_tc0boolbool():
    x = basics.TC0BoolBool()
    y = basics.TC0Bool[bool]()
    z = basics.TC0Bool['bool']()

