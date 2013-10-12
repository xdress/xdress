from __future__ import print_function
import sys

from nose.tools import assert_equal, assert_true
from numpy.testing import assert_array_equal, assert_array_almost_equal

from cppproj import pybasics

def test_func4():
    exp = 42
    obs = pybasics.func4(42)
    assert_equal(exp, obs)
    
def test_person_id():
    assert_equal(pybasics.JOAN, 0)
    assert_equal(pybasics.HOOVER, 1)
    assert_equal(pybasics.MULAN, 42)
    assert_equal(pybasics.LESLIE, 43)

def test_three_nums():
    # instantiation
    x = pybasics.ThreeNums()

    # assignemt
    x.a, x.b, x.c = 5, 10, 15
    assert_equal(x.a, 5.0)    
    assert_equal(x.b, 10.0)
    assert_equal(x.c, 15.0)    
    print("a and b and c = ", x.a, x.b, x.c)

    # callback assignment
    f = lambda a_, b_, c_: a_ + b_ + c_
    print("f's refcount before assignment: ", sys.getrefcount(f))
    x.op = f
    print("f's refcount after assignment: ", sys.getrefcount(f))
    print("x.op = ", x.op)

    # Test that function pointer in C is getting set properly
    x._op = None
    print("x._op is None:", x._op is None)
    print("f's refcount after None assignment: ", sys.getrefcount(f))
    print("x.op = ", x.op)

    v = x.op(14, 16, 17)
    assert_equal(47.0, v)
    print("result of x.op(14, 16, 17) = ", v)
    v = pybasics.call_threenums_op_from_c(x)
    assert_equal(30.0, v)
    print("result of call_threenums_op_from_c(x) = ", v)

    print("-"*40)

    x.op = lambda a_, b_, c_: 2*a_ + b_**2 - c_/2.0
    print("x.op = ", x.op)
    print("f's refcount after op re-assignment: ", sys.getrefcount(f))

    v = x.op(14, 16, 17)
    assert_equal(275.5, v)
    print("result of x.op(14, 16, 17) = ", v)
    v = pybasics.call_threenums_op_from_c(x)
    assert_equal(102.5, v)
    print("result of call_threenums_op_from_c(x) = ", v)

    print("-"*40)

    # test two instances
    y = pybasics.ThreeNums()
    y.a, y.b, y.c = 50, 100, 150
    x.op = lambda a_, b_, c_: a_ + b_ + c_
    print("x.op = ", x.op)
    y.op = lambda a_, b_, c_: 2*a_ + b_**2 - c_/2.0
    print("y.op = ", y.op)
    v = x.op(14, 16, 17)
    assert_equal(47.0, v)
    print("result of x.op(14, 16, 17) = ", v)
    v = pybasics.call_threenums_op_from_c(x)
    assert_equal(30.0, v)
    print("result of call_threenums_op_from_c(x) = ", v)
