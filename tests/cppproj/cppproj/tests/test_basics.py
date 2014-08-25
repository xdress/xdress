import numpy as np

from nose.tools import assert_equal, assert_true, assert_false
from numpy.testing import assert_array_equal, assert_array_almost_equal

from cppproj import basics

def test_a_better_name():
    exp = np.zeros(5, float)
    obs = basics.a_better_name(3.0, range(5))
    assert_array_almost_equal(exp, obs)

def test_func1():
    a = {1: 10.0}
    b = {2: 42.0, 3: 4}
    assert_true (basics.func1(a, b))
    assert_false(basics.func1(b, a))
    # Check that keyword arguments work in any order
    assert_true (basics.func1(i=a, j=b))
    assert_true (basics.func1(j=b, i=a))
    assert_false(basics.func1(i=b, j=a))
    assert_false(basics.func1(j=a, i=b))

def test_func2():
    # FIXME
    #exp = [[0.0,0.0]]
    #obs = basics.func2([13], [14,15])
    #obs2 = basics.func2(b=[14,15], a=[13])
    #assert_equal(exp, obs)
    exp = []
    obs = basics.func2([], [])
    assert_equal(len(exp), len(obs))

def test_func3():
    exp = -1
    obs = basics.func3("Waka Jawaka", ["Vows are apoken"], 42)
    assert_equal(exp, obs)

def test_setfunc():
    pyset = basics.setfunc(1,2,3)
    assert_equal(pyset, set((1,2,3)))
    
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

def test_tclass1int():
    x = basics.TClass1Int()
    y = basics.TClass1[int]()
    z = basics.TClass1['int32']()

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

def test_untemplated():
    x = basics.Untemplated()
    exp = 42
    obs = x.untemplated_method(65.0)
    assert_equal(exp, obs)

def test_silly_bool_min():
    assert_true(basics.silly_bool_min(True, 1))
    assert_false(basics.silly_bool_min(0, False))
    assert_false(basics.silly_bool_min(True, False))
    assert_false(basics.silly_bool_min(1, False))

def test_a():
    x = basics.A(5)
    x.a = 10
    assert_equal(x.a, 10)
    x.a = 42
    assert_equal(x.a, 42)
    x.call()
    assert_equal(x.a, 1)

def test_b():
    x = basics.A()
    x.a = 10
    y = basics.B()
    y.b = 11
    assert_equal(y.b, 11)
    y.b = 43
    assert_equal(y.b, 43)
    assert_equal(y.clist[2][4], 8)
    y.call()
    assert_equal(y.b, 1)
    y.from_a(x)
    assert_equal(y.b, 10)
    assert_true(isinstance(y, basics.A))
    
def test_c():
    x = basics.A()
    x.a = 10
    y = basics.B()
    y.b = 11
    z = basics.C()
    z.c = 12
    assert_equal(z.c, 12)
    z.c = 44
    assert_equal(z.c, 44)
    z.call()
    assert_equal(z.c, 1)
    z.from_a(x)
    assert_equal(z.b, 10)
    assert_true(isinstance(z, basics.A))
    assert_true(isinstance(z, basics.B))

def test_findmin_double_float():
    exp = 42.0
    obs = basics.findmin_int_float(65.0, 42.0)
    assert_equal(exp, obs)

    exp = 42.0
    obs = basics.findmin[float, float](42.0, 65.0)
    assert_equal(exp, obs)

    exp = 42.0
    obs = basics.findmin['float64', 'float32'](42.0, 65.0)
    assert_equal(exp, obs)


def test_tclass0int():
    for x in [basics.TClass0Int(), basics.TClass0[int](), basics.TClass0['int32']()]:
        assert_equal(x.whatstheanswer_int(65), 42)
        # FIXME
        #assert_equal(x.whatstheanswer[int](65), 42)
        #assert_equal(x.whatstheanswer['int32'](65), 42)
        assert_equal(x.whatstheanswer_float(65.0), 42.0)
        #assert_equal(x.whatstheanswer[float](65.0), 42.0)
        #assert_equal(x.whatstheanswer['float32'](65.0), 42.0)

def test_tclass0double():
    for x in [basics.TClass0Double(), basics.TClass0[float](), 
              basics.TClass0['float64']()]:
        pass

def test_lessthan_int_3():
    assert_true(basics.lessthan_int_3(-1))
    assert_true(basics.lessthan_int_3(2))
    assert_false(basics.lessthan_int_3(42))
    assert_true(basics.lessthan[int, 3](-1))
    assert_true(basics.lessthan[int, 3](2))
    assert_false(basics.lessthan[int, 3](42))
    assert_true(basics.lessthan['int32', 3](-1))
    assert_true(basics.lessthan['int32', 3](2))
    assert_false(basics.lessthan['int32', 3](42))


def test_void_fp_struct():
    x = basics.VoidFPStruct()
    q = []
    x.op = q.append
    basics.call_with_void_fp_struct(x)
    assert_equal(q, [10])
