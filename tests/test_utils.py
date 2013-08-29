from __future__ import print_function
from xdress.utils import NotSpecified, RunControl, flatten, split_template_args, \
    ishashable, memoize, memoize_method, apiname, ensure_apiname, sortedbytype, \
    cppint

from nose.tools import assert_equal, with_setup, assert_true, assert_false, \
    assert_not_equal
from tools import unit

@unit
def test_rc_make():
    rc = RunControl(a=NotSpecified, b="hello")
    assert_equal(rc.a, NotSpecified)
    assert_equal(rc.b, 'hello')
    assert_false(hasattr(rc, 'c'))
    
@unit
def test_rc_eq():
    rc = RunControl(a=NotSpecified, b="hello")
    d = {'a': NotSpecified, 'b': 'hello'}
    assert_equal(rc._dict, d)
    assert_equal(rc, RunControl(**d))

@unit
def test_rc_update():
    rc = RunControl(a=NotSpecified, b="hello")
    rc._update(RunControl(c=1, b="world"))
    assert_equal(rc, {'a': NotSpecified, 'b': 'world', 'c': 1})
    rc._update({'a': 42, 'c': NotSpecified})
    assert_equal(rc, {'a': 42, 'b': 'world', 'c': 1})

@unit
def test_flatten():
    exp = ["hello", None, 1, 3, 2, 5, 6]
    obs = [x for x in flatten(["hello", None, (1, 3, (2, 5, 6))])]
    assert_equal(exp, obs)

def check_split_template_args(s, exp):
    obs = split_template_args(s)
    assert_equal(exp, obs)

@unit
def test_split_template_args():
    cases = [
        ('set<int>', ['int']),
        ('map<int, double>', ['int', 'double']),
        ('map<int, set<int> >', ['int', 'set<int>']),
        ('map< int, set<int> >', ['int', 'set<int>']),
        ('map< int, vector<set<int> > >', ['int', 'vector<set<int> >']),
        ]

    for s, exp in cases:
        yield check_split_template_args, s, exp

def check_ishashable(assertion, x):
    assertion(ishashable(x))

@unit
def test_ishashable():
    cases = [
        (assert_true, 'hello'),
        (assert_true, 1),
        (assert_true, ()),
        (assert_true, 1.34e-5),
        (assert_true, frozenset()),
        (assert_true, (1, 3, ((10,42),))),
        (assert_false, []),
        (assert_false, {}),
        (assert_false, set()),
        (assert_false, (1, 3, ({10:42},))),
        ]
    for assertion, x in cases:
        yield check_ishashable, assertion, x

@unit
def test_memoize():
    global z
    z = 1
    
    @memoize
    def inc(x):
        global z
        z += 1
        return x + 1

    assert_equal(inc.cache, {})
    inc(2)
    assert_equal(inc.cache, {((2,), ()): 3})
    assert_equal(z, 2)
    inc(5)
    assert_equal(inc.cache, {((2,), ()): 3, ((5,), ()): 6,})
    assert_equal(z, 3)
    inc(2)
    assert_equal(inc.cache, {((2,), ()): 3, ((5,), ()): 6,})
    assert_equal(z, 3)
    del z

@unit
def test_memoize_method():
    class Joan(object):

        def __init__(self, v=0):
            self.v = v
            self.call_count = 0

        @memoize_method
        def inc(self, arg):
            """I am inc's docstr"""
            self.call_count += 1
            self.v += arg
            return self.v

    j = Joan()
    assert_equal(j.inc(2), 2)
    assert_equal(j.inc(2), j.inc(2))
    assert_equal(j.call_count, 1)
    assert_not_equal(Joan.inc(j, 2), Joan.inc(j, 2))
    assert_equal(j.inc.__name__, "inc")
    assert_equal(j.inc.__doc__, "I am inc's docstr")

def check_ensure_apiname(x, exp):
    obs = ensure_apiname(x)
    assert_equal(exp, obs)

@unit
def test_ensure_apiname():
    cases = [
        (('Joan', 'joan'), apiname('Joan', 'joan', 'joan', 'Joan')), 
        (('Joan', 'joan', 'pyjoan'), apiname('Joan', 'joan', 'pyjoan', 'Joan')), 
        (('Joan', 'joan', 'pyjoan', 'PyJoan'), 
            apiname('Joan', 'joan', 'pyjoan', 'PyJoan')), 
        (['Joan', 'joan', 'pyjoan', 'PyJoan'], 
            apiname('Joan', 'joan', 'pyjoan', 'PyJoan')), 
        ({'srcname': 'Joan', 'srcfile': 'joan'}, 
            apiname('Joan', 'joan', 'joan', 'Joan')), 
        (apiname('Joan', 'joan', 'pyjoan', 'PyJoan'), 
            apiname('Joan', 'joan', 'pyjoan', 'PyJoan')), 
        (apiname('Joan', 'joan', 'pyjoan', NotSpecified), 
            apiname('Joan', 'joan', 'pyjoan', 'Joan')), 
        ]
    for x, exp in cases:
        yield check_ensure_apiname, x, exp

def check_sortedbytype(x, exp):
    obs = sortedbytype(x)
    assert_equal(exp, obs)

@unit
def test_sortedbytype():
    cases = [
        (['a', (1, 2), 'b'], ['a', 'b', (1, 2)]),
        (['a', 1, (1, 2), 10.0, 'b'], [10.0, 1, 'a', 'b', (1, 2)]),
        ]
    for x, exp in cases:
        yield check_sortedbytype, x, exp

def check_cppint(s, exp):
    obs = cppint(s)
    assert_equal(exp, obs)

@unit
def test_cppint():
    cases = [
        ('42', 42),
        ('+42LLu', 42),
        ('-0', 0),
        ('0o52', 42),
        ('-052', -42),
        ('0B101010', 42),
        ('-0x2A', -42),
        ]
    for s, x in cases:
        yield check_cppint, s, x
