from __future__ import print_function
from xdress.utils import NotSpecified, RunControl

from nose.tools import assert_equal, with_setup, assert_false

def test_rc_make():
    rc = RunControl(a=NotSpecified, b="hello")
    assert_equal(rc.a, NotSpecified)
    assert_equal(rc.b, 'hello')
    assert_false(hasattr(rc, 'c'))
    
def test_rc_eq():
    rc = RunControl(a=NotSpecified, b="hello")
    d = {'a': NotSpecified, 'b': 'hello'}
    assert_equal(rc._dict, d)
    assert_equal(rc, RunControl(**d))

def test_rc_update():
    rc = RunControl(a=NotSpecified, b="hello")
    rc._update(RunControl(c=1, b="world"))
    assert_equal(rc, {'a': NotSpecified, 'b': 'world', 'c': 1})
    rc._update({'a': 42, 'c': NotSpecified})
    assert_equal(rc, {'a': 42, 'b': 'world', 'c': 1})

