"""Further tests the part of dtypes that is accessible from Python."""
from __future__ import print_function

import nose
from nose.tools import assert_equal, assert_not_equal, assert_raises, raises, \
    assert_almost_equal, assert_true, assert_false, assert_in

from numpy.testing import assert_array_equal, assert_array_almost_equal

import os
import numpy  as np

from cppproj import dt
from cppproj import stlc


# dtype set<int>
def test_dtype_set_int():
    a = np.array([set([1, 42, -65, 18])], dtype=dt.xd_set_int)
    a[:] = set([1818, -6565, 4242, 11])
    a = np.array([set([1, -65]), set([42, 18])], dtype=dt.xd_set_int)
    b = np.array([set([1, -65]), set([42, 18])], dtype=dt.xd_set_int)
    a[:2] = b[-2:]

    x = np.array([set([16, 42])], dtype=dt.xd_set_int)
    s = x[0]
    s.add(10)
    x[0] = s

# dtype map<str, int>
def test_dtype_map_str_int():
    x = np.array([{"hello": 42}], dtype=dt.xd_map_str_int)
    m = x[0]
    m['world'] = 10 
    x[0] = m

if __name__ == '__main__':
    nose.run()
