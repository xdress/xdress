"""Further tests the part of dtypes that is accessible from Python."""
from __future__ import print_function

import nose
from nose.tools import assert_equal, assert_not_equal, assert_raises, raises, \
    assert_almost_equal, assert_true, assert_false, assert_in

from numpy.testing import assert_array_equal, assert_array_almost_equal

import os
import numpy  as np

from cppproj import dt




# dtype set<int>
def test_dtype_set_int():
    a = np.array([set([1, 42, -65, 18])], dtype=dt.xd_set_int)
    #a[:] = [18, -65, 42, 1]
    #a = np.array([1, -65, 1, -65] + [42, 18, 42, 18], dtype=dt.xd_int)
    #b =  np.array(([1, -65, 1, -65] + [42, 18, 42, 18])[::2], dtype=dt.xd_int)
    #a[:2] = b[-2:]
    print(a)
    print(a[0])
    print(type(a[0]))
    assert False


if __name__ == '__main__':
    nose.run()
