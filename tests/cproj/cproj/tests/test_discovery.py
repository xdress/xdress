from nose.tools import assert_equal, assert_true
from numpy.testing import assert_array_equal, assert_array_almost_equal

from cproj import discovery

def test_face2face():
    obs = discovery.face2face()
    assert_true(obs is None)
    
def test_discovery():
    assert_equal(discovery.HARDER, 0)
    assert_equal(discovery.BETTER, 1)
    assert_equal(discovery.FASTER, 2)
    assert_equal(discovery.STRONGER, 3)

def test_crecendoll():
    x = discovery.Crescendoll()
