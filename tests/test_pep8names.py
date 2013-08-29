from __future__ import print_function
from xdress.utils import apiname, ensure_apiname
from xdress.pep8names import pep8func, pep8class

from nose.tools import assert_equal, with_setup, assert_true, assert_false, \
    assert_not_equal
from tools import unit

@unit
def check_cases(f, x, exp):
    obs = f(x)
    assert_equal(exp, obs)

@unit
def test_pep8func():
    cases = [
        ('joan', 'joan'),
        ('Joan', 'joan'),
        ('joanHoover', 'joan_hoover'),
        ('JoanHoover', 'joan_hoover'),
        ('Joan2Hoover', 'joan2hoover'),
        ('JoanHoover3', 'joan_hoover3'),
        ('JOAN2Hoover', 'joan2hoover'),
        ]
    for s, exp in cases:
        yield check_cases, pep8func, s, exp

@unit
def test_pep8class():
    cases = [
        ('joan', 'Joan'),
        ('Joan', 'Joan'),
        ('joanHoover', 'JoanHoover'),
        ('JoanHoover', 'JoanHoover'),
        ('Joan2Hoover', 'Joan2Hoover'),
        ('JoanHoover3', 'JoanHoover3'),
        ('JOAN2Hoover', 'JOAN2Hoover'),
        ('joan_hoover', 'JoanHoover'),
        ('_Joan2Hoover', '_Joan2Hoover'),
        ('__JoanHoover3', '__JoanHoover3'),
        ('_joan_hoover', '_JoanHoover'),
        ('joan_hoover_3', 'JoanHoover3'),
        ]
    for s, exp in cases:
        yield check_cases, pep8class, s, exp
