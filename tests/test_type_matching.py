from nose.tools import assert_equal

from xdress.types.matching import TypeMatcher, MatchAny, matches

from tools import unit


p1 = ('float64', MatchAny)

type_matcher_cases = [
    [p1, ('float64', 0), True],
    [p1, ('float64', '*'), True],
    [p1, ('float64', '&'), True],
    [p1, ('float64', 'const'), True],
    [p1, 'float64', False],
    [p1, ('f8', 0), False],
    [p1, (('float64', 'const'), '&'), False],
    ]


def check_typematcher(pattern, t, exp):
    tm = TypeMatcher(pattern)
    obs = tm.matches(t)
    assert_equal(exp, obs)


@unit
def test_typematcher():
    for pattern, t, exp in type_matcher_cases:
        yield check_typematcher, pattern, t, exp


def check_matches(pattern, t, exp):
    obs = matches(pattern, t)
    assert_equal(exp, obs)


@unit
def test_matches():
    for pattern, t, exp in type_matcher_cases:
        yield check_matches, pattern, t, exp
