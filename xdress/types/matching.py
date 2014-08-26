import sys

from xdress.utils import flatten

if sys.version_info[0] > 2:
    basestring = str

class MatchAny(object):
    """A singleton helper class for matching any portion of a type."""
    def __repr__(self):
        return "MatchAny"

    def __hash__(self):
        # give consistent hash value across executions
        return hash(repr(self))

MatchAny = MatchAny()


def matches(pattern, t):
    """Indicates whether a type t matches a pattern. See TypeMatcher for more
    details.
    """
    tm = TypeMatcher(pattern)
    return tm.matches(t)


class TypeMatcher(object):
    """A class that is used for checking whether a type matches a given
    pattern.
    """

    def __init__(self, pattern):
        """Parameters
        ----------
        pattern : nested tuples, str, int
            This is a type-like entity that may have certain elements replaced
            by MatchAny, indicating that any value in a concrete type may match
            this pattern.  For example ('float64', MatchAny) will match the
            following::

                ('float64', 0)
                ('float64', '*')
                ('float64', '&')
                ('float64', 'const')

            but will not match::

                'float64'
                (('float64', 'const'), '&')

        """
        self._pattern = pattern

    @property
    def pattern(self):
        """The pattern to match."""
        # Make this field read-only to prevent hashing errors
        return self._pattern

    def __hash__(self):
        # needed so that class can be dict key
        return hash(self.pattern)

    def matches(self, t):
        """Tests that a type matches the pattern, returns True or False."""
        pattern = self.pattern
        if pattern is MatchAny:
            return True
        if t is pattern:
            return True
        if pattern is None:
            return False
        if t == pattern:
            return True
        if isinstance(pattern, basestring):
            #return t == pattern if isinstance(t, basestring) else False
            return False
        if isinstance(t, basestring) or isinstance(t, bool) or \
           isinstance(t, int) or isinstance(t, float):
            return False
        # now we know both pattern and t should be different non-string
        # sequences, nominally tuples or lists
        if len(t) != len(pattern):
            return False
        submatcher = TypeMatcher(None)
        for subt, subpattern in zip(t, pattern):
            submatcher._pattern = subpattern
            if not submatcher.matches(subt):
                return False
        return True

    def flatmatches(self, t):
        """Flattens t and then sees if any part of it matches self.pattern.
        """
        try:
            # See if user gave entire type
            if self.matches(t):
                return True
        except TypeError:
            # This might fail, if it does just move on
            pass

        else:
            if isinstance(t, basestring):
                return self.matches(t)
            elif isinstance(t, (tuple, list)):
                return any([self.matches(i) for i in flatten(t)])

    def __eq__(self, other):
        if isinstance(other, TypeMatcher):
            return self._pattern == other._pattern
        else:
            return self._pattern == other

    def __str__(self):
        return "{0}({1!s})".format(self.__class__.__name__, self._pattern)

    def __repr__(self):
        return "{0}({1!r})".format(self.__class__.__name__, self._pattern)
