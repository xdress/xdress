from collections import Sequence, MutableMapping
from pprint import pformat
import sys

if sys.version_info[0] > 2:
    basestring = str

from .matching import TypeMatcher

_ensuremod = lambda x: x if x is not None and 0 < len(x) else ''
_ensuremoddot = lambda x: x + '.' if x is not None and 0 < len(x) else ''


def _recurse_replace(x, a, b):
    if isinstance(x, basestring):
        return x.replace(a, b)
    elif isinstance(x, Sequence):
        return tuple([_recurse_replace(y, a, b) for y in x])
    else:
        return x


class _LazyConfigDict(MutableMapping):
    def __init__(self, items, ts):
        self._d = items if isinstance(items, MutableMapping) else dict(items)
        self._ts = ts

    def __len__(self):
        return len(self._d)

    def __contains__(self, key):
        return key in self._d

    def __iter__(self):
        for k in self._d:
            yield k

    def __getitem__(self, key):
        value = self._d[key]
        kw = {'extra_types': _ensuremoddot(self._ts.extra_types),
              'dtypes': _ensuremoddot(self._ts.dtypes),
              'stlcontainers': _ensuremoddot(self._ts.stlcontainers), }
        for k, v in kw.items():
            value = _recurse_replace(value, '{' + k + '}', v)
        return value

    def __setitem__(self, key, value):
        self._d[key] = value

    def __delitem__(self, key):
        del self._d[key]

    def update(self, *args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            toup = args[0]
            if isinstance(toup, _LazyConfigDict):
                toup = toup._d
        elif len(args) == 0:
            toup = kwargs
        else:
            raise TypeError("invalid update signature.")
        self._d.update(toup)

    def __str__(self):
        return pformat(self._d)

    def __repr__(self):
        return self.__class__.__name__ + "(" + repr(self._d) + ", TypeSystem())"

class _LazyImportDict(MutableMapping):
    def __init__(self, items, ts):
        self._d = items if isinstance(items, MutableMapping) else dict(items)
        self._ts = ts

    def __len__(self):
        return len(self._d)

    def __contains__(self, key):
        return key in self._d

    def __iter__(self):
        for k in self._d:
            yield k

    def __getitem__(self, key):
        value = self._d[key]
        if callable(value):
            return value
        kw = {'extra_types': _ensuremod(self._ts.extra_types),
              'dtypes': _ensuremod(self._ts.dtypes),
              'stlcontainers': _ensuremod(self._ts.stlcontainers),}
        newvalue = tuple(tuple(x.format(**kw) or None for x in imp if x is not None) \
                            for imp in value if imp is not None) or (None,)
        return newvalue

    def __setitem__(self, key, value):
        self._d[key] = value

    def __delitem__(self, key):
        del self._d[key]

    def update(self, *args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            toup = args[0]
            if isinstance(toup, _LazyImportDict):
                toup = toup._d
        elif len(args) == 0:
            toup = kwargs
        else:
            raise TypeError("invalid update signature.")
        self._d.update(toup)

    def __str__(self):
        return pformat(self._d)

    def __repr__(self):
        return self.__class__.__name__ + "(" + repr(self._d) + ", TypeSystem())"

class _LazyConverterDict(MutableMapping):
    def __init__(self, items, ts):
        self._d = items if isinstance(items, MutableMapping) else dict(items)
        self._tms = set([k for k in self._d if isinstance(k, TypeMatcher)])
        self._ts = ts

    def __len__(self):
        return len(self._d)

    def __contains__(self, key):
        if key in self._d:
            return True  # Check if key is present
        else:
            # check if any TypeMatcher keys actually match
            for tm in self._tms:
                if tm.matches(key):
                    self[key] = self._d[tm]
                    return True
            else:
                return False

    def __iter__(self):
        for k in self._d:
            yield k

    def __getitem__(self, key):
        if key in self._d:
            value = self._d[key]  # Check if key is present
        else:
            # check if any TypeMatcher keys actually match
            for tm in self._tms:
                if tm.matches(key):
                    value = self._d[tm]
                    self[key] = value
                    break
            else:
                raise KeyError("{0} not found".format(key))
        if value is None or value is NotImplemented or callable(value):
            return value
        kw = {'extra_types': _ensuremoddot(self._ts.extra_types),
              'dtypes': _ensuremoddot(self._ts.dtypes),
              'stlcontainers': _ensuremoddot(self._ts.stlcontainers),}
        newvalue = []
        for x in value:
            newx = x
            if isinstance(newx, basestring):
                for k, v in kw.items():
                    newx = newx.replace('{' + k + '}', v)
            newvalue.append(newx)
        return tuple(newvalue)

    def __setitem__(self, key, value):
        self._d[key] = value
        if isinstance(key, TypeMatcher):
            self._tms.add(key)

    def __delitem__(self, key):
        del self._d[key]
        if isinstance(key, TypeMatcher):
            self._tms.remove(key)

    def update(self, *args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            toup = args[0]
            if isinstance(toup, _LazyConverterDict):
                toup = toup._d
        elif len(args) == 0:
            toup = kwargs
        else:
            raise TypeError("invalid update signature.")
        self._d.update(toup)
        self._tms.update([k for k in toup if isinstance(k, TypeMatcher)])

    def __str__(self):
        return pformat(self._d)

    def __repr__(self):
        return self.__class__.__name__ + "(" + repr(self._d) + ", TypeSystem())"
