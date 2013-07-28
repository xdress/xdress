"""Plugin for xdress which converts target names to PEP8 compatible versions.

This module is available as an xdress plugin by the name ``xdress.pep8names``.
It should probably come after ``xdress.autoall`` and after ``xdress.autodescribe``.

:author: Anthony Scopatz <scopatz@gmail.com>

PEP-8 Names Plugin API
======================
"""
from __future__ import print_function
import os
import re
import sys

from .utils import RunControl, NotSpecified, apiname, ensure_apiname
from .plugins import Plugin

if sys.version_info[0] >= 3:
    basestring = str

upperfirst = lambda s: s if 0 == len(s) else s[0].upper() + s[1:]

HAS_UPPER_RE = re.compile('([A-Z])')
HAS_UNDERSCORE_RE = re.compile('(_)')
UPPER_DIGIT_RE = re.compile('([A-Z0-9]+)')
WORD_RE = re.compile('(\w+)')
UNDERSCORE_WORD_RE = re.compile('(_\w+)')
UNDERSCORE_DIGIT_RE = re.compile('(_\d+)')

def _upperrepl(m):
    g1low = m.group(1).lower()
    g1start = m.start(1)
    if 0 == g1start:
        return g1low
    else:
        return '_' + g1low

def _underscorerepl(m):
    g1 = m.group(1)
    g1start = m.start(1)
    if 0 == g1start:
        g1s = g1.split('_')
        g1snew = []
        seennonempty = False
        for s in g1s:
            slen = len(s)
            if 0 == slen:
                if seennonempty:
                    continue
                else:
                    s = '_'
            else:
                seennonempty = True
            g1snew.append(s)
        print(g1snew)
        return ''.join(map(upperfirst, g1snew))
    else:
        return upperfirst(g1.replace('_', ''))

def _usdgrepl(m):
    g1 = m.group(1)
    return g1.replace('_', '')

def pep8func(name):
    """Converts a name, which may not be PEP-8 compliant for functions, 
    methods, or variables, to a name that is compliant.
    """
    if HAS_UPPER_RE.search(name) is not None:
        name = UPPER_DIGIT_RE.sub(_upperrepl, name)
    if UNDERSCORE_DIGIT_RE.search(name[1:]) is not None:
        name = name[0] + UNDERSCORE_DIGIT_RE.sub(_usdgrepl, name[1:])
    return name

pep8var = pep8func

def pep8class(name):
    """Converts a name, which may not be PEP-8 compliant for classes
    to a name that is compliant.
    """
    name = upperfirst(name)
    if HAS_UNDERSCORE_RE.search(name) is not None:
        new = UNDERSCORE_WORD_RE.sub(_underscorerepl, name)
        while new.startswith('_') and not name.startswith('_'):
            new = new[1:]
        name = new
    return name

def ensure_pep8name(name, kind):
    """Ensures that an apiname is valid for its kind: 'var', 'func', or 'class'.
    """
    name = ensure_apiname(name)
    tarname = name.tarname
    old = tarname if isinstance(tarname, basestring) else tarname[0]
    if kind == 'class':
        new = pep8class(old)
    elif kind == 'func':
        new = pep8func(old)
    elif kind == 'var':
        new = pep8var(old)
    else:
        raise ValueError('kind {0!r} not recognized'.format(kind))
    if old != new:
        new = new if isinstance(tarname, basestring) else (new,) + tarname[1:]
        name = name._replace(tarname=new)
    return name

class XDressPlugin(Plugin):
    """This class provides PEP-8 naming functionality for xdress."""

    def setup(self, rc):
        for i, var in enumerate(rc.variables):
            rc.variables[i] = ensure_pep8name(var, 'var')
        for i, fnc in enumerate(rc.functions):
            rc.functions[i] = ensure_pep8name(fnc, 'func')
        for i, cls in enumerate(rc.classes):
            rc.classes[i] = ensure_pep8name(cls, 'class')
