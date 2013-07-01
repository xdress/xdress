"""Helper functions for bright API generation.

:author: Anthony Scopatz <scopatz@gmail.com>

Utilities API
=============
"""
from __future__ import print_function
import os
import io
import sys
from copy import deepcopy
from pprint import pformat
from collections import Mapping
from hashlib import md5
try:
    import cPickle as pickle
except ImportError:
    import pickle

if sys.version_info[0] >= 3: 
    basestring = str

DEFAULT_RC_FILE = "xdressrc.py"
"""Default run control file name."""

DEFAULT_PLUGINS = ('xdress.stlwrap', 'xdress.autoall', 'xdress.cythongen')
"""Default list of plugin module names."""

FORBIDDEN_NAMES = frozenset(['del'])

def indent(s, n=4, join=True):
    """Indents all lines in the string or list s by n spaces."""
    spaces = " " * n
    lines = s.splitlines() if isinstance(s, basestring) else s
    if join:
        return '\n'.join([spaces + l for l in lines])
    else:
        return [spaces + l for l in lines]


class indentstr(str):
    """A special string subclass that can be used to indent the whol string 
    inside of format strings by accessing an ``indentN`` attr.  For example,
    ``s.indent8`` will return a copy of the string s where every line starts 
    with 8 spaces."""
    def __getattr__(self, key):
        if key.startswith('indent'):
            return indent(self, n=int(key[6:]))
        return getattr(super(indentstr, self), key)


def expand_default_args(methods):
    """This function takes a collection of method tuples and expands all of 
    the default arguments, returning a set of all methods possible."""
    methitems = set()
    for mkey, mrtn in methods:
        mname, margs = mkey[0], mkey[1:]
        havedefaults = [3 == len(arg) for arg in margs]
        if any(havedefaults):
            # expand default arguments  
            n = havedefaults.index(True)
            items = [((mname,)+tuple(margs[:n]), mrtn)] + \
                    [((mname,)+tuple(margs[:i]), mrtn) for i in range(n+1,len(margs)+1)]
            methitems.update(items)
        else:
            # no default args
            methitems.add((mkey, mrtn))
    return methitems


def newoverwrite(s, filename, verbose=False):
    """Useful for not forcing re-compiles and thus playing nicely with the 
    build system.  This is acomplished by not writing the file if the existsing
    contents are exactly the same as what would be written out.

    Parameters
    ----------
    s : str
        string contents of file to possible
    filename : str
        Path to file.
    vebose : bool, optional
        prints extra message

    """
    if os.path.isfile(filename):
        with io.open(filename, 'rb') as f:
            old = f.read()
        if s == old:
            return
    with io.open(filename, 'wb') as f:
        f.write(s.encode())
    if verbose:
        print("  wrote " + filename)

def newcopyover(f1, f2, verbose=False):
    """Useful for not forcing re-compiles and thus playing nicely with the 
    build system.  This is acomplished by not writing the file if the existsing
    contents are exactly the same as what would be written out.

    Parameters
    ----------
    f1 : str
        Path to file to copy from
    f2 : str
        Path to file to copy over
    vebose : bool, optional
        prints extra message

    """
    if os.path.isfile(f1):
        with io.open(f1, 'r') as f:
            s = f.read()
        return newoverwrite(s, f2, verbose)

def writenewonly(s, filename, verbose=False):
    """Only writes the contents of the string to a file if the file does not exist.
    Useful for not tocuhing files. 

    Parameters
    ----------
    s : str
        string contents of file to possible
    filename : str
        Path to file.
    vebose : bool, optional
        prints extra message

    """
    if os.path.isfile(filename):
        return
    with open(filename, 'w') as f:
        f.write(str(s))
    if verbose:
        print("  wrote " + filename)

def ensuredirs(f):
    """For a file path, ensure that its directory path exists."""
    d = os.path.split(f)[0]
    if not os.path.isdir(d):
        os.makedirs(d)


def isvardesc(desc):
    """Tests if a description is a variable-type description."""
    return desc is not None and 'type' in desc 

def isfuncdesc(desc):
    """Tests if a description is a function-type description."""
    return desc is not None and 'signatures' in desc

def isclassdesc(desc):
    """Tests if a description is a class-type description."""
    return desc is not None and 'parents' in desc

def exec_file(filename, glb=None, loc=None):
    """A function equivalent to the Python 2.x execfile statement."""
    with io.open(filename, 'r') as f:
        src = f.read()
    exec(compile(src, filename, "exec"), glb, loc)

class NotSpecified(object):
    """A helper class singleton for run control meaning that a 'real' value 
    has not been given."""
    def __repr__(self):
        return "NotSpecified"

NotSpecified = NotSpecified()
"""A helper class singleton for run control meaning that a 'real' value 
has not been given."""

class RunControl(object):
    """A composable configuration class for xdress. Unlike argparse.Namespace, 
    this keeps the object dictionary (__dict__) separate from the run control 
    attributes dictionary (_dict)."""

    def __init__(self, **kwargs):
        """Parameters
        -------------
        kwargs : optional
            Items to place into run control.

        """
        self._dict = {}
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getattr__(self, key):
        if key in self._dict:
            return self._dict[key]
        elif key in self.__dict__:
            return self.__dict__[key]
        else:
            return self.__class__.__dict__[key]

    def __setattr__(self, key, value):
        if key.startswith('_'):
            self.__dict__[key] = value
        else:
            if value is NotSpecified and key in self:
                return
            self._dict[key] = value

    def __delattr__(self, key):
        if key in self._dict:
            del self._dict[key]
        elif key in self.__dict__:
            del self.__dict__[key]
        else:
            del self.__class__.__dict__[key]

    def __iter__(self):
        return iter(self._dict)

    def __repr__(self):
        keys = sorted(self._dict.keys())
        s = ", ".join(["{0!s}={1!r}".format(k, self._dict[k]) for k in keys])
        return "{0}({1})".format(self.__class__.__name__, s)

    def _pformat(self):
        keys = sorted(self._dict.keys())
        f = lambda k: "{0!s}={1}".format(k, pformat(self._dict[k], indent=2))
        s = ",\n ".join(map(f, keys))
        return "{0}({1})".format(self.__class__.__name__, s)

    def __contains__(self, key):
        return key in self._dict or key in self.__dict__ or \
                                    key in self.__class__.__dict__

    def __eq__(self, other):
        if hasattr(other, '_dict'):
            return self._dict == other._dict
        elif isinstance(other, Mapping):
            return self._dict == other
        else:
            return NotImplemented

    def __ne__(self, other):
        if hasattr(other, '_dict'):
            return self._dict != other._dict
        elif isinstance(other, Mapping):
            return self._dict != other
        else:
            return NotImplemented

    _update_as_list = set(['includes'])

    def _update(self, other):
        if hasattr(other, '_dict'):
            other = other._dict
        elif not hasattr(other, 'items'):
            other = dict(other)
        for k, v in other.items():
            if v is NotSpecified:
                pass
            elif k in self._update_as_list and k in self:
                v = list(v) + list(getattr(self, k))
            setattr(self, k, v)

_lang_exts = {
    'c': 'c',
    'c++': 'cpp',
    'f': 'f77',
    'fortran': 'f77',
    'f77': 'f77',
    'f90': 'f90',
    'python': 'py',
    'cython': 'pyx',
    }

_head_exts = {
    'c': 'h',
    'c++': 'hpp',
    'f': 'h',
    'fortran': 'h',
    'f77': 'h',
    'f90': 'h',
    'python': None,
    'cython': 'pxd',
    }

_exts_lang = {
    'c': 'c',
    'h': 'c',
    'cpp': 'c++',
    'hpp': 'c++',
    'cxx': 'c++',
    'hxx': 'c++',
    'c++': 'c++',
    'h++': 'c++',
    'f': 'f77',
    'f77': 'f77',
    'f90': 'f90',
    'py': 'python',
    'pyx': 'cython',
    'pxd': 'cython',
    'pxi': 'cython',
    }

_hdr_exts = frozenset(['h', 'hpp', 'hxx', 'h++', 'pxd'])
_src_exts = frozenset(['c', 'cpp', 'cxx', 'c++', 'f', 'f77', 'f90', 'py', 'pyx'])

def guess_language(filename, default='c++'):
    """Try to guess a files' language from its extention, defaults to C++."""
    ext = filename.rsplit('.', 1)[-1].lower()
    lang = _exts_lang.get(ext, default)
    return lang

def find_source(basename, sourcedir='.'):
    """Finds a source filename, header filename, language name, and language
    source extension given a basename and source directory."""
    files = os.listdir(sourcedir)
    files = [f for f in files if f.startswith(basename + '.')]
    langs = dict([(f, guess_language(f, None)) for f in files])
    lang = src = hdr = srcext = None
    for f, l in langs.items():
        ext = f.rsplit('.', 1)[-1]
        if ext in _hdr_exts:
            hdr = f
        elif ext in _src_exts and (src is None or l != 'python'):
            lang = l
            src = f
            srcext = ext
    if src is None and hdr is not None:
        src = hdr
        lang = langs[hdr]
        srcext = _lang_exts[lang]
    return src, hdr, lang, srcext

nyansep = r'~\_/' * 17 + '~=[,,_,,]:3'
"""WAT?!"""

class DescriptionCache(object):
    """A quick persistent cache for descriptions from files.  
    The keys are (classname, filename, kind) tuples.  The values are 
    (hashes-of-the-file, description-dictionary) tuples."""

    def __init__(self, cachefile=os.path.join('build', 'desc.cache')):
        """Parameters
        -------------
        cachefile : str, optional
            Path to description cachefile.

        """
        self.cachefile = cachefile
        if os.path.isfile(cachefile):
            with io.open(cachefile, 'rb') as f:
                self.cache = pickle.load(f)
        else:
            self.cache = {}

    def isvalid(self, name, filename, kind):
        """Boolean on whether the cach value for a (name, filename, kind)
        tuple matches the state of the file on the system."""
        key = (name, filename, kind)
        if key not in self.cache:
            return False
        cachehash = self.cache[key][0]
        with io.open(filename, 'rb') as f:
            filebytes = f.read()
        currhash = md5(filebytes).hexdigest()
        return cachehash == currhash

    def __getitem__(self, key):
        return self.cache[key][1]  # return the description only

    def __setitem__(self, key, value):
        name, filename, kind = key
        with io.open(filename, 'rb') as f:
            filebytes = f.read()
        currhash = md5(filebytes).hexdigest()
        self.cache[key] = (currhash, value)

    def __delitem__(self, key):
        del self.cache[key]

    def dump(self):
        """Writes the cache out to the filesystem."""
        if not os.path.exists(self.cachefile):
            pardir = os.path.split(self.cachefile)[0]
            if not os.path.exists(pardir):
                os.makedirs(pardir)
        with io.open(self.cachefile, 'wb') as f:
            pickle.dump(self.cache, f, pickle.HIGHEST_PROTOCOL)

    def __str__(self):
        return pformat(self.cache)

def merge_descriptions(descriptions):
    """Given a sequence of descriptions, in order of increasing precedence, 
    merge them into a single description dictionary."""
    attrsmeths = frozenset(['attrs', 'methods', 'signatures'])
    desc = {}
    for description in descriptions:
        for key, value in description.items():
            if key not in desc:
                desc[key] = deepcopy(value)
                continue

            if key in attrsmeths:
                desc[key].update(value)
            elif key == 'docstrings':
                for dockey, docvalue in value.items():
                    if dockey in attrsmeths:
                        desc[key][dockey].update(docvalue)
                    else:
                        desc[key][dockey] = deepcopy(docvalue)
            else:
                desc[key] = deepcopy(value)
    # now sanitize methods
    name = desc['name']
    methods = desc.get('methods', {})
    for methkey, methval in methods.items():
        if methval is None and not methkey[0].endswith(name):
            del methods[methkey]  # constructor for parent
    return desc

