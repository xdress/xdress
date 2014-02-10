"""Helpers for xdress.

:author: Anthony Scopatz <scopatz@gmail.com>

Utilities API
=============
"""
from __future__ import print_function
import os
import io
import re
import ast
import sys
import glob
import functools
from copy import deepcopy
from pprint import pformat
from collections import Mapping, Iterable, Hashable, Sequence, namedtuple
from hashlib import md5
from warnings import warn

try:
    import cPickle as pickle
except ImportError:
    import pickle

try:
    from enum import Enum, IntEnum
except ImportError:
    from ._enum import Enum, IntEnum

import numpy as np

if sys.version_info[0] >= 3:
    basestring = str

DEFAULT_RC_FILE = "xdressrc.py"
"""Default run control file name."""

DEFAULT_PLUGINS = ('xdress.autoall', 'xdress.cythongen', 'xdress.stlwrap')
"""Default list of plugin module names."""

FORBIDDEN_NAMES = frozenset(['del', 'global'])

def warn_forbidden_name(forname, inname=None, rename=None):
    """Warns the user that a forbidden name has been found."""
    msg = "found forbidden name {0!r}".format(forname)
    if inname is not None:
        msg += " in {0!r}".format(inname)
    if rename is not None:
        msg += ", renaming to {0!r}".format(rename)
    warn(msg, RuntimeWarning)

def indent(s, n=4, join=True):
    """Indents all lines in the string or list s by n spaces."""
    spaces = " " * n
    lines = s.splitlines() if isinstance(s, basestring) else s
    lines = lines or ()
    if join:
        return '\n'.join([spaces + l for l in lines if l is not None])
    else:
        return [spaces + l for l in lines if l is not None]


class indentstr(str):
    """A special string subclass that can be used to indent the whol string
    inside of format strings by accessing an ``indentN`` attr.  For example,
    ``s.indent8`` will return a copy of the string s where every line starts
    with 8 spaces."""
    def __getattr__(self, key):
        if key.startswith('indent'):
            return indent(self, n=int(key[6:]))
        return getattr(super(indentstr, self), key)


class Arg(IntEnum):
    NONE = 0
    TYPE = 1
    LIT = 2
    VAR = 3

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


def expand_default_args(methods):
    """This function takes a collection of method tuples and expands all of
    the default arguments, returning a set of all methods possible."""
    methitems = set()
    for mkey, mval in methods:
        mname, margs = mkey[0], mkey[1:]
        mrtn = mval['return']
        mdefargs = mval['defaults']
        havedefaults = [arg is not Arg.NONE for arg in margs]
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

_bool_literals = {'true': True, 'false': False}
_c_literal_int = re.compile(r'^([+-]?([1-9][0-9]*|0o?[0-7]+|0x[0-9a-f]+|'
                                   r'0b[01]+|0))u?(l{1,2})?u?$')
_c_int_bases = {'0x': 16, '0o': 8, '0b': 2, '0': 8,
                '00': 8, '01': 8, '02': 8, '03': 8, '04': 8, '05': 8, '06': 8, '07': 8}
_c_float = re.compile(r'^([+-]?([0-9]+\.[0-9]*|\.[0-9]+)(e[+-]?[0-9]+)?)([lf]?)$')
_c_float_types = {'': float, 'f': np.float32, 'l': np.longfloat}

def c_literal(s):
    """Convert a C/C++ literal to the corresponding Python value."""
    s = s.strip()
    try:
        # ast.literal_eval isn't precisely correct, since there are Python
        # literals which aren't valid C++, but it is close enough and faster
        # than the custom code below in the common case.
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        pass
    if s in _bool_literals:
        return _bool_literals[s]
    lo = s.lower()
    m = _c_literal_int.match(lo)
    if m:
        base = _c_int_bases.get(m.group(2)[:2], 10)
        return int(m.group(1), base)
    m = _c_float.match(lo)
    if m:
        return _c_float_types[m.group(4)](m.group(1))
    raise ValueError('unknown literal: {0!r}'.format(s))

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

def touch(filename):
    """Opens a file and updates the mtime, like the posix command of the same name."""
    with io.open(filename, 'a') as f:
        os.utime(filename, None)

def isvardesc(desc):
    """Tests if a description is a variable-type description."""
    return desc is not None and 'type' in desc and 'signatures' not in desc and \
           'methods' not in desc

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

#
# Run Control
#

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
        self._updaters = {}

    def __getattr__(self, key):
        if key in self._dict:
            return self._dict[key]
        elif key in self.__dict__:
            return self.__dict__[key]
        elif key in self.__class__.__dict__:
            return self.__class__.__dict__[key]
        else:
            msg = "RunControl object has no attribute {0!r}.".format(key)
            raise AttributeError(msg)

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
        elif key in self.__class__.__dict__:
            del self.__class__.__dict__[key]
        else:
            msg = "RunControl object has no attribute {0!r}.".format(key)
            raise AttributeError(msg)

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

    def _update(self, other):
        """Updates the rc with values from another mapping.  If this rc has
        if a key is in self, other, and self._updaters, then the updaters
        value is called to perform the update.  This function should return
        a copy to be safe and not update in-place.
        """
        if hasattr(other, '_dict'):
            other = other._dict
        elif not hasattr(other, 'items'):
            other = dict(other)
        for k, v in other.items():
            if v is NotSpecified:
                pass
            elif k in self._updaters and k in self:
                v = self._updaters[k](getattr(self, k), v)
            setattr(self, k, v)

def parse_global_rc():
    '''Search a global xdressrc file and parse if it exists.
    If nothing is found, an empty RunControl is returned.'''
    home = os.path.expanduser('~')
    globalrcs = [os.path.join(home, '.xdressrc'),
                 os.path.join(home, '.xdressrc.py'),
                 os.path.join(home, '.config', 'xdressrc'),
                 os.path.join(home, '.config', 'xdressrc.py'),
                 ]
    rc = RunControl()
    for globalrc in globalrcs:
        if not os.path.isfile(globalrc):
            continue
        globalrcdict = {}
        exec_file(globalrc, globalrcdict, globalrcdict)
        rc._update(globalrcdict)
    return rc

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
    'cc': 'c++',
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

def find_source(basename, sourcedir='.', language=None):
    """Finds a source filename, header filename, language name, and language
    source extension given a basename and source directory."""
    if isinstance(basename, list):
        files = [ os.path.abspath(x) for x in basename ]
    elif os.path.isabs(basename):
        files = [ basename ]
    else:
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
    if language is not None and language is not NotSpecified:
        lang = language
    return src, hdr, lang, srcext

def find_filenames(srcname, tarname=None, sourcedir='src', language=None):
    """Returns a description dictionary for a class or function
    implemented in a source file and bound into a target file.

    Parameters
    ----------
    srcname : str
        File basename of implementation.
    tarname : str, optional
        File basename where the bindings will be generated.
    srcdir : str, optional
        Source directory.

    Returns
    -------
    desc : dict
        Description dictionary containing only filenames.

    """
    desc = {}
    srcfname, hdrfname, lang, ext = find_source(srcname, sourcedir=sourcedir,
                                                language=language)
    desc['source_filename'] = srcfname
    desc['header_filename'] = hdrfname
    desc['language'] = lang
    desc['language_extension'] = ext
    desc['metadata_filename'] = '{0}.py'.format(srcname)
    if tarname is None:
        desc['pxd_filename'] = desc['pyx_filename'] = \
                               desc['srcpxd_filename'] = None
    else:
        desc['pxd_filename'] = '{0}.pxd'.format(tarname)
        desc['pyx_filename'] = '{0}.pyx'.format(tarname)
        desc['srcpxd_filename'] = '{0}_{1}.pxd'.format(ext, tarname)
    return desc

def extra_filenames(name):
    """Returns a diictionary of extra filenames for an API element
    implemented in a given language and bound into a target.

    Parameters
    ----------
    name : apiname
        The API element name.

    Returns
    -------
    extra : dict
        Extra filenames.

    """
    extra = {}
    ext = _lang_exts[name.language]
    tarbase = name.tarbase
    if tarbase is None:
        extra['pxd_filename'] = extra['pyx_filename'] = \
                                extra['srcpxd_filename'] = None
        srcbase = os.path.splitext(os.path.basename(name.srcfiles[0]))[0]
        extra['pxd_base'] = srcbase
        extra['cpppxd_base'] = '{0}_{1}'.format(ext, srcbase)
    else:
        extra['pxd_filename'] = '{0}.pxd'.format(tarbase)
        extra['pyx_filename'] = '{0}.pyx'.format(tarbase)
        extra['srcpxd_filename'] = '{0}_{1}.pxd'.format(ext, tarbase)
        extra['pxd_base'] = tarbase
        extra['cpppxd_base'] = '{0}_{1}'.format(ext, tarbase)
    return extra

def infer_format(filename, format):
    """Tries to figure out a file format."""
    if isinstance(format, basestring):
        pass
    elif filename.endswith('.pkl.gz'):
        format = 'pkl.gz'
    elif filename.endswith('.pkl'):
        format = 'pkl'
    else:
        raise ValueError("file format could not be determined.")
    return format

def sortedbytype(iterable):
    """Sorts an iterable by types first, then value."""
    items = {}
    for x in iterable:
        t = type(x).__name__
        if t not in items:
            items[t] = []
        items[t].append(x)
    rtn = []
    for t in sorted(items.keys()):
        rtn.extend(sorted(items[t]))
    return rtn

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

    def _hash_srcfiles(self, srcfiles):
        hashes = []
        for srcfile in srcfiles:
            with io.open(srcfile, 'rb') as f:
                filebytes = f.read()
            hashes.append(md5(filebytes).hexdigest())
        return tuple(hashes)

    def isvalid(self, name, kind):
        """Boolean on whether the cach value for a (apiname, kind)
        tuple matches the state of the file on the system."""
        key = tuple(name) + (kind,)
        if key not in self.cache:
            return False
        cachehashes = self.cache[key][0]
        currhashes = self._hash_srcfiles(name.srcfiles)
        return cachehashes == currhashes

    def __getitem__(self, key):
        if len(key) == 2 and isinstance(key[0], apiname):
            key = tuple(key[0]) + key[1:]
        return self.cache[key][1]  # return the description only

    def __setitem__(self, key, value):
        if len(key) == 2 and isinstance(key[0], apiname):
            name, kind = key
            key = tuple(key[0]) + key[1:]
        else:
            name, kind = apiname(*key[0]), key[1]
        currhashes = self._hash_srcfiles(name.srcfiles)
        self.cache[key] = (currhashes, value)

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
    name = desc['name']['srcname'] # srcname or tarname?
    methods = desc.get('methods', {})
    for methkey, methval in list(methods.items()):
        if methval is None:
            methname = methkey if isinstance(methkey, basestring) else methkey[0]
            if methname[0].endswith(name):
                del methods[methkey]
    return desc

def flatten(iterable):
    """Generator which returns flattened version of nested sequences."""
    for el in iterable:
        if isinstance(el, basestring):
            yield el
        elif isinstance(el, Iterable):
            for subel in flatten(el):
                yield subel
        else:
            yield el

def split_template_args(s, open_brace='<', close_brace='>', separator=','):
    """Takes a string with template specialization and returns a list
    of the argument values as strings."""
    targs = []
    ns = s.split(open_brace, 1)[-1].rsplit(close_brace, 1)[0].split(separator)
    count = 0
    targ_name = ''
    for n in ns:
        count += int(open_brace in n)
        count -= int(close_brace in n)
        targ_name += n
        if count == 0:
            targs.append(targ_name.strip())
            targ_name = ''
    return targs

def parse_template(s, open_brace='<', close_brace='>', separator=','):
    """Takes a string -- which may represent a template specialization --
    and returns the corresponding type."""
    if isinstance(s, int):
        return s
    if open_brace not in s and close_brace not in s:
        return s
    t = [s.split(open_brace, 1)[0].split('::')[-1]]
    targs = split_template_args(s, open_brace=open_brace,
                                close_brace=close_brace, separator=separator)
    for targ in targs:
        t.append(parse_template(targ, open_brace=open_brace,
                                close_brace=close_brace, separator=separator))
    t.append(0)
    return tuple(t)

#
# Memoization
#

def ishashable(x):
    """Tests if a value is hashable."""
    if isinstance(x, Hashable):
        if isinstance(x, basestring):
            return True
        elif isinstance(x, Iterable):
            return all(map(ishashable, x))
        else:
            return True
    else:
        return False

def memoize(obj):
    """Generic memoziation decorator based off of code from
    http://wiki.python.org/moin/PythonDecoratorLibrary .
    This is not suitabe for method caching.
    """
    cache = obj.cache = {}
    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        hashable = ishashable(key)
        if hashable:
            if key not in cache:
                cache[key] = obj(*args, **kwargs)
            return cache[key]
        else:
            return obj(*args, **kwargs)
    return memoizer

class memoize_method(object):
    """Decorator suitable for memoizing methods, rather than functions
    and classes.  This is based off of code that may be found at
    http://code.activestate.com/recipes/577452-a-memoize-decorator-for-instance-methods/
    This code was originally released under the MIT license.
    """
    def __init__(self, meth):
        self.meth = meth

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.meth
        p = functools.partial(self, obj)
        p.__doc__ = self.meth.__doc__
        p.__name__ = self.meth.__name__
        return p

    def __call__(self, *args, **kwargs):
        obj = args[0]
        cache = obj._cache = getattr(obj, '_cache', {})
        key = (self.meth, args[1:], tuple(sorted(kwargs.items())))
        hashable = ishashable(key)
        if hashable:
            if key not in cache:
                cache[key] = self.meth(*args, **kwargs)
            return cache[key]
        else:
            return self.meth(*args, **kwargs)


#
# API Name Tuples and Functions
#

class apiname(namedtuple('apiname', ['srcname', 'srcfiles', 'tarbase', 'tarname',
                                     'incfiles', 'sidecars', 'language'])):
    """apiname(srcname=NotSpecified, srcfiles=NotSpecified, tarbase=NotSpecified,
               tarname=NotSpecified, incfiles=NotSpecified, sidecars=NotSpecified,
               language=NotSpecified)

    This is a type, based on a named tuple, that represents the API element
    names and is used to declare or discover where variables, functions, and
    classes live for the rest of xdress.  All of the fields default to
    NotSpecified if not given.  These feilds are the sames as what appear in
    description dictionaries under the 'name' key.

    Parameters
    ----------
    srcname : str, tuple, or NotSpecified
        The element's API name in the original source code, eg. MyClass.
    srcfiles : str, tuple of str, or NotSpecified
        This is a sequence of unique strings which represents the file paths where
        the API element *may* be defined. For example, ('myfile.c', 'myfile.h').
        If the element is defined outside of these files, then the automatic
        discovery or description may fail. When using ensure_apiname(), all strings
        are automatically globbed, so 'src/myfile.*' will be converted to
        ('src/myfile.c', 'src/myfile.h').  Since these files are parsed they must
        actually exist on the filesystem.
    tarbase : str or NotSpecified
        The base portion of all automatically generated (target) files. This does
        not include the directory or the file extension.  For example, if you
        wanted cythongen to create a file name 'mynewfile.pyx' then the value here
        would be simply 'mynewfile'.
    tarname : str, tuple, or NotSpecified
        The element's API name in the automatically generated (target) files,
        e.g. MyNewClass.
    incfiles : tuple of str or NotSpecified
        This is a sequence of all files which must be #include'd to access the
        srcname at compile time.  This should be as minimal of a set as possible,
        preferably only one file.  For example even though most HDF5 API
        declarations are split across multiple files, to use the public API you
        only ever need to include 'hdf5.h'. Because these files are not parsed they
        do not need to exist for xdress to run.  Thus the strings in incfiles are
        not globbed.
    sidecars : str, tuple of str, or NotSpecified
        This is a sequence of all sidecar files to use for this API element. Like
        srcfiles, these files must exist for xdress to run.  They similarly
        globbed.  For example, 'myfile.py'.
    language : str or NotSpecified
        Flag for the language that the srcfiles are implemented in. Valid options
        are: 'c', 'c++', 'f', 'fortran', 'f77', 'f90', 'python', and 'cython'.

    See Also
    --------
    ensure_apiname

    """
    def __new__(cls, srcname=NotSpecified, srcfiles=NotSpecified,
                tarbase=NotSpecified, tarname=NotSpecified, incfiles=NotSpecified,
                sidecars=NotSpecified, language=NotSpecified):
        return super(apiname, cls).__new__(cls, srcname, srcfiles, tarbase, tarname,
                                           incfiles, sidecars, language)

notspecified_apiname = apiname(*([NotSpecified]*len(apiname._fields)))

def _ensure_srcfiles(inp):
    """This ensures that srcsfiles is a tuple of filenames that has been
    expanded out and the files actually exist on the file system.
    """
    if isinstance(inp, basestring):
        inp = (inp,)
    out = []
    for f in inp:
        if os.path.isfile(f):
            out.append(f)
        else:
            out += sorted([x for x in glob.glob(f) if x not in out])
    return tuple(out)

def _guess_base(srcfiles, default=None):
    """Guesses the base name for target files from source file names, or
    failing that, a default value."""
    basefiles = [os.path.basename(f) for f in srcfiles]
    basename = os.path.splitext(os.path.commonprefix(basefiles))[0]
    if len(basename) == 0:
        basename = default
    return basename

def _guess_incfiles(srcfiles):
    """This function guess potential include files from the headers that are
    present in the source files.
    """
    incs = [s for s in srcfiles if os.path.splitext(s)[-1][1:] in _hdr_exts]
    return tuple(incs)

def _ensure_incfiles(inp):
    """This ensures that incfiles is a tuple of unique filenames. This does not
    test that the files actually exist on the file system nor glob the filenames.
    """
    if isinstance(inp, basestring):
        inp = (inp,)
    if inp is NotSpecified or inp is None:
        inp = ()
    out = []
    for f in inp:
        if not isinstance(f, basestring):
            raise ValueError("{0!r} must be a string.".format(f))
        if f not in out:
            out.append(f)
    return tuple(out)

@memoize
def find_sidecar(filename):
    """Finds the sidecar for a filename, if it exists. Otherwise returns None."""
    sc = os.path.splitext(filename)[0] + '.py'
    sc = sc if os.path.isfile(sc) else None
    return sc

def _guess_sidecars(srcfiles):
    """Gueses the sidecar file names from the source file names."""
    scs = set(find_sidecar(f) for f in srcfiles)
    scs.discard(None)
    return tuple(sorted(scs))

def _find_language(lang, srcfiles):
    """Tries to discover the canonical language that the srcfiles are
    implmenetd in."""
    if isinstance(lang, basestring):
        lang = lang.lower()
        if lang not in _lang_exts:
            raise ValueError('{0} is not a valid language'.format(lang))
        return lang
    langs = set(map(guess_language, srcfiles))
    if len(langs) == 1:
        return langs.pop()
    for precedent in ['cython', 'c++', 'c', 'f90','f77', 'f', 'fortran', 'python']:
        if precedent in langs:
            return precedent
    else:
        raise ValueError("no valid language was found.")

def ensure_apiname(name):
    """Takes user input and returns the corresponding apiname named tuple.
    If the name is already an apiname instance with no NotSpecified fields,
    this does not make a copy.
    """
    # ensure is a valid apiname
    if isinstance(name, apiname):
        pass
    elif isinstance(name, Sequence):
        name = notspecified_apiname._replace(**dict(zip(apiname._fields, name)))
    elif isinstance(name, Mapping):
        name = apiname(**name)

    # ensure fields are not NotSpecified
    updates = {}
    if name.srcname is NotSpecified:
        raise ValueError("apiname.srcname cannot be unspecified")
    if name.srcfiles is NotSpecified:
        raise ValueError("apiname.srcfiles cannot be unspecified: {0}".format(name))
    updates['srcfiles'] = _ensure_srcfiles(name.srcfiles)
    if name.tarname is NotSpecified:
        updates['tarname'] = name.srcname
    if name.tarbase is NotSpecified:
        updates['tarbase'] = _guess_base(updates['srcfiles'],
                                         updates.get('tarname', name.tarname))
    if name.incfiles is NotSpecified:
        updates['incfiles'] = _guess_incfiles(updates['srcfiles'])
    updates['incfiles'] = _ensure_incfiles(updates.get('incfiles', name.incfiles))
    if name.sidecars is NotSpecified:
        updates['sidecars'] = _guess_sidecars(updates['srcfiles'])
    updates['sidecars'] = _ensure_srcfiles(updates.get('sidecars', name.sidecars))
    if name.language not in _lang_exts:
        updates['language'] = _find_language(name.language, updates['srcfiles'])
    ensured = name._replace(**updates)
    return name if name == ensured else ensured

