"""Implements a simple, dynamic type system for API generation.

:author: Anthony Scopatz <scopatz@gmail.com>

Introduction
============

This module provides a suite of tools for denoting, describing, and converting
between various data types and the types coming from various systems.  This is
achieved by providing canonical abstractions of various kinds of types:

* Base types (int, str, float, non-templated classes)
* Refined types (even or odd ints, strings containing the letter 'a')
* Dependent types (templates such arrays, maps, sets, vectors)

All types are known by their name (a string identifier) and may be aliased with 
other names.  However, the string id of a type is not sufficient to fully describe
most types.  The system here implements a canonical form for all kinds of types.
This canonical form is itself hashable, being comprised only of strings, ints, 
and tuples.

Canonical Forms
---------------
First, let us examine the base types and the forms that they may take.  Base types
are fiducial.  The type system itself may not make any changes (refinements, 
template filling) to types of this kind.  They are basically a collection of bits.
(The job of ascribing meaning to these bits falls on someone else.)  Thus base types 
may be referred to simply by their string identifier.  For example::

    'str'
    'int32'
    'float64'
    'MyClass'

Aliases to these -- or any -- type names are given in the type_aliases dictionary::

    type_aliases = {
        'i': 'int32',
        'i4': 'int32',
        'int': 'int32',
        'complex': 'complex128',
        'b': 'bool',
        }

Furthermore, length-2 tuples are used to denote a type and the name or flag of its
predicate.  A predicate is a function or transformation that may be applied to 
verify, validate, cast, coerce, or extend a variable of the given type.  A common 
usage is to declare a pointer or reference of the underlying type.  This is done with 
the string flags '*' and '&'::

    ('char', '*')
    ('float64', '&')

If the predicate is a positive integer, then this is interpreted as a 
homogeneous array of the underlying type with the given length.  If this length 
is zero, then the tuple is often interpreted as a scalar of this type, equivalent 
to the type itself.  The length-0 scalar interpretation depends on context.  Here 
are some examples of array types::

    ('char', 42)  # length-42 character array
    ('bool', 1)   # length-1 boolean array
    ('f8', 0)     # scalar 64-bit float

.. note:: 

    length-1 tuples are converted to length-2 tuples with a 0 predicate, 
    i.e. ``('char',)`` will become ``('char', 0)``.
    
The next kind of type are **refinement types** or **refined types**.  A refined type
is a sub-type of another type but restricts in some way what constitutes a valid 
element.  For example, if we first take all integers, the set of all positive 
integers is a refinement of the original.  Similarly, starting with all possible
strings the set of all strings starting with 'A' is a refinement.

In the system here, refined types are given their own unique names (e.g. 'posint' 
and 'astr').  The type system has a mapping (``refined_types``) from all refinement
type names to the names of their super-type.  The user may refer to refinement types
simply by their string name.  However the canonical form refinement types is to use
the refinement as the predicate of the super-type in a length-2 tuple, as above::

    ('int32', 'posint')  # refinement of integers to positive ints
    ('str', 'astr')      # refinement of strings to str starting with 'A'

It is these refinement types that give the second index in the tuple its 'predicate'
name.  Additionally, the predicate is used to look up the converter and validation
functions when doing code generation or type verification.

The last kind of types are known as **dependent types** or **template types**, 
similar in concept to C++ template classes.  These are meta-types whose 
instantiation requires one or more parameters to be filled in by further values or
types. Dependent types may nest with themselves or other dependent types.  Fully 
qualifying a template type requires the resolution of all dependencies.

Classic examples of dependent types include the C++ template classes.  These take
other types as their dependencies.  Other cases may require only values as 
their dependencies.  For example, suppose we want to restrict integers to various
ranges.  Rather than creating a refinement type for every combination of integer
bounds, we can use a single 'intrange' type that defines high and low dependencies.

The ``template_types`` mapping takes the dependent type names (e.g. 'map')
to a tuple of their dependency names ('key', 'value').   The ``refined_types`` 
mapping also accepts keys that are tuples of the following form::

    ('<type name>', '<dep0-name>', ('dep1-name', 'dep1-type'), ...)

Note that template names may be reused as types of other template parameters::

    ('name', 'dep0-name', ('dep1-name', 'dep0-name'))

As we have seen, dependent
types may either be base types (when based off of template classes), refined types,
or both.  Their canonical form thus follows the rules above with some additional 
syntax.  The first element of the tuple is still the type name and the last 
element is still the predicate (default 0).  However the type tuples now have a
length equal to 2 plus the number of dependencies.  These dependencies are 
placed between the name and the predicate: ``('<name>', <dep0>, ..., <predicate>)``.
These dependencies, of course, may be other type names or tuples!  Let's see
some examples.

In the simplest case, take analogies to C++ template classes::

    ('set', 'complex128', 0)
    ('map', 'int32', 'float64', 0)
    ('map', ('int32', 'posint'), 'float64', 0)
    ('map', ('int32', 'posint'), ('set', 'complex128', 0), 0)

Now consider the intrange type from above.  This has the following definition and
canonical form::

    refined_types = {('intrange', ('low', 'int32'), ('high', 'int32')): 'int32'}

    # range from 1 -> 2
    ('int32', ('intrange', ('low', 'int32', 1), ('high', 'int32', 2)))

    # range from -42 -> 42
    ('int32', ('intrange', ('low', 'int32', -42), ('high', 'int32', 42)))

Note that the low and high dependencies here are length three tuples of the form
``('<dep-name>', <dep-type>, <dep-value>)``.  How the dependency values end up 
being used is solely at the discretion of the implementation.  These values may
be anything, though they are most useful when they are easily convertible into 
strings in the target language.

.. warning:: 

    Do not confuse length-3 dependency tuples with length-3 type tuples!  
    The last element here is a value, not a predicate.

Next, consider a 'range' type which behaves similarly to 'intrange' except that
it also accepts the type as dependency.  This has the following definition and
canonical form::

    refined_types = {('range', 'vtype', ('low', 'vtype'), ('high', 'vtype')): 'vtype'}

    # integer range from 1 -> 2
    ('int32', ('range', 'int32', ('low', 'int32', 1), ('high', 'int32', 2)))    

    # positive integer range from 42 -> 65
    (('int32', 'posint'), ('range', ('int32', 'posint'),
                                    ('low', ('int32', 'posint'), 42),
                                    ('high', ('int32', 'posint'), 65)))

Shorthand Forms
---------------
The canonical forms for types contain all the information needed to fully describe
different kinds of types.  However, as human-facing code, they can be exceedingly 
verbose.  Therefore there are number of shorthand techniques that may be used to 
also denote the various types.  Converting from these shorthands to the fully
expanded version may be done via the the ``canon(t)`` function.  This function
takes a single type and returns the canonical form of that type.  The following
are operations that ``canon()``  performs:

* Base type are returned as their name::

    canon('str') == 'str'

* Aliases are resolved::

    canon('f4') == 'float32'

* Expands length-1 tuples to scalar predicates::

    t = ('int32',)
    canon(t) == ('int32', 0)

* Determines the super-type of refinements::

    canon('posint') == ('int32', 'posint')

* Applies templates::

    t = ('set', 'float')
    canon(t) == ('set', 'float64', 0)

* Applies dependencies:: 

    t = ('intrange', 1, 2)
    canon(t) = ('int32', ('intrange', ('low', 'int32', 1), ('high', 'int32', 2)))

    t = ('range', 'int32', 1, 2)
    canon(t) = ('int32', ('range', 'int32', ('low', 'int32', 1), ('high', 'int32', 2)))

* Performs all of the above recursively::

    t = (('map', 'posint', ('set', ('intrange', 1, 2))),)
    canon(t) == (('map', 
                 ('int32', 'posint'),  
                 ('set', ('int32', 
                    ('intrange', ('low', 'int32', 1), ('high', 'int32', 2))), 0)), 0)

These shorthands are thus far more useful and intuitive than canonical form described
above.  It is therefore recommended that users and developers write code that uses
the shorter versions, Note that ``canon()`` is guaranteed to return strings, tuples, 
and integers only -- making the output of this function hashable.

Built-in Template Types
-----------------------
Template type definitions that come stock with xdress::

    template_types = {
        'map': ('key_type', 'value_type'),
        'dict': ('key_type', 'value_type'),
        'pair': ('key_type', 'value_type'),
        'set': ('value_type',),
        'list': ('value_type',),
        'tuple': ('value_type',),
        'vector': ('value_type',),
        }

Built-in Refined Types
-----------------------
Refined type definitions that come stock with xdress::

    refined_types = {
        'nucid': 'int32',
        'nucname': 'str',
        ('enum', ('name', 'str'), ('aliases', ('dict', 'str', 'int32', 0))): 'int32',
        ('function', ('arguments', ('list', ('pair', 'str', 'type'))), ('returns', 'type')): 'void', 
        ('function_pointer', ('arguments', ('list', ('pair', 'str', 'type'))), ('returns', 'type')): ('void', '*'), 
        }

Type System API
===============

"""
from __future__ import print_function
import sys
import functools
from contextlib import contextmanager
from collections import Sequence, Set, Iterable, MutableMapping

if sys.version_info[0] >= 3: 
    basestring = str

def _ishashable(x):
    try:
        hash(x)
        return True
    except TypeError:
        return False 

def _memoize(obj):
    # based off code from http://wiki.python.org/moin/PythonDecoratorLibrary
    cache = obj.cache = {}
    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        key = args + tuple(sorted(kwargs.items()))
        hashable = _ishashable(key)
        if hashable:
            if key not in cache:
                cache[key] = obj(*args, **kwargs)
            return cache[key]
        else:
            return obj(*args, **kwargs)
    return memoizer


base_types = set(['char', 'uchar', 'str', 'int16', 'int32', 'int64', 'int128', 
                  'uint16', 'uint32', 'uint64', 'uint128', 'float32', 'float64', 
                  'float128', 'complex128', 'void', 'bool', 'type', 'file'])
"""Base types in the type system."""

template_types = {
    'map': ('key_type', 'value_type'),
    'dict': ('key_type', 'value_type'),
    'pair': ('key_type', 'value_type'),
    'set': ('value_type',),
    'list': ('value_type',),
    'tuple': ('value_type',),
    'vector': ('value_type',),
    }
"""Template types are types whose instantiations are based on meta-types.
this dict maps their names to meta-type names in order."""

@_memoize
def istemplate(t):
    """Returns whether t is a template type or not."""
    if isinstance(t, basestring):
        return t in template_types
    if isinstance(t, Sequence):
        return istemplate(t[0])
    return False

refined_types = {
    'nucid': 'int32',
    'nucname': 'str',
    ('enum', ('name', 'str'), ('aliases', ('dict', 'str', 'int32', 0))): 'int32',
    ('function', ('arguments', ('list', ('pair', 'str', 'type'))), 
                 ('returns', 'type')): 'void', 
    ('function_pointer', ('arguments', ('list', ('pair', 'str', 'type'))), 
                         ('returns', 'type')): ('void', '*'), 
    }
"""This is a mapping from refinement type names to the parent types.
The parent types may either be base types, compound types, template 
types, or other refined types!"""


@_memoize
def isenum(t):
    t = canon(t)
    return isinstance(t, Sequence) and t[0] == 'int32' and \
           isinstance(t[1], Sequence) and t[1][0] == 'enum'

@_memoize
def isfunctionpointer(t):
    t = canon(t)
    return isinstance(t, Sequence) and t[0] == ('void', '*') and \
           isinstance(t[1], Sequence) and t[1][0] == 'function_pointer'

_humannames = {
    'char': 'character',
    'uchar': 'unsigned character',
    'str': 'string',
    'bool': 'boolean',
    'int16': 'short integer',
    'int32': 'integer',
    'int64': 'long integer',
    'int128': 'very long integer',
    'uint16': 'unsigned short integer',
    'uint32': 'unsigned integer',
    'uint64': 'unsigned long integer',
    'uint128': 'unsigned very long integer',
    'float32': 'float',
    'float64': 'double',
    'float128': 'long double',
    'complex128': 'complex',
    'file': 'file',
    'dict': 'dict of ({key_type}, {value_type}) items',
    'map': 'map of ({key_type}, {value_type}) items',
    'pair': '({key_type}, {value_type}) pair',
    'set': 'set of {value_type}',
    'vector': 'vector [ndarray] of {value_type}',
    }

@_memoize
def humanname(t, hnt=None):
    """Computes human names for types."""
    if hnt is None:
        t = canon(t)
        if isinstance(t, basestring):
            return t, _humannames[t]
        elif t[0] in base_types:
            return t, _humannames[t[0]]
        return humanname(t, _humannames[t[0]])
    d = {}
    for key, x in zip(template_types[t[0]], t[1:-1]):
        if isinstance(x, basestring):
            val = _humannames[x]
        elif x[0] in base_types:
            val = _humannames[x[0]]
        else: 
            val, _ = humanname(x, _humannames[x[0]])
        d[key] = val
    return t, hnt.format(**d)

@_memoize
def isdependent(t):
    """Returns whether t is a dependent type or not."""
    deptypes = set([k[0] for k in refined_types if not isinstance(k, basestring)])
    if isinstance(t, basestring):
        return t in deptypes
    if isinstance(t, Sequence):
        return isdependent(t[0])
    return False


@_memoize
def isrefinement(t):
    """Returns whether t is a refined type."""
    if isinstance(t, basestring):
        return t in refined_types
    return isdependent(t)
        

def _raise_type_error(t):
    raise TypeError("type of {0!r} could not be determined".format(t))

@_memoize
def _resolve_dependent_type(tname, tinst=None):
    depkey = [k for k in refined_types if k[0] == tname][0]
    depval = refined_types[depkey]
    #istemplated = any([isinstance(x, basestring) for x in depkey[1:]])
    istemplated = istemplate(depkey)
    if tinst is None:
        return depkey
    elif istemplated:
        assert len(tinst) == len(depkey)
        typemap = dict([(k, tinst[i]) for i, k in enumerate(depkey[1:], 1) \
                                                    if isinstance(k, basestring)])
        for k in typemap:
            if k in type_aliases:
                raise TypeError('template type {0} already exists'.format(k))
        type_aliases.update(typemap)
        resotype = canon(depval), (tname,) + \
                        tuple([canon(k) for k in depkey[1:] if k in typemap]) + \
                        tuple([(k[0], canon(k[1]), instval) \
                            for k, instval in zip(depkey[1:], tinst[1:])
                            if k not in typemap])
        for k in typemap:
            del type_aliases[k]
            canon.cache.pop((k,), None)
        return resotype
    else:
        assert len(tinst) == len(depkey)
        return canon(depval), (tname,) + tuple([(kname, canon(ktype), instval) \
                        for (kname, ktype), instval in zip(depkey[1:], tinst[1:])])

@_memoize
def canon(t):
    """Turns the type into its canonical form. See module docs for more information."""
    if isinstance(t, basestring):
        if t in base_types:
            return t
        elif t in type_aliases:
            return canon(type_aliases[t])
        elif t in refined_types:
            return (canon(refined_types[t]), t)
        elif isdependent(t):
            return _resolve_dependent_type(t)
        else:
            _raise_type_error(t)
            # BELOW this would be for complicated string representations, 
            # such as 'char *' or 'map<nucid, double>'.  Would need to write
            # the parse_type() function and that might be a lot of work.
            #parse_type(t)  
    elif isinstance(t, Sequence):
        t0 = t[0]
        tlen = len(t)
        if 0 == tlen:
            _raise_type_error(t)
        last_val = 0 if tlen == 1 else t[-1]
        if isinstance(t0, basestring):
            if isdependent(t0):
                return _resolve_dependent_type(t0, t)
            elif t0 in template_types:
                templen = len(template_types[t0])
                last_val = 0 if tlen == 1 + templen else t[-1]
                filledt = [t0] + [canon(tt) for tt in t[1:1+templen]] + [last_val]
                return tuple(filledt)
            else:
                #if 2 < tlen:
                #    _raise_type_error(t)
                return (canon(t0), last_val)
        elif isinstance(t0, Sequence):
            t00 = t0[0]
            if isinstance(t00, basestring):
                # template or independent refinement type
                return (canon(t0), last_val)
            elif isinstance(t00, Sequence):
                # zOMG dependent type
                return _resolve_dependent_type(t00, t0)
            # BELOW is for possible compound types...
            #return (tuple([canon(subt) for subt in t[0]]), last_val)
        else:
            _raise_type_error(t)
    else:
        _raise_type_error(t)

@_memoize
def strip_predicates(t):
    """Removes all outer predicates from a type."""
    t = canon(t)
    if isinstance(t, basestring):
        return t
    elif isinstance(t, Sequence):
        tlen = len(t)
        if tlen == 2:
            sp0 = strip_predicates(t[0])
            return (sp0, 0) if t[1] == 0 else sp0
        else:
            return t[:-1] + (0,)
    else:
        _raise_type_error(t)
    


class MatchAny(object):
    """A singleton helper class for matching any portion of a type."""
    def __repr__(self):
        return "MatchAny"

    def __hash__(self):
        # give consistent hash value across executions
        return hash(repr(self))

MatchAny = MatchAny()
"""A singleton helper class for matching any portion of a type."""

class TypeMatcher(object):
    """A class that is used for checking whether a type matches a given pattern."""

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
        # now we know both pattern and t should be different non-string sequences, 
        # nominally tuples or lists
        if len(t) != len(pattern):
            return False
        submatcher = TypeMatcher(None)
        for subt, subpattern in zip(t, pattern):
            submatcher._pattern = subpattern
            if not submatcher.matches(subt):
                return False
        return True

    def __eq__(self, other):
        if isinstance(other, TypeMatcher):
            return self._pattern == other._pattern
        else:
            return self._pattern == other

    def __str__(self):
        return "{0}({1!s})".format(self.__class__.__name__, self._pattern)

    def __repr__(self):
        return "{0}({1!r})".format(self.__class__.__name__, self._pattern)

def matches(pattern, t):
    """Indicates whether a type t matches a pattern. See TypeMatcher for more details.
    """
    tm = TypeMatcher(pattern)
    return tm.matches(t)


#################### Type System Above This Line ##########################

EXTRA_TYPES = 'xdress_extra_types'

STLCONTAINERS = 'stlcontainers' 

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
    def __init__(self, items):
        self._d = dict(items)

    def __len__(self):
        return len(self._d)

    def __contains__(self, key):
        return key in self._d

    def __iter__(self):
        for k in self._d:
            yield k

    def __getitem__(self, key):
        value = self._d[key]
        kw = {'extra_types': _ensuremoddot(EXTRA_TYPES),
              'stlcontainers': _ensuremoddot(STLCONTAINERS),}
        for k, v in kw.items():
            value = _recurse_replace(value, '{' + k + '}', v)
        return value

    def __setitem__(self, key, value):
        self._d[key] = value

    def __delitem__(self, key):
        del self._d[key]

class _LazyImportDict(MutableMapping):
    def __init__(self, items):
        self._d = dict(items)

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
        kw = {'extra_types': _ensuremod(EXTRA_TYPES),
              'stlcontainers': _ensuremod(STLCONTAINERS),}
        newvalue = tuple(tuple(x.format(**kw) or None for x in imp if x is not None) \
                            for imp in value if imp is not None) or (None,)
        return newvalue

    def __setitem__(self, key, value):
        self._d[key] = value

    def __delitem__(self, key):
        del self._d[key]

class _LazyConverterDict(MutableMapping):
    def __init__(self, items):
        self._d = dict(items)
        self._tms = set([k for k in self._d if isinstance(k, TypeMatcher)])

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
        kw = {'extra_types': _ensuremoddot(EXTRA_TYPES),
              'stlcontainers': _ensuremoddot(STLCONTAINERS),}
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

type_aliases = _LazyConfigDict({
    'i': 'int32',
    'i2': 'int16',
    'i4': 'int32',
    'i8': 'int64',
    'i16': 'int128',
    'int': 'int32',
    'ui': 'uint32',
    'ui2': 'uint16',
    'ui4': 'uint32',
    'ui8': 'uint64',
    'ui16': 'uint128',
    'uint': 'uint32',
    'f': 'float64',
    'f4': 'float32',
    'f8': 'float64',
    'f16': 'float128',
    'float': 'float64',
    'complex': 'complex128',
    'b': 'bool',
    'v': 'void',
    's': 'str',
    'string': 'str',
    'FILE': 'file',
    '_IO_FILE': 'file',
    # 'c' has char / complex ambiguity, not included
    'NPY_BYTE': 'char',
    'NPY_UBYTE': 'uchar',
    'NPY_STRING': 'str',
    'NPY_INT16': 'int16',
    'NPY_INT32': 'int32',
    'NPY_INT64': 'int64',
    'NPY_UINT16': 'uint16',
    'NPY_UINT32': 'uint32',
    'NPY_UINT64': 'uint64',
    'NPY_FLOAT32': 'float32',
    'NPY_FLOAT64': 'float64',
    'NPY_COMPLEX128': 'complex128',
    'NPY_BOOL': 'bool',
    'NPY_VOID': 'void',
    'NPY_OBJECT': 'void',
    'np.NPY_BYTE': 'char',
    'np.NPY_UBYTE': 'uchar',
    'np.NPY_STRING': 'str',
    'np.NPY_INT16': 'int16',
    'np.NPY_INT32': 'int32',
    'np.NPY_INT64': 'int64',
    'np.NPY_UINT16': 'uint16',
    'np.NPY_UINT32': 'uint32',
    'np.NPY_UINT64': 'uint64',
    'np.NPY_FLOAT32': 'float32',
    'np.NPY_FLOAT64': 'float64',
    'np.NPY_COMPLEX128': 'complex128',
    'np.NPY_BOOL': 'bool',
    'np.NPY_VOID': 'void',
    'np.NPY_OBJECT': 'void',
    })
"""Aliases that may be used to substitute one type name for another."""


#########################   Cython Functions   ################################

_cython_ctypes = _LazyConfigDict({
    'char': 'char',
    #'uchar': 'unsigned char',
    'uchar': '{extra_types}uchar',
    'str': 'std_string',
    'int16': 'short',
    #'int32': 'long',
    'int32': 'int',
    #'int64': 'long long',
    'int64': '{extra_types}int64',
    #'uint16': 'unsigned short',
    'uint16': '{extra_types}uint16',
    #'uint32': 'unsigned long',
    'uint32': '{extra_types}uint32',  
    #'uint64': 'unsigned long long',
    'uint64': '{extra_types}uint64',
    'float32': 'float',
    'float64': 'double',
    #'float128': 'long double',
    'float128': '{extra_types}float128',
    'complex128': '{extra_types}complex_t',
    'bool': 'bint',
    'void': 'void', 
    'file': 'c_file',
    'map': 'cpp_map',
    'dict': 'dict',
    'pair': 'cpp_pair',
    'set': 'cpp_set',
    'vector': 'cpp_vector',
    })

def _cython_ctypes_function(t):
    rtnct = cython_ctype(t[2][2])
    argcts = [cython_ctype(argt) for n, argt in t[1][2]]
    if argcts == ['void']:
        argcts = []
    return rtnct + " {type_name}(" + ", ".join(argcts) + ")"
_cython_ctypes['function'] = _cython_ctypes_function

def _cython_ctypes_function_pointer(t):
    rtnct = cython_ctype(t[2][2])
    argcts = [cython_ctype(argt) for n, argt in t[1][2]]
    if argcts == ['void']:
        argcts = []
    return rtnct + " (*{type_name})(" + ", ".join(argcts) + ")"
_cython_ctypes['function_pointer'] = _cython_ctypes_function_pointer


def _cython_ctype_add_predicate(t, last):
    """Adds a predicate to a ctype"""
    if last == 'const':
        x, y = last, t
    else:
        x, y = t, last
    return '{0} {1}'.format(x, y)


@_memoize
def cython_ctype(t):
    """Given a type t, returns the corresponding Cython C type declaration."""
    t = canon(t)
#    if t in _cython_ctypes:
#        return _cython_ctypes[t]
    if isinstance(t, basestring):
        if  t in base_types:
            return _cython_ctypes[t]
    # must be tuple below this line
    tlen = len(t)
    if 2 == tlen:
        if 0 == t[1]:
            return cython_ctype(t[0])
        elif isrefinement(t[1]):
            if t[1][0] in _cython_ctypes:
                subtype = _cython_ctypes[t[1][0]]
                if callable(subtype):
                    subtype = subtype(t[1])
                return subtype
            else:
                return cython_ctype(t[0])
        else:
            last = '[{0}]'.format(t[-1]) if isinstance(t[-1], int) else t[-1]
            return _cython_ctype_add_predicate(cython_ctype(t[0]), last)
    elif 3 <= tlen:
        assert t[0] in template_types
        assert len(t) == len(template_types[t[0]]) + 2
        template_name = _cython_ctypes[t[0]]
        assert template_name is not NotImplemented
        template_filling = ', '.join([cython_ctype(x) for x in t[1:-1]])
        cyct = '{0}[{1}]'.format(template_name, template_filling)
        if 0 != t[-1]:
            last = '[{0}]'.format(t[-1]) if isinstance(t[-1], int) else t[-1]
            cyct = _cython_ctype_add_predicate(cyct, last)
        return cyct


_cython_cimports = _LazyImportDict({
    'char': (None,),
    'uchar':  (('{extra_types}',),),
    'str': (('libcpp.string', 'string', 'std_string'),),
    'int16': (None,),
    'int32': (None,),
    ('int32', '*'): 'int *',
    'int64':  (('{extra_types}',),),
    'uint16':  (('{extra_types}',),),
    'uint32': (('{extra_types}',),),  
    'uint64':  (('{extra_types}',),),
    'float32': (None,),
    'float64': (None,),
    'float128':  (('{extra_types}',),),
    'complex128': (('{extra_types}',),),
    'bool': (None,), 
    'void': (None,), 
    'file': (('libc.stdio', 'FILE', 'c_file'),),
    'map': (('libcpp.map', 'map', 'cpp_map'),),
    'dict': (None,),
    'pair': (('libcpp.utility', 'pair', 'cpp_pair'),),
    'set': (('libcpp.set', 'set', 'cpp_set'),),
    'vector': (('libcpp.vector', 'vector', 'cpp_vector'),),
    'nucid': (('pyne', 'cpp_nucname'),),
    'nucname': (('pyne', 'cpp_nucname'), ('libcpp.string', 'string', 'std_string')),
    })

def _cython_cimports_functionish(t, seen):
    seen.add(('cython.operator', 'dereference', 'deref'))
    for n, argt in t[1][2]:
        cython_cimport_tuples(argt, seen=seen, inc=('c',))
    cython_cimport_tuples(t[2][2], seen=seen, inc=('c',))
_cython_cimports['function'] = _cython_cimports_functionish
_cython_cimports['function_pointer'] = _cython_cimports_functionish

_cython_cyimports = _LazyImportDict({
    'char': (None,),
    'uchar': (None,),
    'str': (None,),
    'int16': (None,),
    'int32': (None,),
    'int64': (None,),
    'uint16': (None,),
    'uint32': (None,),
    'uint64': (None,),
    'float32': (None,),
    'float64': (None,),
    'float128': (None,),
    'complex128': (('{extra_types}',),),  # for py2c_complex()
    'bool': (None,), 
    'void': (None,), 
    'file': (('{extra_types}',),), 
    'map': (('{stlcontainers}',),),
    'dict': (None,),
    'pair': (('{stlcontainers}',),), 
    'set': (('{stlcontainers}',),),
    'vector': (('numpy', 'as', 'np'),),
    'nucid': (('pyne', 'nucname'),),
    'nucname': (('pyne', 'nucname'),),
    })

def _cython_cyimports_functionish(t, seen):
    for n, argt in t[1][2]:
        cython_cimport_tuples(argt, seen=seen, inc=('cy',))
    cython_cimport_tuples(t[2][2], seen=seen, inc=('cy',))
_cython_cyimports['function'] = _cython_cyimports_functionish
_cython_cyimports['function_pointer'] = _cython_cyimports_functionish

@_memoize
def cython_cimport_tuples(t, seen=None, inc=frozenset(['c', 'cy'])):
    """Given a type t, and possibly previously seen cimport tuples (set), 
    return the set of all seen cimport tuples.  These tuple have four possible 
    interpretations based on the length and values:

    * ``(module-name,)`` becomes ``cimport {module-name}``
    * ``(module-name, var-or-mod)`` becomes 
      ``from {module-name} cimport {var-or-mod}``
    * ``(module-name, var-or-mod, alias)`` becomes 
      ``from {module-name} cimport {var-or-mod} as {alias}``
    * ``(module-name, 'as', alias)`` becomes ``cimport {module-name} as {alias}``

    """
    t = canon(t)
    if seen is None:
        seen = set()
    if isinstance(t, basestring):
        if t in base_types:
            if 'c' in inc:
                seen.update(_cython_cimports[t])
            if 'cy' in inc:
                seen.update(_cython_cyimports[t])
            seen -= set((None, (None,)))
            return seen        
    # must be tuple below this line
    tlen = len(t)
    if 2 == tlen:
        if 'c' in inc:
            if isrefinement(t[1]) and t[1][0] in _cython_cimports:
                f = _cython_cimports[t[1][0]]
                if callable(f):
                    f(t[1], seen)
            seen.update(_cython_cimports.get(t[0], (None,)))
            seen.update(_cython_cimports.get(t[1], (None,)))
        if 'cy' in inc:
            if isrefinement(t[1]) and t[1][0] in _cython_cyimports:
                f = _cython_cyimports[t[1][0]]
                if callable(f):
                    f(t[1], seen)
            seen.update(_cython_cyimports.get(t[0], (None,)))
            seen.update(_cython_cyimports.get(t[1], (None,)))
        seen -= set((None, (None,)))
        return cython_cimport_tuples(t[0], seen, inc)
    elif 3 <= tlen:
        assert t[0] in template_types
        if 'c' in inc:
            seen.update(_cython_cimports[t[0]])
        if 'cy' in inc:
            seen.update(_cython_cyimports[t[0]])
        for x in t[1:-1]:
            cython_cimport_tuples(x, seen, inc)
        seen -= set((None, (None,)))
        return seen

_cython_cimport_cases = {
    1: lambda tup: "cimport {0}".format(*tup),
    2: lambda tup: "from {0} cimport {1}".format(*tup),
    3: lambda tup: ("cimport {0} as {2}".format(*tup) if tup[1] == 'as' else \
                    "from {0} cimport {1} as {2}".format(*tup)),
    }

@_memoize
def cython_cimports(x, inc=frozenset(['c', 'cy'])):
    """Returns the cimport lines associated with a type or a set of seen tuples.
    """
    if not isinstance(x, Set):
        x = cython_cimport_tuples(x, inc=inc)
    return set([_cython_cimport_cases[len(tup)](tup) for tup in x if 0 != len(tup)])



_cython_pyimports = _LazyImportDict({
    'char': (None,),
    'uchar': (None,),
    'str': (None,),
    'int16': (None,),
    'int32': (None,),
    'int64': (None,),
    'uint16': (None,),
    'uint32': (None,),
    'uint64': (None,),
    'float32': (None,),
    'float64': (None,),
    'float128': (None,),
    'complex128': (None,),
    'bool': (None,), 
    'void': (None,), 
    'file': (None,), 
    'map': (('{stlcontainers}',),),
    'dict': (None,),
    'pair': (('{stlcontainers}',),),
    'set': (('{stlcontainers}',),),
    'vector': (('numpy', 'as', 'np'),),
    'nucid': (('pyne', 'nucname'),),
    'nucname': (('pyne', 'nucname'),),
    })

def _cython_pyimports_functionish(t, seen):
    seen.add(('warnings',))
    for n, argt in t[1][2]:
        cython_import_tuples(argt, seen=seen)
    cython_import_tuples(t[2][2], seen=seen)
_cython_pyimports['function'] = _cython_pyimports_functionish
_cython_pyimports['function_pointer'] = _cython_pyimports_functionish


@_memoize
def cython_import_tuples(t, seen=None):
    """Given a type t, and possibly previously seen import tuples (set), 
    return the set of all seen import tuples.  These tuple have four possible 
    interpretations based on the length and values:

    * ``(module-name,)`` becomes ``import {module-name}``
    * ``(module-name, var-or-mod)`` becomes 
      ``from {module-name} import {var-or-mod}``
    * ``(module-name, var-or-mod, alias)`` becomes 
      ``from {module-name} import {var-or-mod} as {alias}``
    * ``(module-name, 'as', alias)`` becomes ``import {module-name} as {alias}``

    """
    t = canon(t)
    if seen is None:
        seen = set()
    if isinstance(t, basestring):
        if  t in base_types:
            seen.update(_cython_pyimports[t])
            seen -= set((None, (None,)))
            return seen
    # must be tuple below this line
    tlen = len(t)
    if 2 == tlen:
        if isrefinement(t[1]) and t[1][0] in _cython_pyimports:
            f = _cython_pyimports[t[1][0]]
            if callable(f):
                f(t[1], seen)
        seen.update(_cython_pyimports.get(t[0], (None,)))
        seen.update(_cython_pyimports.get(t[1], (None,)))
        seen -= set((None, (None,)))
        return cython_import_tuples(t[0], seen)
    elif 3 <= tlen:
        assert t[0] in template_types
        seen.update(_cython_pyimports[t[0]])
        for x in t[1:-1]:
            cython_import_tuples(x, seen)
        seen -= set((None, (None,)))
        return seen

_cython_import_cases = {
    1: lambda tup: "import {0}".format(*tup),
    2: lambda tup: "from {0} import {1}".format(*tup),
    3: lambda tup: ("import {0} as {2}".format(*tup) if tup[1] == 'as' else \
                    "from {0} import {1} as {2}".format(*tup)),
    }

@_memoize
def cython_imports(x):
    """Returns the import lines associated with a type or a set of seen tuples.
    """
    if not isinstance(x, Set):
        x = cython_import_tuples(x)
    x = [tup for tup in x if 0 < len(tup)]
    return set([_cython_import_cases[len(tup)](tup) for tup in x])


_cython_cytypes = _LazyConfigDict({
    'char': 'char',
    'uchar': 'unsigned char',
    'str': 'char *',
    'int16': 'short',
    #'int32': 'long',
    'int32': 'int',
    ('int32', '*'): 'int *',
    'int64': 'long long',
    'uint16': 'unsigned short',  
    'uint32': 'unsigned long',  # 'unsigned int'
    'uint64': 'unsigned long long', 
    'float32': 'float',
    'float64': 'float',
    'float128': 'long double',
    'complex128': 'object',
    'bool': 'bint',
    'void': 'void',
    'file': 'c_file',
    'map': '{stlcontainers}_Map{key_type}{value_type}',
    'dict': 'dict',
    'pair': '{stlcontainers}_Pair{value_type}',
    'set': '{stlcontainers}_Set{value_type}',
    'vector': 'np.ndarray',
    'function': 'object',
    'function_pointer': 'object',
    })

_cython_functionnames = _LazyConfigDict({
    # base types
    'char': 'char',
    'uchar': 'uchar',
    'str': 'str',
    'int16': 'short',
    'int32': 'int',
    'int64': 'long',
    'uint16': 'ushort',
    'uint32': 'uint',
    'uint64': 'ulong',
    'float32': 'float',
    'float64': 'double',
    'float128': 'longdouble',
    'complex128': 'complex',
    'bool': 'bool',
    'void': 'void',
    'file': 'file',
    # template types
    'map': 'map_{key_type}_{value_type}',
    'dict': 'dict',
    'pair': 'pair_{key_type}_{value_type}',
    'set': 'set_{value_type}',
    'vector': 'vector_{value_type}',    
    'nucid': 'nucid', 
    'nucname': 'nucname',
    'function': 'function', 
    'function_pointer': 'functionpointer', 
    })

@_memoize
def cython_functionname(t, cycyt=None):
    """Computes variable or function names for cython types."""
    if cycyt is None:
        t = canon(t)
        if isinstance(t, basestring):
            return t, _cython_functionnames[t]
        elif t[0] in base_types:
            return t, _cython_functionnames[t[0]]
        return cython_functionname(t, _cython_functionnames[t[0]])
    d = {}
    for key, x in zip(template_types[t[0]], t[1:-1]):
        if isinstance(x, basestring):
            val = _cython_functionnames[x]
        elif x[0] in base_types:
            val = _cython_functionnames[x[0]]
        else: 
            _, val = cython_functionname(x, _cython_functionnames[x[0]])
        d[key] = val
    return t, cycyt.format(**d)

cython_variablename = cython_functionname

_cython_classnames = _LazyConfigDict({
    # base types
    'char': 'Char',
    'uchar': 'UChar',
    'str': 'Str',
    'int32': 'Short',
    'int32': 'Int',
    'int64': 'Long',
    'uint16': 'UShort',
    'uint32': 'UInt',
    'uint64': 'ULong',
    'float32': 'Float',
    'float64': 'Double',
    'float128': 'Longdouble',
    'complex128': 'Complex',
    'bool': 'Bool',
    'void': 'Void',
    'file': 'File',
    # template types
    'map': 'Map{key_type}{value_type}',
    'dict': 'Dict',
    'pair': 'Pair{key_type}{value_type}',
    'set': 'Set{value_type}',
    'vector': 'Vector{value_type}',    
    'nucid': 'Nucid', 
    'nucname': 'Nucname',
    })

@_memoize
def _fill_cycyt(cycyt, t):
    """Helper for cython_cytype()."""
    d = {}
    for key, x in zip(template_types[t[0]], t[1:-1]):
        if isinstance(x, basestring):
            val = _cython_classnames[x]
        elif x[0] in base_types:
            val = _cython_classnames[x[0]]
        else: 
            val, _ = _fill_cycyt(_cython_classnames[x[0]], x)
        d[key] = val
    return cycyt.format(**d), t
    
@_memoize
def cython_classname(t, cycyt=None):
    """Computes classnames for cython types."""
    if cycyt is None:
        t = canon(t)
        if isinstance(t, basestring):
            return t, _cython_classnames[t]
        elif t[0] in base_types:
            return t, _cython_classnames[t[0]]
        return cython_classname(t, _cython_classnames[t[0]])
    d = {}
    for key, x in zip(template_types[t[0]], t[1:-1]):
        if isinstance(x, basestring):
            val = _cython_classnames[x]
        elif x[0] in base_types:
            val = _cython_classnames[x[0]]
        else: 
            _, val = cython_classname(x, _cython_classnames[x[0]])
        d[key] = val
    return t, cycyt.format(**d)

def _cython_cytype_add_predicate(t, last):
    """Adds a predicate to a cytype"""
    if last == '*':
        return '{0} {1}'.format(t, last)
    elif isinstance(last, int) and 0 < last:
        return '{0} [{1}]'.format(t, last)        
    else:
        return t

@_memoize
def cython_cytype(t):
    """Given a type t, returns the corresponding Cython type."""
    t = canon(t)
#    if t in _cython_cytypes:
#        return _cython_cytypes[t]
    if isinstance(t, basestring):
        if t in base_types or t in _cython_cytypes:
            return _cython_cytypes[t]
    # must be tuple below this line
    tlen = len(t)
    if 2 == tlen:
        if 0 == t[1]:
            return cython_cytype(t[0])
        elif isrefinement(t[1]):
            if t[1][0] in _cython_cytypes:
                subtype = _cython_cytypes[t[1][0]]
                if callable(subtype):
                    subtype = subtype(t[1])
                return subtype
            else:
                return cython_cytype(t[0])
        else:
            return _cython_cytype_add_predicate(cython_cytype(t[0]), t[-1])
    elif 3 <= tlen:
        if t in _cython_cytypes:
            return _cython_cytypes[t]
        assert t[0] in template_types
        assert len(t) == len(template_types[t[0]]) + 2
        template_name = _cython_cytypes[t[0]]
        assert template_name is not NotImplemented        
        cycyt = _cython_cytypes[t[0]]
        cycyt, t = _fill_cycyt(cycyt, t)
        cycyt = _cython_cytype_add_predicate(cycyt, t[-1])
        return cycyt


_cython_pytypes = _LazyConfigDict({
    'char': 'str',
    'uchar': 'str',
    'str': 'str',
    'int16': 'int',
    'int32': 'int',
    'int64': 'int',
    'uint16': 'int',  # 'unsigned int'
    'uint32': 'int',  # 'unsigned int'
    'uint64': 'int',  # 'unsigned int'
    'float32': 'float',
    'float64': 'float',
    'float128': 'np.float128',
    'complex128': 'object',
    'file': 'file',
    'bool': 'bool',
    'void': 'object',
    'map': '{stlcontainers}Map{key_type}{value_type}',
    'dict': 'dict',
    'pair': '{stlcontainers}Pair{value_type}',
    'set': '{stlcontainers}Set{value_type}',
    'vector': 'np.ndarray',
    })

@_memoize
def _fill_cypyt(cypyt, t):
    """Helper for cython_pytype()."""
    d = {}
    for key, x in zip(template_types[t[0]], t[1:-1]):
        if isinstance(x, basestring):
            val = _cython_classnames[x]
        elif x[0] in base_types:
            val = _cython_classnames[x[0]]
        else: 
            val, _ = _fill_cypyt(_cython_classnames[x[0]], x)
        d[key] = val
    return cypyt.format(**d), t
    

@_memoize
def cython_pytype(t):
    """Given a type t, returns the corresponding Python type."""
    t = canon(t)
#    if t in _cython_pytypes:
#        return _cython_pytypes[t]
    if isinstance(t, basestring):
        if  t in base_types:
            return _cython_pytypes[t]
    # must be tuple below this line
    tlen = len(t)
    if 2 == tlen:
        if 0 == t[1]:
            return cython_pytype(t[0])
        elif isrefinement(t[1]):
            return cython_pytype(t[0])
        else:
            # FIXME last is ignored for strings, but what about other types?
            #last = '[{0}]'.format(t[-1]) if isinstance(t[-1], int) else t[-1]
            #return cython_pytype(t[0]) + ' {0}'.format(last)
            return cython_pytype(t[0])
    elif 3 <= tlen:
        if t in _cython_pytypes:
            return _cython_pytypes[t]
        assert t[0] in template_types
        assert len(t) == len(template_types[t[0]]) + 2
        template_name = _cython_pytypes[t[0]]
        assert template_name is not NotImplemented        
        cypyt = _cython_pytypes[t[0]]
        cypyt, t = _fill_cypyt(cypyt, t)
        # FIXME last is ignored for strings, but what about other types?
        #if 0 != t[-1]:
        #    last = '[{0}]'.format(t[-1]) if isinstance(t[-1], int) else t[-1]
        #    cypyt += ' {0}'.format(last)
        return cypyt




_numpy_types = _LazyConfigDict({
    'char': 'np.NPY_BYTE',
    'uchar': 'np.NPY_UBYTE',
    #'str': 'np.NPY_STRING',
    'int16': 'np.NPY_INT16',
    'int32': 'np.NPY_INT32',
    'int64': 'np.NPY_INT64',
    'uint16': 'np.NPY_UINT16',
    'uint32': 'np.NPY_UINT32',
    'uint64': 'np.NPY_UINT64',
    'float32': 'np.NPY_FLOAT32',
    'float64': 'np.NPY_FLOAT64',
    'float128': 'np.NPY_FLOAT128',
    'complex128': 'np.NPY_COMPLEX128',
    'bool': 'np.NPY_BOOL',
    'void': 'np.NPY_VOID',     
    })

@_memoize
def cython_nptype(t, depth=0):
    """Given a type t, returns the corresponding numpy type.  If depth is 
    greater than 0 then this returns of a list of numpy types for all internal 
    template types, ie the float in ('vector', 'float', 0)."""
    t = canon(t)
    if isinstance(t, basestring):
        return _numpy_types[t] if t in _numpy_types else 'np.NPY_OBJECT'
    # must be tuple below this line
    tlen = len(t)
    if t in _numpy_types and depth < 1:
        return _numpy_types[t]
    elif 2 == tlen:
        if 0 == t[1]:
            return cython_nptype(t[0])
        elif isrefinement(t[1]):
            return cython_nptype(t[0])
        else:
            # FIXME last is ignored for strings, but what about other types?
            #last = '[{0}]'.format(t[-1]) if isinstance(t[-1], int) else t[-1]
            #return cython_pytype(t[0]) + ' {0}'.format(last)
            return cython_nptype(t[0])
    elif 0 < depth and istemplate(t):
        depth -= 1
        return [cython_nptype(u, depth=depth) for u in t[1:-1]]
    elif 3 == tlen and istemplate(t):
        return cython_nptype(t[1])
    else:  #elif 3 <= tlen:
        return 'np.NPY_OBJECT'

_cython_c2py_conv = _LazyConverterDict({
    # Has tuple form of (copy, [view, [cached_view]])
    # base types
    'char': ('chr(<int> {var})',),
    'uchar': ('chr(<unsigned int> {var})',),
    'str': ('bytes(<char *> {var}.c_str()).decode()',),
    'int16': ('int({var})',),
    'int32': ('int({var})',),
    ('int32', '*'): ('int({var}[0])',),
    'int64': ('int({var})',),
    'uint16': ('int({var})',),
    'uint32': ('int({var})',),
    ('uint32', '*'): ('int({var}[0])',),
    'uint64': ('int({var})',),
    'float32': ('float({var})',),
    'float64': ('float({var})',),
    'float128': ('np.array({var}, dtype=np.float128)',),
    'complex128': ('complex(float({var}.re), float({var}.im))',),
    'bool': ('bool({var})',),
    'void': ('None',),
    'file': ('{extra_types}PyFile_FromFile(&{var}, "{var}", "r+", NULL)',),
    ('file', '*'): ('{extra_types}PyFile_FromFile({var}, "{var}", "r+", NULL)',),
    # template types
    'map': ('{pytype}({var})', 
           ('{proxy_name} = {pytype}(False, False)\n'
            '{proxy_name}.map_ptr = &{var}\n'),
           ('if {cache_name} is None:\n'
            '    {proxy_name} = {pytype}(False, False)\n'
            '    {proxy_name}.map_ptr = &{var}\n'
            '    {cache_name} = {proxy_name}\n'
            )),
    'dict': ('dict({var})',),
    'pair': ('{pytype}({var})',
             ('{proxy_name} = {pytype}(False, False)\n'
              '{proxy_name}.pair_ptr = &{var}\n'),
             ('if {cache_name} is None:\n'
              '    {proxy_name} = {pytype}(False, False)\n'
              '    {proxy_name}.pair_ptr = &{var}\n'
              '    {cache_name} = {proxy_name}\n'
              )),
    'set': ('{pytype}({var})',
           ('{proxy_name} = {pytype}(False, False)\n'
            '{proxy_name}.set_ptr = &{var}\n'),
           ('if {cache_name} is None:\n'
            '    {proxy_name} = {pytype}(False, False)\n'
            '    {proxy_name}.set_ptr = &{var}\n'
            '    {cache_name} = {proxy_name}\n'
            )),
    'vector': (('{proxy_name}_shape[0] = <np.npy_intp> {var}.size()\n'
                '{proxy_name} = np.PyArray_SimpleNewFromData(1, {var}_shape, {nptypes[0]}, &{var}[0])\n'
                '{proxy_name} = np.PyArray_Copy({proxy_name})\n'),
               ('{proxy_name}_shape[0] = <np.npy_intp> {var}.size()\n'
                '{proxy_name} = np.PyArray_SimpleNewFromData(1, {proxy_name}_shape, {nptypes[0]}, &{var}[0])\n'),
               ('if {cache_name} is None:\n'
                '    {proxy_name}_shape[0] = <np.npy_intp> {var}.size()\n'
                '    {proxy_name} = np.PyArray_SimpleNewFromData(1, {proxy_name}_shape, {nptypes[0]}, &{var}[0])\n'
                '    {cache_name} = {proxy_name}\n'
                )),
    ('vector', 'bool', 0): (  # C++ standard is silly here
               ('cdef int i\n'
                '{proxy_name}_shape[0] = <np.npy_intp> {var}.size()\n'
                '{proxy_name} = np.PyArray_SimpleNew(1, {proxy_name}_shape, {nptypes[0]})\n'
                'for i in range({proxy_name}_shape[0]):\n' 
                '    {proxy_name}[i] = {var}[i]\n'),
               ('cdef int i\n'
                '{proxy_name}_shape[0] = <np.npy_intp> {var}.size()\n'
                '{proxy_name} = np.PyArray_SimpleNew(1, {proxy_name}_shape, {nptypes[0]})\n'
                'for i in range({proxy_name}_shape[0]):\n' 
                '    {proxy_name}[i] = {var}[i]\n'),
               ('cdef int i\n'
                'if {cache_name} is None:\n'
                '    {proxy_name}_shape[0] = <np.npy_intp> {var}.size()\n'
                '    {proxy_name} = np.PyArray_SimpleNew(1, {proxy_name}_shape, {nptype[0]})\n'
                '    for i in range({proxy_name}_shape[0]):\n' 
                '        {proxy_name}[i] = {var}[i]\n'
                '    {cache_name} = {proxy_name}\n'
                )),
    ('vector', 'char', 0): (  # C/C++ chars are ints while Python Chars are length-1 strings
               ('cdef int i\n'
                '{proxy_name}_shape[0] = <np.npy_intp> {var}.size()\n'
                '{proxy_name} = np.empty({proxy_name}_shape[0], "U1")\n'
                'for i in range({proxy_name}_shape[0]):\n' 
                '    {proxy_name}[i] = chr(<int> {var}[i])\n'),
               ('cdef int i\n'
                '{proxy_name}_shape[0] = <np.npy_intp> {var}.size()\n'
                '{proxy_name} = np.empty({proxy_name}_shape[0], "U1")\n'
                'for i in range({proxy_name}_shape[0]):\n' 
                '    {proxy_name}[i] = chr(<int> {var}[i])\n'),
               ('cdef int i\n'
                'if {cache_name} is None:\n'
                '    {proxy_name}_shape[0] = <np.npy_intp> {var}.size()\n'
                '    for i in range({proxy_name}_shape[0]):\n'
                '        {proxy_name}[i] = chr(<int> {var}[i])\n'
                '    {cache_name} = {proxy_name}\n'
                )),
    'nucid': ('nucname.zzaaam({var})',),
    'nucname': ('nucname.name({var})',),
    TypeMatcher((('enum', MatchAny, MatchAny), '*')): ('int({var}[0])',),
    TypeMatcher((('int32', ('enum', MatchAny, MatchAny)), '*')): ('int({var}[0])',),
    })

def _cython_c2py_conv_function_pointer(t_):
    t = t_[1]
    argnames = []
    argdecls = []
    argbodys = []
    argrtns = []
    for n, argt in t[1][2]:
        argnames.append(n)
        decl, body, rtn = cython_py2c(n, argt, proxy_name="c_" + n)
        argdecls.append(decl)
        argbodys.append(body)
        argrtns.append(rtn)
    rtnname = 'rtn'
    rtnprox = 'c_' + rtnname
    rtncall = 'c_call_' + rtnname
    while rtnname in argnames or rtnprox in argnames:
        rtnname += '_'
        rtnprox += '_'
    argdecls = _indent4(argdecls)
    argbodys = _indent4(argbodys)
#    rtndecl, rtnbody, rtnrtn, _ = cython_c2py(rtnname, t[2][2], cached=False, 
#                                              proxy_name=rtnprox, 
#                                              existing_name=rtncall)
    rtndecl, rtnbody, rtnrtn, _ = cython_c2py(rtncall, t[2][2], cached=False, 
                                              proxy_name=rtnprox,
                                              existing_name=rtncall)
    if rtndecl is None and rtnbody is None:
        rtnprox = rtnname
    rtndecl = _indent4([rtndecl, "cdef {0} {1}".format(cython_ctype(t[2][2]), rtncall)])
    #rtndecl = _indent4(["cdef {0} {1}".format(cython_ctype(t[2][2]), rtncall)])
    rtnbody = _indent4([rtnbody])
    s = """def {{proxy_name}}({arglist}):
{argdecls}
{rtndecl}
    if {{var}} == NULL:
        raise RuntimeError("{{var}} is NULL and may not be safely called!")
{argbodys}
    {rtncall} = {{var}}({carglist})
{rtnbody}
"""
    s = s.format(arglist=", ".join(argnames), argdecls=argdecls, 
                 cvartypeptr=cython_ctype(t_).format(type_name='cvartype'), 
                 argbodys=argbodys, rtndecl=rtndecl, rtnprox=rtnprox, rtncall=rtncall,
                 carglist=", ".join(argrtns), rtnbody=rtnbody)
    caches = 'if {cache_name} is None:\n' + _indent4([s]) 
    if t[2][2] != 'void':
        caches += "\n        return {rtnrtn}".format(rtnrtn=rtnrtn)
    caches += '\n    {cache_name} = {proxy_name}\n'
    return s, s, caches

_cython_c2py_conv['function_pointer'] = _cython_c2py_conv_function_pointer

from_pytypes = {
    'str': ['basestring'],
    'char': ['basestring'],
    'uchar': ['basestring'],
    'int16': ['int'],
    'int32': ['int'],
    'int64': ['int'],
    'uint16': ['int'],
    'uint32': ['int'],
    'uint64': ['int'],
    'float32': ['float', 'int'],
    'float64': ['float', 'int'],
    'complex128': ['complex', 'float'],
    'file': ['file'],
    ('file', '*'): ['file'],
    'set': ['set', 'list', 'basestring', 'tuple'],
    'vector': ['list', 'tuple', 'np.ndarray'],
    }


@_memoize
def cython_c2py(name, t, view=True, cached=True, inst_name=None, proxy_name=None, 
                cache_name=None, cache_prefix='self', existing_name=None):
    """Given a varibale name and type, returns cython code (declaration, body, 
    and return statements) to convert the variable from C/C++ to Python."""
    tkey = t = canon(t)
    while tkey not in _cython_c2py_conv and not isinstance(tkey, basestring):
        #tkey = tkey[0]
        tkey = tkey[1] if (0 < len(tkey) and isrefinement(tkey[1])) else tkey[0]
    if tkey not in _cython_c2py_conv:
        tkey = t
        while tkey not in _cython_c2py_conv and not isinstance(tkey, basestring):
            tkey = tkey[0]
    c2pyt = _cython_c2py_conv[tkey]
    if callable(c2pyt):
        _cython_c2py_conv[t] = c2pyt(t)
        c2pyt = _cython_c2py_conv[t]
    ind = int(view) + int(cached)
    if cached and not view:
        raise ValueError('cached views require view=True.')
    if c2pyt is NotImplemented:
        raise NotImplementedError('conversion from C/C++ to Python for ' + \
                                  t + 'has not been implemented for when ' + \
                                  'view={0}, cached={1}'.format(view, cached))
    ct = cython_ctype(t)
    cyt = cython_cytype(t)
    pyt = cython_pytype(t)
    npt = cython_nptype(t)
    npts = cython_nptype(t, depth=1)
    npts = [npts] if isinstance(npts, basestring) else npts
    t_nopred = strip_predicates(t)
    ct_nopred = cython_ctype(t_nopred)
    cyt_nopred = cython_cytype(t_nopred)
    var = name if inst_name is None else "{0}.{1}".format(inst_name, name)
    var = existing_name or var
    cache_name = "_{0}".format(name) if cache_name is None else cache_name
    cache_name = cache_name if cache_prefix is None else "{0}.{1}".format(cache_prefix, cache_name)
    proxy_name = "{0}_proxy".format(name) if proxy_name is None else proxy_name
    iscached = False
    template_kw = dict(var=var, cache_name=cache_name, ctype=ct, cytype=cyt, 
                       pytype=pyt, proxy_name=proxy_name, nptype=npt, 
                       nptypes=npts, ctype_nopred=ct_nopred, 
                       cytype_nopred=cyt_nopred,)
    if 1 == len(c2pyt) or ind == 0:
        decl = body = None
        rtn = c2pyt[0].format(**template_kw)
    elif ind == 1:
        decl = "cdef {0} {1}".format(cyt, proxy_name)
        body = c2pyt[1].format(**template_kw)
        rtn = proxy_name
    elif ind == 2:
        decl = "cdef {0} {1}".format(cyt, proxy_name)
        body = c2pyt[2].format(**template_kw)
        rtn = cache_name
        iscached = True
    if body is not None and 'np.npy_intp' in body:
        decl = decl or ''
        decl += "\ncdef np.npy_intp {proxy_name}_shape[1]".format(proxy_name=proxy_name)
    if decl is not None and body is not None:
        newdecl = '\n'+"\n".join([l for l in body.splitlines() if l.startswith('cdef')])
        body = "\n".join([l for l in body.splitlines() if not l.startswith('cdef')])
        proxy_in_newdecl = proxy_name in [l.split()[-1] for l in newdecl.splitlines() if 0 < len(l)]
        if proxy_in_newdecl:
            for d in decl.splitlines():
                if d.split()[-1] != proxy_name:
                    newdecl += '\n' + d
            decl = newdecl
        else:
            decl += newdecl
    return decl, body, rtn, iscached


_cython_py2c_conv = _LazyConverterDict({
    # Has tuple form of (body or return,  return or False)
    # base types
    'char': ('{var}_bytes = {var}.encode()', '(<char *> {var}_bytes)[0]'),
    ('char', '*'): ('{var}_bytes = {var}.encode()', '<char *> {var}_bytes'),
    'uchar': ('{var}_bytes = {var}.encode()', '(<unsigned char *> {var}_bytes)[0]'),
    ('uchar', '*'): ('{var}_bytes = {var}.encode()', '<unsigned char *> {var}_bytes'),
    'str': ('{var}_bytes = {var}.encode()', 'std_string(<char *> {var}_bytes)'),
    'int16': ('<short> {var}', False),
    'int32': ('<int> {var}', False),
    #('int32', '*'): ('&(<int> {var})', False),
    ('int32', '*'): ('cdef int {proxy_name}_ = {var}', '&{proxy_name}_'),
    'int64': ('<long long> {var}', False),
    'uint16': ('<unsigned short> {var}', False),
    'uint32': ('<{ctype}> long({var})', False),
    #'uint32': ('<unsigned long> {var}', False),
    #('uint32', '*'): ('cdef unsigned long {proxy_name}_ = {var}', '&{proxy_name}_'),
    ('uint32', '*'): ('cdef unsigned int {proxy_name}_ = {var}', '&{proxy_name}_'),
    'uint64': ('<unsigned long long> {var}', False),
    'float32': ('<float> {var}', False),
    'float64': ('<double> {var}', False),
    'float128': ('<long double> {var}', False),
    'complex128': ('{extra_types}py2c_complex({var})', False),
    'bool': ('<bint> {var}', False),
    'void': ('NULL', False),
    ('void', '*'): ('NULL', False),
    'file': ('{extra_types}PyFile_AsFile({var})[0]', False), 
    ('file', '*'): ('{extra_types}PyFile_AsFile({var})', False), 
    # template types
    'map': ('{proxy_name} = {pytype}({var}, not isinstance({var}, {cytype}))',
            '{proxy_name}.map_ptr[0]'),
    'dict': ('dict({var})', False),
    'pair': ('{proxy_name} = {pytype}({var}, not isinstance({var}, {cytype}))',
             '{proxy_name}.pair_ptr[0]'),
    'set': ('{proxy_name} = {pytype}({var}, not isinstance({var}, {cytype}))', 
            '{proxy_name}.set_ptr[0]'),
    'vector': (('cdef int i\n'
                'cdef int {var}_size\n'
                'cdef {npctypes[0]} * {var}_data\n'
                '{var}_size = len({var})\n'
                'if isinstance({var}, np.ndarray) and (<np.ndarray> {var}).descr.type_num == {nptype}:\n'
                '    {var}_data = <{npctypes[0]} *> np.PyArray_DATA(<np.ndarray> {var})\n'
                '    {proxy_name} = {ctype}(<size_t> {var}_size)\n' 
                '    for i in range({var}_size):\n'
                '        {proxy_name}[i] = {var}_data[i]\n'
                'else:\n'
                '    {proxy_name} = {ctype}(<size_t> {var}_size)\n' 
                '    for i in range({var}_size):\n'
                '        {proxy_name}[i] = <{npctypes[0]}> {var}[i]\n'),
               '{proxy_name}'),     # FIXME There might be improvements here...
    ('vector', 'char', 0): ((
                'cdef int i\n'
                'cdef int {var}_size\n'
                'cdef {npctypes[0]} * {var}_data\n'
                '{var}_size = len({var})\n'
                'if isinstance({var}, np.ndarray) and (<np.ndarray> {var}).descr.type_num == <int> {nptype}:\n'
                '    {var}_data = <{npctypes[0]} *> np.PyArray_DATA(<np.ndarray> {var})\n'
                '    {proxy_name} = {ctype}(<size_t> {var}_size)\n' 
                '    for i in range({var}_size):\n'
                '        {proxy_name}[i] = {var}[i]\n'
                'else:\n'
                '    {proxy_name} = {ctype}(<size_t> {var}_size)\n' 
                '    for i in range({var}_size):\n'
                '        _ = {var}[i].encode()\n'
                '        {proxy_name}[i] = deref(<char *> _)\n'),
               '{proxy_name}'),
    TypeMatcher(('vector', MatchAny, '&')): ((
                'cdef int i\n'
                'cdef int {var}_size\n'
                'cdef {npctypes[0]} * {var}_data\n'
                '{var}_size = len({var})\n'
                'if isinstance({var}, np.ndarray) and (<np.ndarray> {var}).descr.type_num == {nptype}:\n'
                '    {var}_data = <{npctypes[0]} *> np.PyArray_DATA(<np.ndarray> {var})\n'
                '    {proxy_name} = {ctype_nopred}(<size_t> {var}_size)\n' 
                '    for i in range({var}_size):\n'
                '        {proxy_name}[i] = {var}_data[i]\n'
                'else:\n'
                '    {proxy_name} = {ctype_nopred}(<size_t> {var}_size)\n' 
                '    for i in range({var}_size):\n'
                '        {proxy_name}[i] = <{npctypes[0]}> {var}[i]\n'),
                '{proxy_name}'),     # FIXME There might be improvements here...
    # refinement types
    'nucid': ('nucname.zzaaam({var})', False),
    'nucname': ('nucname.name({var})', False),
    TypeMatcher((('enum', MatchAny, MatchAny), '*')): \
        ('cdef int {proxy_name}_ = {var}', '&{proxy_name}_'),
    TypeMatcher((('int32', ('enum', MatchAny, MatchAny)), '*')): \
        ('cdef int {proxy_name}_ = {var}', '&{proxy_name}_'),
    })

_cython_py2c_conv[TypeMatcher((('vector', MatchAny, '&'), 'const'))] = \
    _cython_py2c_conv[TypeMatcher((('vector', MatchAny, 'const'), '&'))] = \
    _cython_py2c_conv[TypeMatcher(('vector', MatchAny, '&'))]
    
    

def _cython_py2c_conv_function_pointer(t):
    t = t[1]
    argnames = []
    argcts = []
    argdecls = []
    argbodys = []
    argrtns = []
    for n, argt in t[1][2]:
        argnames.append(n)
        decl, body, rtn, _ = cython_c2py(n, argt, proxy_name="c_" + n, cached=False)
        argdecls.append(decl)
        #argdecls.append("cdef {0} {1}".format(cython_pytype(argt), "c_" + n))
        argbodys.append(body)
        argrtns.append(rtn)
        argct = cython_ctype(argt)
        argcts.append(argct)
    rtnname = 'rtn'
    rtnprox = 'c_' + rtnname
    rtncall = 'call_' + rtnname
    while rtnname in argnames or rtnprox in argnames:
        rtnname += '_'
        rtnprox += '_'
    rtnct = cython_ctype(t[2][2])
    argdecls = _indent4(argdecls)
    argbodys = _indent4(argbodys)
    #rtndecl, rtnbody, rtnrtn = cython_py2c(rtnname, t[2][2], proxy_name=rtnprox)
    #rtndecl, rtnbody, rtnrtn = cython_py2c(rtnname, t[2][2], proxy_name=rtncall)
    rtndecl, rtnbody, rtnrtn = cython_py2c(rtncall, t[2][2], proxy_name=rtnprox)
    if rtndecl is None and rtnbody is None:
        rtnprox = rtnname
    rtndecl = _indent4([rtndecl])
    rtnbody = _indent4([rtnbody])
    s = """cdef {rtnct} {{proxy_name}}({arglist}):
{argdecls}
{rtndecl}
    global {{var}}
{argbodys}
    {rtncall} = {{var}}({pyarglist})
{rtnbody}
    return {rtnrtn}
"""
    arglist = ", ".join(["{0} {1}".format(*x) for x in zip(argcts, argnames)])
    pyarglist=", ".join(argrtns)
    s = s.format(rtnct=rtnct, arglist=arglist, argdecls=argdecls, rtndecl=rtndecl,
                 argbodys=argbodys, rtnprox=rtnprox, pyarglist=pyarglist,
                 rtnbody=rtnbody, rtnrtn=rtnrtn, rtncall=rtncall)
    return s, False

_cython_py2c_conv['function_pointer'] = _cython_py2c_conv_function_pointer


@_memoize
def cython_py2c(name, t, inst_name=None, proxy_name=None):
    """Given a varibale name and type, returns cython code (declaration, body, 
    and return statement) to convert the variable from Python to C/C++."""
    t = canon(t)
    if isinstance(t, basestring) or 0 == t[-1] or isrefinement(t[-1]):
        last = ''
    elif isinstance(t[-1], int):
        last = ' [{0}]'.format(t[-1])
    else:
        last = ' ' + t[-1]
    tkey = t
    tinst = None
    while tkey not in _cython_py2c_conv and not isinstance(tkey, basestring):
        tinst = tkey
        tkey = tkey[1] if (0 < len(tkey) and isrefinement(tkey[1])) else tkey[0]
    if tkey not in _cython_py2c_conv:
        tkey = t
        while tkey not in _cython_py2c_conv and not isinstance(tkey, basestring):
            tkey = tkey[0]
    py2ct = _cython_py2c_conv[tkey]
    if callable(py2ct):
        _cython_py2c_conv[t] = py2ct(t)
        py2ct = _cython_py2c_conv[t]
    if py2ct is NotImplemented or py2ct is None:
        raise NotImplementedError('conversion from Python to C/C++ for ' + \
                                  str(t) + ' has not been implemented.')
    body_template, rtn_template = py2ct
    ct = cython_ctype(t)
    cyt = cython_cytype(t)
    pyt = cython_pytype(t)
    npt = cython_nptype(t)
    npct = cython_ctype(npt)
    npts = cython_nptype(t, depth=1)
    npcts = [npct] if isinstance(npts, basestring) else _maprecurse(cython_ctype, npts)
    t_nopred = strip_predicates(t)
    ct_nopred = cython_ctype(t_nopred)
    cyt_nopred = cython_cytype(t_nopred)
    var = name if inst_name is None else "{0}.{1}".format(inst_name, name)
    proxy_name = "{0}_proxy".format(name) if proxy_name is None else proxy_name
    template_kw = dict(var=var, proxy_name=proxy_name, pytype=pyt, cytype=cyt, 
                       ctype=ct, last=last, nptype=npt, npctype=npct, 
                       nptypes=npts, npctypes=npcts, ctype_nopred=ct_nopred,
                       cytype_nopred=cyt_nopred)
    nested = False
    if isdependent(tkey):
        tsig = [ts for ts in refined_types if ts[0] == tkey][0]
        for ts, ti in zip(tsig[1:], tinst[1:]):
            if isinstance(ts, basestring):
                template_kw[ts] = cython_ctype(ti)
            else:
                template_kw[ti[0]] = ti[2]
        vartype = refined_types[tsig]
        if vartype in tsig[1:]:
            vartype = tinst[tsig.index(vartype)][1]
        if isrefinement(vartype):
            nested = True
            vdecl, vbody, vrtn = cython_py2c(var, vartype)
            template_kw['var'] = vrtn
    body_filled = body_template.format(**template_kw)
    if rtn_template:
        if '{ctype}'in body_template:
            deft = ct 
        elif '{ctype_nopred}'in body_template:
            deft = ct_nopred
        elif '{cytype_nopred}'in body_template:
            deft = cyt_nopred
        else:
            deft = cyt
        decl = "cdef {0} {1}".format(deft, proxy_name)
        body = body_filled
        rtn = rtn_template.format(**template_kw)
        decl += '\n'+"\n".join([l for l in body.splitlines() if l.startswith('cdef')])
        body = "\n".join([l for l in body.splitlines() if not l.startswith('cdef')])
    else:
        decl = body = None
        rtn = body_filled
    if nested:
        decl = '' if decl is None else decl
        vdecl = '' if vdecl is None else vdecl
        decl = (vdecl + '\n' + decl).strip()
        decl = None if 0 == len(decl) else decl
        body = '' if body is None else body
        vbody = '' if vbody is None else vbody
        body = (vbody + '\n' + body).strip()
        body = None if 0 == len(body) else body
    return decl, body, rtn
 


######################  Some utility functions for the typesystem #############

@_memoize
def _ensure_importable(x):
    if isinstance(x, basestring) or x is None:
        r = ((x,),)
    elif isinstance(x, Iterable) and (isinstance(x[0], basestring) or x[0] is None):
        r = (x,)
    else:
        r = x
    return r

def register_class(name=None, template_args=None, cython_c_type=None, 
                   cython_cimport=None, cython_cy_type=None, cython_py_type=None,
                   cython_template_class_name=None, cython_cyimport=None, 
                   cython_pyimport=None, cython_c2py=None, cython_py2c=None):
    """Classes are user specified types.  This function will add a class to 
    the type system so that it may be used normally with the rest of the 
    type system.

    """
    # register the class name
    isbase = True
    if template_args is None: 
        base_types.add(name)  # normal class        
    elif isinstance(template_args, Sequence):
        if 0 == len(template_args):
            base_types.add(name)  # normal class
        elif isinstance(template_args, basestring):
            _raise_type_error(name)
        else:
            template_types[name] = tuple(template_args)  # templated class...
            isbase = False

    # Register with Cython C/C++ types
    if (cython_c_type is not None):
        _cython_ctypes[name] = cython_c_type
    if (cython_cy_type is not None):
        _cython_cytypes[name] = cython_cy_type
    if (cython_py_type is not None):
        _cython_pytypes[name] = cython_py_type

    if (cython_cimport is not None):
        cython_cimport = _ensure_importable(cython_cimport)
        _cython_cimports[name] = cython_cimport
    if (cython_cyimport is not None):
        cython_cyimport = _ensure_importable(cython_cyimport)
        _cython_cyimports[name] = cython_cyimport
    if (cython_pyimport is not None):
        cython_pyimport = _ensure_importable(cython_pyimport)
        _cython_pyimports[name] = cython_pyimport

    if (cython_c2py is not None):
        if isinstance(cython_c2py, basestring):
            cython_c2py = (cython_c2py,)
        cython_c2py = None if cython_c2py is None else tuple(cython_c2py)
        _cython_c2py_conv[name] = cython_c2py
    if (cython_py2c is not None):
        if isinstance(cython_py2c, basestring):
            cython_py2c = (cython_py2c, False)
        _cython_py2c_conv[name] = cython_py2c
    if (cython_template_class_name is not None):
        _cython_classnames[name] = cython_template_class_name


def deregister_class(name):
    """This function will remove a previously registered class from the type system.
    """
    isbase = name in base_types
    if not isbase and name not in template_types:
        _raise_type_error(name)

    if isbase:
        base_types.remove(name)
    else:
        template_types.pop(name, None)

    _cython_ctypes.pop(name, None)
    _cython_cytypes.pop(name, None)
    _cython_pytypes.pop(name, None)
    _cython_cimports.pop(name, None)
    _cython_cyimports.pop(name, None)
    _cython_pyimports.pop(name, None)

    _cython_c2py_conv.pop(name, None)
    _cython_py2c_conv.pop(name, None)
    _cython_classnames.pop(name, None)

    # clear all caches
    funcs = [isdependent, isrefinement, _resolve_dependent_type, canon, 
             cython_ctype, cython_cimport_tuples, cython_cimports, _fill_cycyt,
             cython_cytype, _fill_cypyt, cython_pytype, cython_c2py, cython_py2c]
    for f in funcs:
        f.cache.clear()


def register_refinement(name, refinementof, cython_cimport=None, cython_cyimport=None, 
                        cython_pyimport=None, cython_c2py=None, cython_py2c=None):
    """This function will add a refinement to the type system so that it may be used 
    normally with the rest of the type system.
    """
    refined_types[name] = refinementof

    cyci = _ensure_importable(cython_cimport)
    _cython_cimports[name] = _cython_cimports[name] = cyci

    cycyi = _ensure_importable(cython_cyimport)
    _cython_cyimports[name] = _cython_cyimports[name] = cycyi

    cypyi = _ensure_importable(cython_pyimport)
    _cython_pyimports[name] = _cython_pyimports[name] = cypyi

    if isinstance(cython_c2py, basestring):
        cython_c2py = (cython_c2py,)
    cython_c2py = None if cython_c2py is None else tuple(cython_c2py)
    if cython_c2py is not None:
        _cython_c2py_conv[name] = cython_c2py

    if isinstance(cython_py2c, basestring):
        cython_py2c = (cython_py2c, False)
    if cython_py2c is not None:
        _cython_py2c_conv[name] = cython_py2c


def deregister_refinement(name):
    """This function will remove a previously registered refinement from the type
    system.
    """
    refined_types.pop(name, None)
    _cython_c2py_conv.pop(name, None)
    _cython_py2c_conv.pop(name, None)
    _cython_cimports.pop(name, None)
    _cython_cyimports.pop(name, None)
    _cython_pyimports.pop(name, None)


def register_specialization(t, cython_c_type=None, cython_cy_type=None, 
                            cython_py_type=None, cython_cimport=None, 
                            cython_cyimport=None, cython_pyimport=None):
    """This function will add a template specialization so that it may be used 
    normally with the rest of the type system.
    """
    t = canon(t)
    if cython_c_type is not None:
        _cython_ctypes[t] = cython_c_type
    if cython_cy_type is not None:
        _cython_cytypes[t] = cython_cy_type
    if cython_py_type is not None:
        _cython_pytypes[t] = cython_py_type
    if cython_cimport is not None:
        _cython_cimports[t] = cython_cimport
    if cython_cyimport is not None:
        _cython_cyimports[t] = cython_cyimport
    if cython_pyimport is not None:
        _cython_pyimports[t] = cython_pyimport

def deregister_specialization(t):
    """This function will remove previously registered template specialization."""
    t = canon(t)
    _cython_ctypes.pop(t, None)
    _cython_cytypes.pop(t, None)
    _cython_pytypes.pop(t, None)
    _cython_cimports.pop(t, None)
    _cython_cyimports.pop(t, None)
    _cython_pyimports.pop(t, None)


def register_numpy_dtype(t, cython_cimport=None, cython_cyimport=None, cython_pyimport=None):
    """This function will add a type to the system as numpy dtype that lives in
    the stlcontainers module.
    """
    t = canon(t)
    if t in _numpy_types:
        return
    varname = cython_variablename(t)[1]
    _numpy_types[t] = '{stlcontainers}xd_' + varname + '.num'
    type_aliases[_numpy_types[t]] = t
    type_aliases['xd_' + varname] = t
    type_aliases['xd_' + varname + '.num'] = t
    type_aliases['{stlcontainers}xd_' + varname] = t
    type_aliases['{stlcontainers}xd_' + varname + '.num'] = t
    if cython_cimport is not None:
        x = _ensure_importable(_cython_cimports._d.get(t, None))
        x = x + _ensure_importable(cython_cimport)
        _cython_cimports[t] = x
    # cython imports
    x = (('{stlcontainers}',),)
    x = x + _ensure_importable(_cython_cyimports._d.get(t, None))
    x = x + _ensure_importable(cython_cyimport)
    _cython_cyimports[t] = x
    # python imports
    x = (('{stlcontainers}',),)
    x = x + _ensure_importable(_cython_pyimports._d.get(t, None))
    x = x + _ensure_importable(cython_pyimport)
    _cython_pyimports[t] = x


#################### Type system helpers #######################################

def clearmemo():
    """Clears all function memoizations in this module."""
    for x in globals().values():
        if callable(x) and hasattr(x, 'cache'):
            x.cache.clear()

@contextmanager
def swap_stlcontainers(s):
    """A context manager for temporarily swapping out the STLCONTAINERS value
    with a new value and replacing the original value before exiting."""
    global STLCONTAINERS
    old = STLCONTAINERS
    STLCONTAINERS = s
    clearmemo()
    yield
    clearmemo()
    STLCONTAINERS = old

def _undot_class_name(name, d):
    value = d[name]
    if '.' not in value:
        return ''
    v1, v2 = value.rsplit('.', 1)
    d[name] = v2
    return v1

def _redot_class_name(name, d, value):
    if 0 == len(value):
        return
    d[name] = value + '.' + d[name]

@contextmanager
def local_classes(classnames, typesets=frozenset(['cy', 'py'])):
    """A context manager for making sure the given classes are local."""
    saved = {}
    for name in classnames:
        if 'c' in typesets:
            saved[name, 'c'] = _undot_class_name(name, _cython_ctypes)
        if 'cy' in typesets:
            saved[name, 'cy'] = _undot_class_name(name, _cython_cytypes)
        if 'py' in typesets:
            saved[name, 'py'] = _undot_class_name(name, _cython_pytypes)
    clearmemo()
    yield
    for name in classnames:
        if 'c' in typesets:
            _redot_class_name(name, _cython_ctypes, saved[name, 'c'])
        if 'cy' in typesets:
            _redot_class_name(name, _cython_cytypes, saved[name, 'cy'])
        if 'py' in typesets:
            _redot_class_name(name, _cython_pytypes, saved[name, 'py'])
    clearmemo()

_indent4 = lambda x: '' if x is None else "\n".join(["    " + l for l in "\n".join(
                     [xx for xx in x if xx is not None]).splitlines()])

def _maprecurse(f, x):
    if not isinstance(x, list):
        return [f(x)]
    #return [_maprecurse(f, y) for y in x]
    l = []
    for y in x:
        l += _maprecurse(f, y)
    return l
     
