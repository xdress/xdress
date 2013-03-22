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
functions in when doing code generation or type verification.

The last kind of type are known as **dependent types** or **template types**, 
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

Type System API
===============

"""
import functools
from collections import Sequence, Set, Iterable, MutableMapping

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


base_types = set(['char', 'str', 'int32', 'int64', 'uint32', 'uint64', 'float32', 
                  'float64', 'complex128', 'void', 'bool'])
"""Base types in the type system."""

type_aliases = {
    'i': 'int32',
    'i4': 'int32',
    'i8': 'int64',
    'int': 'int32',
    'ui': 'uint32',
    'ui4': 'uint32',
    'ui8': 'uint64',
    'uint': 'uint32',
    'f': 'float64',
    'f4': 'float32',
    'f8': 'float64',
    'float': 'float64',
    'complex': 'complex128',
    'b': 'bool',
    'v': 'void',
    's': 'str',
    'string': 'str',
    # 'c' has char / complex ambiquity, not included
    'NPY_BYTE': 'char',
    'NPY_STRING': 'str',
    'NPY_INT32': 'int32',
    'NPY_UINT32': 'uint32',
    'NPY_FLOAT32': 'float32',
    'NPY_FLOAT64': 'float64',
    'NPY_COMPLEX128': 'complex128',
    'NPY_BOOL': 'bool',
    'NPY_VOID': 'void',
    'NPY_OBJECT': 'void',
    'np.NPY_BYTE': 'char',
    'np.NPY_STRING': 'str',
    'np.NPY_INT32': 'int32',
    'np.NPY_UINT32': 'uint32',
    'np.NPY_FLOAT32': 'float32',
    'np.NPY_FLOAT64': 'float64',
    'np.NPY_COMPLEX128': 'complex128',
    'np.NPY_BOOL': 'bool',
    'np.NPY_VOID': 'void',
    'np.NPY_OBJECT': 'void',
    }
"""Aliases that may be used to subsitute one type name for another."""

template_types = {
    'map': ('key_type', 'value_type'),
    'dict': ('key_type', 'value_type'),
    'pair': ('key_type', 'value_type'),
    'set': ('value_type',),
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
    }
"""This is a mapping from refinement type names to the parent types.
The parent types may either be base types, compound types, template 
types, or other refined types!"""



_humannames = {
    'char': 'character',
    'str': 'string',
    'int32': 'integer',
    'uint32': 'unsigned integer',
    'float32': 'float',
    'float64': 'double',
    'complex128': 'complex',
    'dict': 'map of ({key_type}, {value_type}) items',
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
    istemplated = any([isinstance(x, basestring) for x in depkey[1:]])
    if tinst is None:
        return depkey
    elif istemplated:
        assert len(tinst) == len(depkey)
        typemap = {k: tinst[i] for i, k in enumerate(depkey[1:], 1) \
                                                    if isinstance(k, basestring)}
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
            if t0 in template_types:
                templen = len(template_types[t0])
                last_val = 0 if tlen == 1 + templen else t[-1]
                filledt = [t0] + [canon(tt) for tt in t[1:1+templen]] + [last_val]
                return tuple(filledt)
            elif isdependent(t0):
                return _resolve_dependent_type(t0, t)
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

#################### Type System Above This Line ##########################

EXTRA_TYPES = 'xdress_extra_types'

STLCONTAINERS = 'stlcontainers' 

_ensuremoddot = lambda x: x + '.' if x is not None and 0 < len(x) else ''

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
            value = value.replace('{' + k + '}', v)
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
        kw = {'extra_types': _ensuremoddot(EXTRA_TYPES),
              'stlcontainers': _ensuremoddot(STLCONTAINERS),}
        newvalue = tuple(tuple(x.format(**kw) or None for x in imp) for imp in value)
        return newvalue

    def __setitem__(self, key, value):
        self._d[key] = value

    def __delitem__(self, key):
        del self._d[key]

class _LazyConverterDict(MutableMapping):
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

    def __delitem__(self, key):
        del self._d[key]

#########################   Cython Functions   ################################

_cython_ctypes = _LazyConfigDict({
    'char': 'char',
    'str': 'std_string',
    'int32': 'int',
    'uint32': '{extra_types}uint',  # 'unsigned int'
    'float32': 'float',
    'float64': 'double',
    'complex128': '{extra_types}complex_t',
    'bool': 'bint',
    'void': 'void', 
    'map': 'cpp_map',
    'dict': 'dict',
    'pair': 'cpp_pair',
    'set': 'cpp_set',
    'vector': 'cpp_vector',
    })

@_memoize
def cython_ctype(t):
    """Given a type t, returns the cooresponding Cython C type declaration."""
    t = canon(t)
    if isinstance(t, basestring):
        if  t in base_types:
            return _cython_ctypes[t]
    # must be tuple below this line
    tlen = len(t)
    if 2 == tlen:
        if 0 == t[1]:
            return cython_ctype(t[0])
        elif isrefinement(t[1]):
            return cython_ctype(t[0])
        else:
            last = '[{0}]'.format(t[-1]) if isinstance(t[-1], int) else t[-1]
            return cython_ctype(t[0]) + ' {0}'.format(last)
    elif 3 <= tlen:
        assert t[0] in template_types
        assert len(t) == len(template_types[t[0]]) + 2
        template_name = _cython_ctypes[t[0]]
        assert template_name is not NotImplemented
        template_filling = ', '.join([cython_ctype(x) for x in t[1:-1]])
        cyct = '{0}[{1}]'.format(template_name, template_filling)
        if 0 != t[-1]:
            last = '[{0}]'.format(t[-1]) if isinstance(t[-1], int) else t[-1]
            cyct += ' {0}'.format(last)
        return cyct


_cython_cimports = _LazyImportDict({
    'char': (None,),
    'str': (('libcpp.string', 'string', 'std_string'),),
    'int32': (None,),
    'uint32': (('{extra_types}'),),  # 'unsigned int'
    'float32': (None,),
    'float64': (None,),
    'complex128': (('{extra_types}'),),
    'bool': (None,), 
    'void': (None,), 
    'map': (('libcpp.map', 'map', 'cpp_map'),),
    'dict': (None,),
    'pair': (('libcpp.utility', 'pair', 'cpp_pair'),),
    'set': (('libcpp.set', 'set', 'cpp_set'),),
    'vector': (('libcpp.vector', 'vector', 'cpp_vector'),),
    'nucid': (('pyne', 'cpp_nucname'),),
    'nucname': (('pyne', 'cpp_nucname'), ('libcpp.string', 'string', 'std_string')),
    })

_cython_cyimports = _LazyImportDict({
    'char': (None,),
    'str': (None,),
    'int32': (None,),
    'uint32': (None,),
    'float32': (None,),
    'float64': (None,),
    'complex128': (('{extra_types}',),),  # for py2c_complex()
    'bool': (None,), 
    'void': (None,), 
    'map': (('{stlcontainers}',),),
    'dict': (None,),
    'pair': (('{stlcontainers}',),), 
    'set': (('{stlcontainers}',),),
    'vector': (('numpy', 'as', 'np'),),
    'nucid': (('pyne', 'nucname'),),
    'nucname': (('pyne', 'nucname'),),
    })

@_memoize
def cython_cimport_tuples(t, seen=None, inc=frozenset(['c', 'cy'])):
    """Given a type t, and possibily previously seen cimport tuples (set), 
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
            seen.update(_cython_cimports.get(t[0], (None,)))
            seen.update(_cython_cimports.get(t[1], (None,)))
        if 'cy' in inc:
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
    """Retuns the cimport lines associtated with a type or a set of seen tuples.
    """
    if not isinstance(x, Set):
        x = cython_cimport_tuples(x, inc=inc)
    return set([_cython_cimport_cases[len(tup)](tup) for tup in x])



_cython_pyimports = _LazyImportDict({
    'char': (None,),
    'str': (None,),
    'int32': (None,),
    'uint32': (None,),
    'float32': (None,),
    'float64': (None,),
    'complex128': (None,),
    'bool': (None,), 
    'void': (None,), 
    'map': (('{stlcontainers}',),),
    'dict': (None,),
    'pair': (('{stlcontainers}',),),
    'set': (('{stlcontainers}',),),
    'vector': (('numpy', 'as', 'np'),),
    'nucid': (('pyne', 'nucname'),),
    'nucname': (('pyne', 'nucname'),),
    })

@_memoize
def cython_import_tuples(t, seen=None):
    """Given a type t, and possibily previously seen import tuples (set), 
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
    """Retuns the import lines associtated with a type or a set of seen tuples.
    """
    if not isinstance(x, Set):
        x = cython_import_tuples(x)
    return set([_cython_import_cases[len(tup)](tup) for tup in x])


_cython_cytypes = _LazyConfigDict({
    'char': 'char',
    'str': 'char *',
    'int32': 'int',
    'uint32': 'long',  # 'unsigned int'
    'float32': 'float',
    'float64': 'float',
    'complex128': 'object',
    'bool': 'bint',
    'void': 'void',
    'map': '{stlcontainers}_Map{key_type}{value_type}',
    'dict': 'dict',
    'pair': '{stlcontainers}_Pair{value_type}',
    'set': '{stlcontainers}_Set{value_type}',
    'vector': 'np.ndarray',
    })

_cython_functionnames = _LazyConfigDict({
    # base types
    'char': 'char',
    'str': 'str',
    'int32': 'int',
    'uint32': 'uint',
    'float32': 'float',
    'float64': 'double',
    'complex128': 'complex',
    'bool': 'bool',
    'void': 'void',
    # template types
    'map': 'map_{key_type}_{value_type}',
    'dict': 'dict',
    'pair': 'pair_{key_type}_{value_type}',
    'set': 'set_{value_type}',
    'vector': 'vector_{value_type}',    
    'nucid': 'nucid', 
    'nucname': 'nucname',
    })

@_memoize
def cython_functionname(t, cycyt=None):
    """Computes function names for cython types."""
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
            val, _ = cython_functionname(x, _cython_functionnames[x[0]])
        d[key] = val
    return t, cycyt.format(**d)


_cython_classnames = _LazyConfigDict({
    # base types
    'char': 'Char',
    'str': 'Str',
    'int32': 'Int',
    'uint32': 'UInt',
    'float32': 'Float',
    'float64': 'Double',
    'complex128': 'Complex',
    'bool': 'Bool',
    'void': 'Void',
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
            val, _ = _fill_cycyt(x, _cython_classnames[x[0]])
        d[key] = val
    return t, cycyt.format(**d)
    

@_memoize
def cython_cytype(t):
    """Given a type t, returns the cooresponding Cython type."""
    t = canon(t)
    if isinstance(t, basestring):
        if  t in base_types:
            return _cython_cytypes[t]
    # must be tuple below this line
    tlen = len(t)
    if 2 == tlen:
        if 0 == t[1]:
            return cython_cytype(t[0])
        elif isrefinement(t[1]):
            return cython_cytype(t[0])
        else:
            last = '[{0}]'.format(t[-1]) if isinstance(t[-1], int) else t[-1]
            return cython_cytype(t[0]) + ' {0}'.format(last)
    elif 3 <= tlen:
        if t in _cython_cytypes:
            return _cython_cytypes[t]
        assert t[0] in template_types
        assert len(t) == len(template_types[t[0]]) + 2
        template_name = _cython_cytypes[t[0]]
        assert template_name is not NotImplemented        
        cycyt = _cython_cytypes[t[0]]
        cycyt, t = _fill_cycyt(cycyt, t)
        if 0 != t[-1]:
            last = '[{0}]'.format(t[-1]) if isinstance(t[-1], int) else t[-1]
            cycyt += ' {0}'.format(last)
        return cycyt


_cython_pytypes = _LazyConfigDict({
    'char': 'str',
    'str': 'str',
    'int32': 'int',
    'uint32': 'int',  # 'unsigned int'
    'float32': 'float',
    'float64': 'float',
    'complex128': 'object',
    'bool': 'bool',
    'void': 'object',
    'map': '{stlcontainers}Map{key_type}{value_type}',
    'dict': 'dict',
    'pair': '{stlcontainers}Pair{value_type}',
    'set': '{stlcontainers}Set{value_type}',
    'vector': '{stlcontainers}Vector{value_type}',
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
    """Given a type t, returns the cooresponding Python type."""
    t = canon(t)
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




_numpy_types = {
    'char': 'np.NPY_BYTE',
    'str': 'np.NPY_STRING',
    'int32': 'np.NPY_INT32',
    'uint32': 'np.NPY_UINT32',
    'float32': 'np.NPY_FLOAT32',
    'float64': 'np.NPY_FLOAT64',
    'complex128': 'np.NPY_COMPLEX128',
    'bool': 'np.NPY_BOOL',
    'void': 'np.NPY_VOID',     
    }

@_memoize
def cython_nptype(t):
    """Given a type t, returns the cooresponding NumPy type."""
    t = canon(t)
    if isinstance(t, basestring):
        return _numpy_types.get(t, 'np.NPY_OBJECT')
    # must be tuple below this line
    tlen = len(t)
    if 2 == tlen:
        if 0 == t[1]:
            return cython_nptype(t[0])
        elif isrefinement(t[1]):
            return cython_nptype(t[0])
        else:
            # FIXME last is ignored for strings, but what about other types?
            #last = '[{0}]'.format(t[-1]) if isinstance(t[-1], int) else t[-1]
            #return cython_pytype(t[0]) + ' {0}'.format(last)
            return cython_nptype(t[0])
    elif 3 <= tlen:
        return _numpy_types.get(t, 'np.NPY_OBJECT')

_cython_c2py_conv = _LazyConverterDict({
    # Has tuple form of (copy, [view, [cached_view]])
    # base types
    'char': ('str(<char *> {var})',),
    'str': ('str(<char *> {var}.c_str())',),
    'int32': ('int({var})',),
    'uint32': ('int({var})',),
    'float32': ('float({var})',),
    'float64': ('float({var})',),
    'complex128': ('complex(float({var}.re), float({var}.im))',),
    'bool': ('bool({var})',),
    'void': ('None',),
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
                '{proxy_name} = np.PyArray_SimpleNewFromData(1, {var}_shape, {nptype}, &{var}[0])\n'
                '{proxy_name} = np.PyArray_Copy({proxy_name})\n'),
               ('{proxy_name}_shape[0] = <np.npy_intp> {var}.size()\n'
                '{proxy_name} = np.PyArray_SimpleNewFromData(1, {proxy_name}_shape, {nptype}, &{var}[0])\n'
                #'{proxy_name} = np.PyArray_Copy({proxy_name})\n'
                ),
               ('if {cache_name} is None:\n'
                '    {proxy_name}_shape[0] = <np.npy_intp> {var}.size()\n'
                '    {proxy_name} = np.PyArray_SimpleNewFromData(1, {proxy_name}_shape, {nptype}, &{var}[0])\n'
                '    {cache_name} = {proxy_name}\n'
                )),
    'nucid': ('nucname.zzaaam({var})',),
    'nucname': ('nucname.name({var})',),
    })


from_pytypes = {
    'str': ['basestring'],
    'int32': ['int', 'long'],
    'uint32': ['int', 'long'],
    'float32': ['float', 'int', 'long'],
    'float64': ['float', 'int', 'long'],
    'complex128': ['complex', 'float'],
    'set': ['set', 'list', 'basestring', 'tuple'],
    'vector': ['list', 'tuple', 'np.ndarray'],
    }


@_memoize
def cython_c2py(name, t, view=True, cached=True, inst_name=None, proxy_name=None, 
                cache_name=None, cache_prefix='self', existing_name=None):
    """Given a varibale name and type, returns cython code (declaration, body, 
    and return statements) to convert the variable from C/C++ to Python."""
    tkey = canon(t)
    while not isinstance(tkey, basestring):
        tkey = tkey[0]
    c2pyt = _cython_c2py_conv[tkey]
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
    if istemplate(t) and (2 == len(t) or 3 == len(t) and t[-1] == 0):
        npt = cython_nptype(t[1])
    else:
        npt = cython_nptype(t)
    var = name if inst_name is None else "{0}.{1}".format(inst_name, name)
    cache_name = "_{0}".format(name) if cache_name is None else cache_name
    cache_name = cache_name if cache_prefix is None else "{0}.{1}".format(cache_prefix, cache_name)
    proxy_name = "{0}_proxy".format(name) if proxy_name is None else proxy_name
    iscached = False
    if 1 == len(c2pyt) or ind == 0:
        decl = body = None
        rtn = c2pyt[0].format(var=var, ctype=ct, cytype=cyt, pytype=pyt, nptype=npt)
    elif ind == 1 and existing_name is None:
        decl = "cdef {0} {1}".format(cyt, proxy_name)
        body = c2pyt[1].format(var=var, ctype=ct, cytype=cyt, pytype=pyt, nptype=npt, 
                               proxy_name=proxy_name)
        rtn = proxy_name
    elif ind == 1 and existing_name is not None:
        decl = None
        body = c2pyt[1].format(var=existing_name, ctype=ct, cytype=cyt, pytype=pyt, 
                               nptype=npt, proxy_name=proxy_name)
        rtn = proxy_name
    elif ind == 2:
        decl = "cdef {0} {1}".format(cyt, proxy_name)
        body = c2pyt[2].format(var=var, cache_name=cache_name, ctype=ct, cytype=cyt, 
                               pytype=pyt, proxy_name=proxy_name, nptype=npt)
        rtn = cache_name
        iscached = True
    if body is not None and 'np.npy_intp' in body:
        decl = decl or ''
        decl += "\ncdef np.npy_intp {proxy_name}_shape[1]".format(proxy_name=proxy_name)
    return decl, body, rtn, iscached


_cython_py2c_conv = _LazyConverterDict({
    # Has tuple form of (body or return,  return or False)
    # base types
    'char': ('<char{last}> {var}', False),
    'str': ('std_string(<char *> {var})', False),
    'int32': ('{var}', False),
    'uint32': ('<{ctype}> long({var})', False),
    'float32': ('<float> {var}', False),
    'float64': ('<double> {var}', False),
    'complex128': ('{extra_types}py2c_complex({var})', False),
    'bool': ('<bint> {var}', False),
    'void': ('NULL', False),
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
                'cdef {npctype} * {var}_data\n'
                '{var}_size = len({var})\n'
                'if isinstance({var}, np.ndarray) and (<np.ndarray> {var}).descr.type_num == {nptype}:\n'
                '    {var}_data = <{npctype} *> np.PyArray_DATA(<np.ndarray> {var})\n'
                '    {proxy_name} = {ctype}(<size_t> {var}_size)\n' 
                '    for i in range({var}_size):\n'
                '        {proxy_name}[i] = {var}_data[i]\n'
                'else:\n'
                '    {proxy_name} = {ctype}(<size_t> {var}_size)\n' 
                '    for i in range({var}_size):\n'
                '        {proxy_name}[i] = <{npctype}> {var}[i]\n'),
               '{proxy_name}'),     # FIXME There might be imporvements here...
    # refinement types
    'nucid': ('nucname.zzaaam({var})', False),
    'nucname': ('nucname.name({var})', False),
    })

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
    while not isinstance(tkey, basestring):
        tinst = tkey
        tkey = tkey[1] if (0 < len(tkey) and isrefinement(tkey[1])) else tkey[0]
    py2ct = _cython_py2c_conv[tkey]
    if py2ct is NotImplemented or py2ct is None:
        raise NotImplementedError('conversion from Python to C/C++ for ' + \
                                  t + 'has not been implemented.')
    body_template, rtn_template = py2ct
    ct = cython_ctype(t)
    cyt = cython_cytype(t)
    pyt = cython_pytype(t)
    if istemplate(t) and 1 == len(template_types.get(tkey, ())):
        npt = cython_nptype(t[1])
    else:
        npt = cython_nptype(t)
    npct = cython_ctype(npt)
    var = name if inst_name is None else "{0}.{1}".format(inst_name, name)
    proxy_name = "{0}_proxy".format(name) if proxy_name is None else proxy_name
    template_kw = dict(var=var, proxy_name=proxy_name, pytype=pyt, cytype=cyt, 
                       ctype=ct, last=last, nptype=npt, npctype=npct)
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
        deft = ct if '{ctype}'in body_template else cyt
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

def register_class(name, template_args=None, cython_c_type=None, 
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
    if (cython_c_type is not None) or (cython_cy_type is not None):
        cython_cimport = _ensure_importable(cython_cimport)
        cython_cyimport = _ensure_importable(cython_cyimport)
        cython_pyimport = _ensure_importable(cython_pyimport)

        if isinstance(cython_c2py, basestring):
            cython_c2py = (cython_c2py,)
        cython_c2py = None if cython_c2py is None else tuple(cython_c2py)

        if isinstance(cython_py2c, basestring):
            cython_py2c = (cython_py2c, False)

        _cython_ctypes[name] = cython_c_type
        _cython_cytypes[name] = cython_cy_type
        _cython_pytypes[name] = cython_py_type
        _cython_cimports[name] = cython_cimport
        _cython_cyimports[name] = cython_cyimport
        _cython_pyimports[name] = cython_pyimport

        _cython_c2py_conv[name] = cython_c2py
        _cython_py2c_conv[name] = cython_py2c
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
