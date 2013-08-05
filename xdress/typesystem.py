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

Major Classes Overview
----------------------
Holistically, the following classes are important to type system:

* ``TypeSystem``:  This *is* the type system.
* ``TypeMatcher``: An imutable type for matching types against a pattern.
* ``MatchAny``: A singleton used to denote patterns.
* ``typestr``: Various string representations of a type as properties.

Type System API
===============

"""
from __future__ import print_function
import os
import io
import sys
from contextlib import contextmanager
from collections import Sequence, Set, Iterable, MutableMapping, Mapping
from numbers import Number
from pprint import pprint, pformat
import gzip
try:
    import cPickle as pickle
except ImportError:
    import pickle

from .utils import flatten, indent, memoize_method, infer_format

if sys.version_info[0] >= 3:
    basestring = str

class TypeSystem(object):
    """A class representing a type system.
    """

    datafields = set(['base_types', 'template_types', 'refined_types', 'humannames', 
        'extra_types', 'stlcontainers', 'type_aliases', 'cpp_types', 'numpy_types', 
        'from_pytypes', 'cython_ctypes', 'cython_cytypes', 'cython_pytypes', 
        'cython_cimports', 'cython_cyimports', 'cython_pyimports', 
        'cython_functionnames', 'cython_classnames', 'cython_c2py_conv', 
        'cython_py2c_conv'])

    def __init__(self, base_types=None, template_types=None, refined_types=None, 
                 humannames=None, extra_types='xdress_extra_types', 
                 stlcontainers='stlcontainers', type_aliases=None, cpp_types=None, 
                 numpy_types=None, from_pytypes=None, cython_ctypes=None, 
                 cython_cytypes=None, cython_pytypes=None, cython_cimports=None, 
                 cython_cyimports=None, cython_pyimports=None, 
                 cython_functionnames=None, cython_classnames=None, 
                 cython_c2py_conv=None, cython_py2c_conv=None, typestring=None):
        """Parameters
        ----------
        base_types : set of str, optional
            The base or primitive types in the type system.
        template_types : dict, optional
            Template types are types whose instantiations are based on meta-types.
            this dict maps their names to meta-type names in order.
        refined_types : dict, optional 
            This is a mapping from refinement type names to the parent types.
            The parent types may either be base types, compound types, template
            types, or other refined types!
        humannames : dict, optional
            The human readable names for types.
        extra_types : str, optional
            The name of the xdress extra types module.
        stlcontainers : str, optional
            The name of the xdress C++ standard library containers wrapper module.
        type_aliases : dict, optional
            Aliases that may be used to substitute one type name for another.
        cpp_types : dict, optional
            The C/C++ representation of the types.
        numpy_types : dict, optional
            NumPy's Cython representation of the types.
        from_pytypes : dict, optional
            List of Python types which match may be converted to these types.
        cython_ctypes : dict, optional
            Cython's C/C++ representation of the types.
        cython_cytypes : dict, optional
            Cython's Cython representation of the types.
        cython_pytypes : dict, optional
            Cython's Python representation of the types.
        cython_cimports : dict, optional
            A sequence of tuples representing cimports that are needed for Cython
            to represent C/C++ types.
        cython_cyimports : dict, optional
            A sequence of tuples representing cimports that are needed for Cython
            to represent Cython types.
        cython_pyimports : dict, optional
            A sequence of tuples representing imports that are needed for Cython
            to represent Python types.
        cython_functionnames : dict, optional
            Cython alternate name fragments used for mangling function and 
            variable names.  These should try to adhere to a lowercase_and_underscore
            convention.  These may contain template argument namess as part of a 
            format string, ie ``{'map': 'map_{key_type}_{value_type}'}``.
        cython_classnames : dict, optional
            Cython alternate name fragments used for mangling class names.  
            These should try to adhere to a CapCase convention.  These may contain 
            template argument namess as part of a format string, 
            ie ``{'map': 'Map{key_type}{value_type}'}``.
        cython_c2py_conv : dict, optional
            Cython convertors from C/C++ types to the representative Python types.
        cython_py2c_conv : dict, optional
            Cython convertors from Python types to the representative C/C++ types.
            Valuse are tuples with the form of ``(body or return, return or False)``.
        typestring : typestr or None, optional
            An type that is used to format types to strings in conversion routines.

        """
        self.base_types = base_types if base_types is not None else set(['char', 
            'uchar', 'str', 'int16', 'int32', 'int64', 'int128', 'uint16', 'uint32', 
            'uint64', 'uint128', 'float32', 'float64', 'float128', 'complex128', 
            'void', 'bool', 'type', 'file', 'exception'])
        self.template_types = template_types if template_types is not None else {
            'map': ('key_type', 'value_type'),
            'dict': ('key_type', 'value_type'),
            'pair': ('key_type', 'value_type'),
            'set': ('value_type',),
            'list': ('value_type',),
            'tuple': ('value_type',),
            'vector': ('value_type',),
            'enum': ('name', 'aliases'),
            'function': ('arguments', 'returns'),
            'function_pointer': ('arguments', 'returns'),
            }
        self.refined_types = refined_types if refined_types is not None else {
            'nucid': 'int32',
            'nucname': 'str',
            ('enum', ('name', 'str'), 
                     ('aliases', ('dict', 'str', 'int32', 0))): 'int32',
            ('function', ('arguments', ('list', ('pair', 'str', 'type'))),
                         ('returns', 'type')): 'void',
            ('function_pointer', ('arguments', ('list', ('pair', 'str', 'type'))),
                                 ('returns', 'type')): ('void', '*'),
            }
        self.humannames = humannames if humannames is not None else {
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
            'exception': 'exception',
            'dict': 'dict of ({key_type}, {value_type}) items',
            'map': 'map of ({key_type}, {value_type}) items',
            'pair': '({key_type}, {value_type}) pair',
            'set': 'set of {value_type}',
            'vector': 'vector [ndarray] of {value_type}',
            }
        self.extra_types = extra_types
        self.stlcontainers = stlcontainers
        self.type_aliases = _LazyConfigDict(type_aliases if type_aliases is not \
                                                                      None else {
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
            'double': 'float64',
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
            }, self)

        def cpp_types_function(t, ts):
            rtnct = ts.cpp_type(t[2][2])
            argcts = [ts.cpp_type(argt) for n, argt in t[1][2]]
            if argcts == ['void']:
                argcts = []
            return rtnct + " {type_name}(" + ", ".join(argcts) + ")"

        def cpp_types_function_pointer(t, ts):
            rtnct = ts.cpp_type(t[2][2])
            argcts = [ts.cpp_type(argt) for n, argt in t[1][2]]
            if argcts == ['void']:
                argcts = []
            return rtnct + " (*{type_name})(" + ", ".join(argcts) + ")"

        self.cpp_types = _LazyConfigDict(cpp_types if cpp_types is not None else {
            'char': 'char',
            'uchar': 'unsigned char',
            'str': 'std::string',
            'int16': 'short',
            'int32': 'int',
            'int64': 'long long',
            'uint16': 'unsigned short',
            'uint32': 'unsigned long',
            'uint64': 'unsigned long long',
            'float32': 'float',
            'float64': 'double',
            'float128': 'long double',
            'complex128': '{extra_types}complex_t',
            'bool': 'bool',
            'void': 'void', 
            'file': 'FILE',
            'exception': '{extra_types}exception',
            'map': 'std::map',
            'dict': 'std::map',
            'pair': 'std::pair',
            'set': 'std::set',
            'vector': 'std::vector',
            True: 'true',
            'true': 'true',
            'True': 'true',
            False: 'false',
            'false': 'false',
            'False': 'false',
            'function': cpp_types_function,
            'function_pointer': cpp_types_function_pointer,
            }, self)

        self.numpy_types = _LazyConfigDict(numpy_types if numpy_types is not None \
                                                                             else {
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
            }, self)

        self.from_pytypes = from_pytypes if from_pytypes is not None else {
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

        def cython_ctypes_function(t, ts):
            rtnct = ts.cython_ctype(t[2][2])
            argcts = [ts.cython_ctype(argt) for n, argt in t[1][2]]
            if argcts == ['void']:
                argcts = []
            return rtnct + " {type_name}(" + ", ".join(argcts) + ")"

        def cython_ctypes_function_pointer(t, ts):
            rtnct = ts.cython_ctype(t[2][2])
            argcts = [ts.cython_ctype(argt) for n, argt in t[1][2]]
            if argcts == ['void']:
                argcts = []
            return rtnct + " (*{type_name})(" + ", ".join(argcts) + ")"

        self.cython_ctypes = _LazyConfigDict(cython_ctypes if cython_ctypes is not \
                                                                         None else {
            'char': 'char',
            'uchar': '{extra_types}uchar',
            'str': 'std_string',
            'int16': 'short',
            'int32': 'int',
            'int64': '{extra_types}int64',
            'uint16': '{extra_types}uint16',
            'uint32': '{extra_types}uint32',
            'uint64': '{extra_types}uint64',
            'float32': 'float',
            'float64': 'double',
            'float128': '{extra_types}float128',
            'complex128': '{extra_types}complex_t',
            'bool': 'bint',
            'void': 'void',
            'file': 'c_file',
            'exception': '{extra_types}exception',
            'map': 'cpp_map',
            'dict': 'dict',
            'pair': 'cpp_pair',
            'set': 'cpp_set',
            'vector': 'cpp_vector',
            'function': cython_ctypes_function,
            'function_pointer': cython_ctypes_function_pointer,
            }, self)

        self.cython_cytypes = _LazyConfigDict(cython_cytypes if cython_cytypes \
                                              is not None else {
            'char': 'char',
            'uchar': 'unsigned char',
            'str': 'char *',
            'int16': 'short',
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
            'exception': '{extra_types}exception',
            'map': '{stlcontainers}_Map{key_type}{value_type}',
            'dict': 'dict',
            'pair': '{stlcontainers}_Pair{value_type}',
            'set': '{stlcontainers}_Set{value_type}',
            'vector': 'np.ndarray',
            'function': 'object',
            'function_pointer': 'object',
            }, self)

        self.cython_pytypes = _LazyConfigDict(cython_pytypes if cython_pytypes \
                                              is not None else {
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
            'exception': 'Exception',
            'bool': 'bool',
            'void': 'object',
            'map': '{stlcontainers}Map{key_type}{value_type}',
            'dict': 'dict',
            'pair': '{stlcontainers}Pair{value_type}',
            'set': '{stlcontainers}Set{value_type}',
            'vector': 'np.ndarray',
            }, self)

        def cython_cimports_functionish(t, ts, seen):
            seen.add(('cython.operator', 'dereference', 'deref'))
            for n, argt in t[1][2]:
                ts.cython_cimport_tuples(argt, seen=seen, inc=('c',))
            ts.cython_cimport_tuples(t[2][2], seen=seen, inc=('c',))

        self.cython_cimports = _LazyImportDict(cython_cimports if cython_cimports \
                                               is not None else {
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
            'exception': (('{extra_types}',),),
            'map': (('libcpp.map', 'map', 'cpp_map'),),
            'dict': (None,),
            'pair': (('libcpp.utility', 'pair', 'cpp_pair'),),
            'set': (('libcpp.set', 'set', 'cpp_set'),),
            'vector': (('libcpp.vector', 'vector', 'cpp_vector'),),
            'nucid': (('pyne', 'cpp_nucname'),),
            'nucname': (('pyne', 'cpp_nucname'), 
                        ('libcpp.string', 'string', 'std_string')),
            'function': cython_cimports_functionish,
            'function_pointer': cython_cimports_functionish,
            }, self)

        def cython_cyimports_functionish(t, ts, seen):
            for n, argt in t[1][2]:
                ts.cython_cimport_tuples(argt, seen=seen, inc=('cy',))
            ts.cython_cimport_tuples(t[2][2], seen=seen, inc=('cy',))

        self.cython_cyimports = _LazyImportDict(cython_cyimports if \
                                    cython_cyimports is not None else {
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
            'exception': (('{extra_types}',),),
            'map': (('{stlcontainers}',),),
            'dict': (None,),
            'pair': (('{stlcontainers}',),),
            'set': (('{stlcontainers}',),),
            'vector': (('numpy', 'as', 'np'), ('{stlcontainers}',)),
            'nucid': (('pyne', 'nucname'),),
            'nucname': (('pyne', 'nucname'),),
            'function': cython_cyimports_functionish,
            'function_pointer': cython_cyimports_functionish,
            }, self)

        def cython_pyimports_functionish(t, ts, seen):
            seen.add(('warnings',))
            for n, argt in t[1][2]:
                ts.cython_import_tuples(argt, seen=seen)
            ts.cython_import_tuples(t[2][2], seen=seen)

        self.cython_pyimports = _LazyImportDict(cython_pyimports if \
                                    cython_pyimports is not None else {
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
            'exception': (None,),
            'map': (('{stlcontainers}',),),
            'dict': (None,),
            'pair': (('{stlcontainers}',),),
            'set': (('{stlcontainers}',),),
            'vector': (('numpy', 'as', 'np'),),
            'nucid': (('pyne', 'nucname'),),
            'nucname': (('pyne', 'nucname'),),
            'function': cython_pyimports_functionish,
            'function_pointer': cython_pyimports_functionish,
            }, self)

        self.cython_functionnames = _LazyConfigDict(cython_functionnames if \
                                        cython_functionnames is not None else {
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
            'exception': 'exception',
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
            }, self)

        self.cython_classnames = _LazyConfigDict(cython_classnames if \
                                                 cython_classnames is not None else {
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
            'exception': 'Exception',
            # template types
            'map': 'Map{key_type}{value_type}',
            'dict': 'Dict',
            'pair': 'Pair{key_type}{value_type}',
            'set': 'Set{value_type}',
            'vector': 'Vector{value_type}',
            'nucid': 'Nucid',
            'nucname': 'Nucname',
            }, self)

        def cython_c2py_conv_function_pointer(t_, ts):
            """Wrap function pointers in C/C++ to Python functions."""
            t = t_[1]
            argnames = []
            argdecls = []
            argbodys = []
            argrtns = []
            for n, argt in t[1][2]:
                argnames.append(n)
                decl, body, rtn = ts.cython_py2c(n, argt, proxy_name="c_" + n)
                argdecls += decl.split('\n') if isinstance(decl,basestring) else [decl]
                argbodys += body.split('\n') if isinstance(body,basestring) else [body]
                argrtns += rtn.split('\n') if isinstance(rtn,basestring) else [rtn]
            rtnname = 'rtn'
            rtnprox = 'c_' + rtnname
            rtncall = 'c_call_' + rtnname
            while rtnname in argnames or rtnprox in argnames:
                rtnname += '_'
                rtnprox += '_'
            argdecls = indent(argdecls)
            argbodys = indent(argbodys)
            rtndecl, rtnbody, rtnrtn, _ = ts.cython_c2py(rtncall, t[2][2], 
                cached=False, proxy_name=rtnprox, existing_name=rtncall)
            if rtndecl is None and rtnbody is None:
                rtnprox = rtnname
            rtndecl = indent([rtndecl, 
                              "cdef {0} {1}".format(ts.cython_ctype(t[2][2]), 
                              rtncall)])
            rtnbody = indent(rtnbody)
            s = ('def {{proxy_name}}({arglist}):\n'
                 '{argdecls}\n'
                 '{rtndecl}\n'
                 '    if {{var}} == NULL:\n'
                 '        raise RuntimeError("{{var}} is NULL and may not be '
                                             'safely called!")\n'
                 '{argbodys}\n'
                 '    {rtncall} = {{var}}({carglist})\n'
                 '{rtnbody}\n')
            s = s.format(arglist=", ".join(argnames), argdecls=argdecls,
                         cvartypeptr=ts.cython_ctype(t_).format(type_name='cvartype'),
                         argbodys=argbodys, rtndecl=rtndecl, rtnprox=rtnprox, 
                         rtncall=rtncall, carglist=", ".join(argrtns), rtnbody=rtnbody)
            caches = 'if {cache_name} is None:\n' + indent(s)
            if t[2][2] != 'void':
                caches += "\n        return {rtnrtn}".format(rtnrtn=rtnrtn)
                caches += '\n    {cache_name} = {proxy_name}\n'
            return s, s, caches

        self.cython_c2py_conv = _LazyConverterDict(cython_c2py_conv if \
                                     cython_c2py_conv is not None else {
            # Has tuple form of (copy, [view, [cached_view]])
            # base types
            'char': ('chr(<int> {var})',),
            'uchar': ('chr(<unsigned int> {var})',),
            'str': ('bytes(<char *> {var}.c_str()).decode()',),
            ('str', '*'): ('bytes(<char *> {var}[0].c_str()).decode()',),
            'int16': ('int({var})',),
            ('int16', '*'): ('int({var}[0])',),
            'int32': ('int({var})',),
            ('int32', '*'): ('int({var}[0])',),
            'int64': ('int({var})',),
            ('int64', '*'): ('int({var}[0])',),
            'uint16': ('int({var})',),
            ('uint16', '*'): ('int({var}[0])',),
            'uint32': ('int({var})',),
            ('uint32', '*'): ('int({var}[0])',),
            'uint64': ('int({var})',),
            'float32': ('float({var})',),
            ('float32', '*'): ('float({var}[0])',),
            'float64': ('float({var})',),
            ('float64', '*'): ('float({var}[0])',),
            'float128': ('np.array({var}, dtype=np.float128)',),
            ('float128', '*'): ('np.array({var}[0], dtype=np.float128)',),
            'complex128': ('complex(float({var}.re), float({var}.im))',),
            ('complex128', '*'): ('complex(float({var}[0].re), float({var}[0].im))',),
            'bool': ('bool({var})',),
            ('bool', '*'): ('bool({var}[0])',),
            'void': ('None',),
            'file': ('{extra_types}PyFile_FromFile(&{var}, "{var}", "r+", NULL)',),
            ('file', '*'): (
                '{extra_types}PyFile_FromFile({var}, "{var}", "r+", NULL)',),
            # template types
            'map': ('{t.cython_pytype}({var})',
                   ('{proxy_name} = {t.cython_pytype}(False, False)\n'
                    '{proxy_name}.map_ptr = &{var}\n'),
                   ('if {cache_name} is None:\n'
                    '    {proxy_name} = {t.cython_pytype}(False, False)\n'
                    '    {proxy_name}.map_ptr = &{var}\n'
                    '    {cache_name} = {proxy_name}\n'
                    )),
            'dict': ('dict({var})',),
            'pair': ('{t.cython_pytype}({var})',
                     ('{proxy_name} = {t.cython_pytype}(False, False)\n'
                      '{proxy_name}.pair_ptr = &{var}\n'),
                     ('if {cache_name} is None:\n'
                      '    {proxy_name} = {t.cython_pytype}(False, False)\n'
                      '    {proxy_name}.pair_ptr = &{var}\n'
                      '    {cache_name} = {proxy_name}\n'
                      )),
            'set': ('{t.cython_pytype}({var})',
                   ('{proxy_name} = {t.cython_pytype}(False, False)\n'
                    '{proxy_name}.set_ptr = &{var}\n'),
                   ('if {cache_name} is None:\n'
                    '    {proxy_name} = {t.cython_pytype}(False, False)\n'
                    '    {proxy_name}.set_ptr = &{var}\n'
                    '    {cache_name} = {proxy_name}\n'
                    )),
            'vector': (
                ('{proxy_name}_shape[0] = <np.npy_intp> {var}.size()\n'
                 '{proxy_name} = np.PyArray_SimpleNewFromData(1, {var}_shape, {t.cython_nptypes[0]}, &{var}[0])\n'
                 '{proxy_name} = np.PyArray_Copy({proxy_name})\n'),
                ('{proxy_name}_shape[0] = <np.npy_intp> {var}.size()\n'
                 '{proxy_name} = np.PyArray_SimpleNewFromData(1, {proxy_name}_shape, {t.cython_nptypes[0]}, &{var}[0])\n'),
                ('if {cache_name} is None:\n'
                 '    {proxy_name}_shape[0] = <np.npy_intp> {var}.size()\n'
                 '    {proxy_name} = np.PyArray_SimpleNewFromData(1, {proxy_name}_shape, {t.cython_nptypes[0]}, &{var}[0])\n'
                 '    {cache_name} = {proxy_name}\n'
                )),
            ('vector', 'bool', 0): (  # C++ standard is silly here
                ('cdef int i\n'
                 '{proxy_name}_shape[0] = <np.npy_intp> {var}.size()\n'
                 '{proxy_name} = np.PyArray_SimpleNew(1, {proxy_name}_shape, {t.cython_nptypes[0]})\n'
                 'for i in range({proxy_name}_shape[0]):\n'
                 '    {proxy_name}[i] = {var}[i]\n'),
                ('cdef int i\n'
                 '{proxy_name}_shape[0] = <np.npy_intp> {var}.size()\n'
                 '{proxy_name} = np.PyArray_SimpleNew(1, {proxy_name}_shape, {t.cython_nptypes[0]})\n'
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
            # C/C++ chars are ints while Python Chars are length-1 strings
            ('vector', 'char', 0): (
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
            TypeMatcher((('int32', ('enum', MatchAny, MatchAny)), '*')): \
                                                                ('int({var}[0])',),
            # Strip const when going c -> py 
            TypeMatcher((MatchAny, 'const')): (
                lambda t, ts: ts.cython_c2py_getitem(t[0])),
            TypeMatcher(((MatchAny, 'const'), '&')) : (
                lambda t, ts: ts.cython_c2py_getitem((t[0][0], '&'))),
            TypeMatcher(((MatchAny, 'const'), '*')): (
                lambda t, ts: ts.cython_c2py_getitem((t[0][0], '*'))),
            'function_pointer': cython_c2py_conv_function_pointer,
            }, self)

        cython_py2c_conv_vector_ref = ((
            '# {var} is a {t.type}\n'
            'cdef int i{var}\n'
            'cdef int {var}_size\n'
            'cdef {t.cython_npctypes_nopred[0]} * {var}_data\n'
            '{var}_size = len({var})\n'
            'if isinstance({var}, np.ndarray) and (<np.ndarray> {var}).descr.type_num == {t.cython_nptype}:\n'
            '    {var}_data = <{t.cython_npctypes_nopred[0]} *> np.PyArray_DATA(<np.ndarray> {var})\n'
            '    {proxy_name} = {t.cython_ctype_nopred}(<size_t> {var}_size)\n'
            '    for i{var} in range({var}_size):\n'
            '        {proxy_name}[i{var}] = {var}_data[i{var}]\n'
            'else:\n'
            '    {proxy_name} = {t.cython_ctype_nopred}(<size_t> {var}_size)\n'
            '    for i{var} in range({var}_size):\n'
            '        {proxy_name}[i{var}] = <{t.cython_npctypes_nopred[0]}> {var}[i{var}]\n'),
            '{proxy_name}')     # FIXME There might be improvements here...

        def cython_py2c_conv_function_pointer(t, ts):
            t = t[1]
            argnames = []
            argcts = []
            argdecls = []
            argbodys = []
            argrtns = []
            for n, argt in t[1][2]:
                argnames.append(n)
                decl, body, rtn, _ = ts.cython_c2py(n, argt, proxy_name="c_" + n, 
                                                    cached=False)
                argdecls.append(decl)
                #argdecls.append("cdef {0} {1}".format(cython_pytype(argt), "c_" + n))
                argbodys.append(body)
                argrtns.append(rtn)
                argct = ts.cython_ctype(argt)
                argcts.append(argct)
            rtnname = 'rtn'
            rtnprox = 'c_' + rtnname
            rtncall = 'call_' + rtnname
            while rtnname in argnames or rtnprox in argnames:
                rtnname += '_'
                rtnprox += '_'
            rtnct = ts.cython_ctype(t[2][2])
            argdecls = indent(argdecls)
            argbodys = indent(argbodys)
            #rtndecl, rtnbody, rtnrtn = cython_py2c(rtnname, t[2][2], proxy_name=rtnprox)
            #rtndecl, rtnbody, rtnrtn = cython_py2c(rtnname, t[2][2], proxy_name=rtncall)
            rtndecl, rtnbody, rtnrtn = ts.cython_py2c(rtncall, t[2][2], 
                                                      proxy_name=rtnprox)
            if rtndecl is None and rtnbody is None:
                rtnprox = rtnname
            rtndecl = indent([rtndecl])
            rtnbody = indent([rtnbody])
            s = ('cdef {rtnct} {{proxy_name}}({arglist}):\n'
                 '{argdecls}\n'
                 '{rtndecl}\n'
                 '    global {{var}}\n'
                 '{argbodys}\n'
                 '    {rtncall} = {{var}}({pyarglist})\n'
                 '{rtnbody}\n'
                 '    return {rtnrtn}\n')
            arglist = ", ".join(["{0} {1}".format(*x) for x in zip(argcts, argnames)])
            pyarglist=", ".join(argrtns)
            s = s.format(rtnct=rtnct, arglist=arglist, argdecls=argdecls, 
                         rtndecl=rtndecl, argbodys=argbodys, rtnprox=rtnprox, 
                         pyarglist=pyarglist, rtnbody=rtnbody, rtnrtn=rtnrtn, 
                         rtncall=rtncall)
            return s, False

        self.cython_py2c_conv = _LazyConverterDict(cython_py2c_conv if \
                                    cython_py2c_conv is not None else {
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
            'uint32': ('<{t.cython_ctype}> long({var})', False),
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
            'map': ('{proxy_name} = {t.cython_pytype}({var}, not isinstance({var}, {t.cython_cytype}))',
                    '{proxy_name}.map_ptr[0]'),
            'dict': ('dict({var})', False),
            'pair': ('{proxy_name} = {t.cython_pytype}({var}, not isinstance({var}, {t.cython_cytype}))',
                     '{proxy_name}.pair_ptr[0]'),
            'set': ('{proxy_name} = {t.cython_pytype}({var}, not isinstance({var}, {t.cython_cytype}))',
                    '{proxy_name}.set_ptr[0]'),
            'vector': ((
                '# {var} is a {t.type}\n'
                'cdef int i{var}\n'
                'cdef int {var}_size\n'
                'cdef {t.cython_npctypes[0]} * {var}_data\n'
                '{var}_size = len({var})\n'
                'if isinstance({var}, np.ndarray) and (<np.ndarray> {var}).descr.type_num == {t.cython_nptype}:\n'
                '    {var}_data = <{t.cython_npctypes[0]} *> np.PyArray_DATA(<np.ndarray> {var})\n'
                '    {proxy_name} = {t.cython_ctype}(<size_t> {var}_size)\n'
                '    for i{var} in range({var}_size):\n'
                '        {proxy_name}[i{var}] = {var}_data[i{var}]\n'
                'else:\n'
                '    {proxy_name} = {t.cython_ctype}(<size_t> {var}_size)\n'
                '    for i{var} in range({var}_size):\n'
                '        {proxy_name}[i{var}] = <{t.cython_npctypes[0]}> {var}[i{var}]\n'),
                '{proxy_name}'),     # FIXME There might be improvements here...
            ('vector', 'char', 0): ((
                '# {var} is a {t.type}\n'
                'cdef int i{var}\n'
                'cdef int {var}_size\n'
                'cdef {t.cython_npctypes[0]} * {var}_data\n'
                '{var}_size = len({var})\n'
                'if isinstance({var}, np.ndarray) and (<np.ndarray> {var}).descr.type_num == <int> {t.cython_nptype}:\n'
                '    {var}_data = <{t.cython_npctypes[0]} *> np.PyArray_DATA(<np.ndarray> {var})\n'
                '    {proxy_name} = {t.cython_ctype}(<size_t> {var}_size)\n'
                '    for i{var} in range({var}_size):\n'
                '        {proxy_name}[i{var}] = {var}[i{var}]\n'
                'else:\n'
                '    {proxy_name} = {t.cython_ctype}(<size_t> {var}_size)\n'
                '    for i{var} in range({var}_size):\n'
                '        _ = {var}[i{var}].encode()\n'
                '        {proxy_name}[i{var}] = deref(<char *> _)\n'),
                '{proxy_name}'),
            TypeMatcher(('vector', MatchAny, '&')): cython_py2c_conv_vector_ref,
            TypeMatcher((('vector', MatchAny, 0), '&')): cython_py2c_conv_vector_ref,
            TypeMatcher((('vector', MatchAny, '&'), 0)): cython_py2c_conv_vector_ref,
            TypeMatcher((('vector', MatchAny, '&'), 'const')): cython_py2c_conv_vector_ref,
            TypeMatcher((('vector', MatchAny, 'const'), '&')): cython_py2c_conv_vector_ref,
            TypeMatcher(((('vector', MatchAny, 0), 'const'), '&')): cython_py2c_conv_vector_ref,
            TypeMatcher(((('vector', MatchAny, 0), '&'), 'const')): cython_py2c_conv_vector_ref,
            # refinement types
            'nucid': ('nucname.zzaaam({var})', False),
            'nucname': ('nucname.name({var})', False),
            TypeMatcher((('enum', MatchAny, MatchAny), '*')): \
                ('cdef int {proxy_name}_ = {var}', '&{proxy_name}_'),
            TypeMatcher((('int32', ('enum', MatchAny, MatchAny)), '*')): \
                ('cdef int {proxy_name}_ = {var}', '&{proxy_name}_'),
            'function_pointer': cython_py2c_conv_function_pointer,
            }, self)

        self.typestr = typestring or typestr

    @classmethod
    def empty(cls):
        """This is a class method which returns an empty type system."""
        x = cls(base_types=set(), template_types={}, refined_types={}, humannames={}, 
                type_aliases={}, cpp_types={}, numpy_types={}, from_pytypes={}, 
                cython_ctypes={}, cython_cytypes={}, cython_pytypes={}, 
                cython_cimports={}, cython_cyimports={}, cython_pyimports={}, 
                cython_functionnames={}, cython_classnames={}, cython_c2py_conv={}, 
                cython_py2c_conv={})
        del x.extra_types
        del x.stlcontainers
        return x

    @classmethod
    def load(cls, filename, format=None, mode='rb'):
        """Loads a type system from disk into a new type system instance.
        This is a class method.

        Parameters
        ----------
        filename : str
            Path to file.
        format : str, optional
            The file format to save the type system as.  If this is not provided, 
            it is infered from the filenme.  Options are:

            * pickle ('.pkl')
            * gzipped pickle ('.pkl.gz')

        mode : str, optional
            The mode to open the file with.

        """
        format = infer_format(filename, format)
        if not os.path.isfile(filename):
            raise RuntimeError("{0!r} not found.".format(filename))
        if format == 'pkl.gz':
            f = gzip.open(filename, 'rb')
            data = pickle.loads(f.read())
            f.close()
        elif format == 'pkl':
            with io.open(filename, 'rb') as f:
                data = pickle.loads(f.read())
        x = cls(**data)
        return x

    def dump(self, filename, format=None, mode='wb'):
        """Saves a type system out to disk.

        Parameters
        ----------
        filename : str
            Path to file.
        format : str, optional
            The file format to save the type system as.  If this is not provided, 
            it is infered from the filenme.  Options are:

            * pickle ('.pkl')
            * gzipped pickle ('.pkl.gz')

        mode : str, optional
            The mode to open the file with.

        """
        data = dict([(k, getattr(self, k, None)) for k in self.datafields])
        format = infer_format(filename, format)
        if format == 'pkl.gz':
            f = gzip.open(filename, mode)
            f.write(pickle.dumps(data, pickle.HIGHEST_PROTOCOL))
            f.close()
        elif format == 'pkl':
            with io.open(filename, mode) as f:
                f.write(pickle.dumps(data, pickle.HIGHEST_PROTOCOL))

    def update(self, *args, **kwargs):
        """Updates the type system in-place. Only updates the data attributes 
        named in 'datafields'.  This may be called with any of the following 
        signatures::

            ts.update(<TypeSystem>)
            ts.update(<dict-like>)
            ts.update(key1=value1, key2=value2, ...)

        Valid keyword arguments are the same here as for the type system 
        constructor.  See this documentation for more detail.
        """
        datafields = self.datafields
        # normalize arguments
        if len(args) == 1 and len(kwargs) == 0:
            toup = args[0]
            if isinstance(toup, TypeSystem):
                toup = dict([(k, getattr(toup, k)) for k in datafields \
                              if hasattr(toup, k)])
            elif not isinstance(toup, Mapping):
                toup = dict(toup)
        elif len(args) == 0:
            toup = kwargs
        else:
            msg = "invalid siganture: args={0!r}, kwargs={1!0}"
            raise TypeError(msg.fomat(args, kwargs))
        # verify keys
        for k in toup:
            if k not in datafields:
                msg = "{0} is not a member of {1}"
                raise AttributeError(msg.format(k, self.__class__.__name__))
        # perform the update
        for k, v in toup.items():
            x = getattr(self, k)
            if isinstance(v, Mapping):
                x.update(v)
            elif isinstance(v, Set):
                x.update(v)
            else:
                setattr(self, k, v)

    def __str__(self):
        s = pformat(dict([(k, getattr(self, k, None)) for k in \
                                                      sorted(self.datafields)]))
        return s

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += ", ".join(["{0}={1!r}".format(k, getattr(self, k, None)) \
                        for k in sorted(self.datafields)])
        s += ")"
        return s

    #################### Importnat Methods below ###############################

    @memoize_method
    def istemplate(self, t):
        """Returns whether t is a template type or not."""
        if isinstance(t, basestring):
            return t in self.template_types
        if isinstance(t, Sequence):
            return self.istemplate(t[0])
        return False

    @memoize_method
    def isenum(self, t):
        t = self.canon(t)
        return isinstance(t, Sequence) and t[0] == 'int32' and \
           isinstance(t[1], Sequence) and t[1][0] == 'enum'

    @memoize_method
    def isfunctionpointer(self, t):
        t = self.canon(t)
        return isinstance(t, Sequence) and t[0] == ('void', '*') and \
               isinstance(t[1], Sequence) and t[1][0] == 'function_pointer'

    @memoize_method
    def humanname(self, t, hnt=None):
        """Computes human names for types."""
        if hnt is None:
            t = self.canon(t)
            if isinstance(t, basestring):
                return t, self.humannames[t]
            elif t[0] in self.base_types:
                return t, self.humannames[t[0]]
            return self.humanname(t, self.humannames[t[0]])
        d = {}
        for key, x in zip(self.template_types[t[0]], t[1:-1]):
            if isinstance(x, basestring):
                val = self.humannames[x]
            elif isinstance(x, int):
                val = x
            elif x[0] in self.base_types:
                val = self.humannames[x[0]]
            else:
                val, _ = self.humanname(x, self.humannames[x[0]])
            d[key] = val
        return t, hnt.format(**d)

    @memoize_method
    def isdependent(self, t):
        """Returns whether t is a dependent type or not."""
        deptypes = set([k[0] for k in self.refined_types \
                        if not isinstance(k, basestring)])
        if isinstance(t, basestring):
            return t in deptypes
        if isinstance(t, Sequence):
            return self.isdependent(t[0])
        return False

    @memoize_method
    def isrefinement(self, t):
        """Returns whether t is a refined type."""
        if isinstance(t, basestring):
            return t in self.refined_types
        return self.isdependent(t)

    @memoize_method
    def _resolve_dependent_type(self, tname, tinst=None):
        depkey = [k for k in self.refined_types if k[0] == tname][0]
        depval = self.refined_types[depkey]
        istemplated = self.istemplate(depkey)
        if tinst is None:
            return depkey
        elif istemplated:
            assert len(tinst) == len(depkey)
            typemap = dict([(k, tinst[i]) for i, k in enumerate(depkey[1:], 1) \
                                                   if isinstance(k, basestring)])
            for k in typemap:
                if k in self.type_aliases:
                    raise TypeError('template type {0} already exists'.format(k))
            self.type_aliases.update(typemap)
            resotype = self.canon(depval), (tname,) + \
                        tuple([self.canon(k) for k in depkey[1:] if k in typemap]) + \
                        tuple([(k[0], self.canon(k[1]), instval) \
                            for k, instval in zip(depkey[1:], tinst[1:]) 
                            if k not in typemap])
            for k in typemap:
                del self.type_aliases[k]
                self.delmemo('canon', k)
            return resotype
        else:
            assert len(tinst) == len(depkey)
            return self.canon(depval), (tname,) + tuple([(kname, self.canon(ktype),
                instval) for (kname, ktype), instval in zip(depkey[1:], tinst[1:])])

    @memoize_method
    def canon(self, t):
        """Turns the type into its canonical form. See module docs for more information."""
        if isinstance(t, basestring):
            if t in self.base_types:
                return t
            elif t in self.type_aliases:
                return self.canon(self.type_aliases[t])
            elif t in self.refined_types:
                return (self.canon(self.refined_types[t]), t)
            elif self.isdependent(t):
                return self._resolve_dependent_type(t)
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
            if not isinstance(t0, basestring) and not isinstance(t0, Sequence):
                _raise_type_error(t)
            if self.isdependent(t0):
                return self._resolve_dependent_type(t0, t)
            elif t0 in self.template_types:
                templen = len(self.template_types[t0])
                last_val = 0 if tlen == 1 + templen else t[-1]
                filledt = [t0]
                for tt in t[1:1+templen]:
                    if isinstance(tt, Number):  # includes bool!
                        filledt.append(tt)
                    elif isinstance(tt, basestring):
                        try:
                            canontt = self.canon(tt)
                        except TypeError:
                            canontt = tt
                        except:
                            raise
                        filledt.append(canontt)
                    elif isinstance(tt, Sequence):
                        filledt.append(self.canon(tt))
                    else:
                        _raise_type_error(tt)
                filledt.append(last_val)
                return tuple(filledt)
            else:
                #if 2 < tlen:
                #    _raise_type_error(t)
                return (self.canon(t0), last_val)
        else:
            _raise_type_error(t)

    @memoize_method
    def strip_predicates(self, t):
        """Removes all outer predicates from a type."""
        t = self.canon(t)
        if isinstance(t, basestring):
            return t
        elif isinstance(t, Sequence):
            tlen = len(t)
            if tlen == 2:
                sp0 = self.strip_predicates(t[0])
                return (sp0, 0) if t[1] == 0 else sp0
            else:
                return t[:-1] + (0,)
        else:
            _raise_type_error(t)

    @memoize_method
    def basename(self, t):
        """Retrieves basename from a type, e.g. 'map' in ('map', 'int', 'float')."""
        t = self.canon(t)
        if isinstance(t, basestring):
            return t
        elif isinstance(t, Sequence):
            t0 = t
            while not isinstance(t0, basestring):
                t0 = t0[0]
            return t0
        else:
            _raise_type_error(t)

    ###########################   C/C++ Methods   #############################

    def _cpp_type_add_predicate(self, t, last):
        """Adds a predicate to a C++ type"""
        if last == 'const':
            x, y = last, t
        else:
            x, y = t, last
        return '{0} {1}'.format(x, y)

    @memoize_method
    def cpp_type(self, t):
        """Given a type t, returns the corresponding C++ type declaration."""
        if t in self.cpp_types:
            return self.cpp_types[t]
        t = self.canon(t)
        if isinstance(t, basestring):
            if  t in self.base_types:
                return self.cpp_types[t]
        # must be tuple below this line
        tlen = len(t)
        if 2 == tlen:
            if 0 == t[1]:
                return self.cpp_type(t[0])
            elif self.isrefinement(t[1]):
                if t[1][0] in self.cpp_types:
                    subtype = self.cpp_types[t[1][0]]
                    if callable(subtype):
                        subtype = subtype(t[1], self)
                    return subtype
                else:
                    return self.cpp_type(t[0])
            else:
                last = '[{0}]'.format(t[-1]) if isinstance(t[-1], int) else t[-1]
                return self._cpp_type_add_predicate(self.cpp_type(t[0]), last)
        elif 3 <= tlen:
            assert t[0] in self.template_types
            assert len(t) == len(self.template_types[t[0]]) + 2
            template_name = self.cpp_types[t[0]]
            assert template_name is not NotImplemented
            template_filling = []
            for x in t[1:-1]:
                if isinstance(x, bool):
                    x = self.cpp_types[x]
                elif isinstance(x, Number):
                    x = str(x)
                else:
                    x = self.cpp_type(x)
                template_filling.append(x)
            cppt = '{0}< {1} >'.format(template_name, ', '.join(template_filling))
            if 0 != t[-1]:
                last = '[{0}]'.format(t[-1]) if isinstance(t[-1], int) else t[-1]
                cppt = self._cpp_type_add_predicate(cppt, last)
            return cppt

    @memoize_method
    def cpp_funcname(self, name):
        """This returns a name for a function based on its name, rather than
        its type.  The name may be either a string or a tuple of the form 
        ('name', template_arg_type1, template_arg_type2, ...).  This is not ment
        to replace cpp_type(), but complement it.
        """
        if isinstance(name, basestring):
            return name
        fname = name[0]
        cts = []
        for x in name[1:]:
            if isinstance(x, bool):
                x = self.cpp_types[x]
            elif isinstance(x, Number):
                x = str(x)
            else:
                x = self.cpp_type(x)
            cts.append(x)
        fname += '' if 0 == len(cts) else "< " + ", ".join(cts) + " >"
        return fname

    @memoize_method
    def gccxml_type(self, t):
        """Given a type t, returns the corresponding GCC-XML type name."""
        cppt = self.cpp_type(t)
        gxt = cppt.replace('< ', '<').replace(' >', '>').\
                   replace('>>', '> >').replace(', ', ',')
        return gxt

    @memoize_method
    def cython_nptype(self, t, depth=0):
        """Given a type t, returns the corresponding numpy type.  If depth is
        greater than 0 then this returns of a list of numpy types for all internal
        template types, ie the float in ('vector', 'float', 0).

        """
        if isinstance(t, Number):
            return 'np.NPY_OBJECT'
        t = self.canon(t)
        if isinstance(t, basestring):
            return self.numpy_types[t] if t in self.numpy_types else 'np.NPY_OBJECT'
        # must be tuple below this line
        tlen = len(t)
        if t in self.numpy_types and depth < 1:
            return self.numpy_types[t]
        elif 2 == tlen:
            if 0 == t[1]:
                return self.cython_nptype(t[0])
            elif self.isrefinement(t[1]):
                return self.cython_nptype(t[0])
            else:
                # FIXME last is ignored for strings, but what about other types?
                #last = '[{0}]'.format(t[-1]) if isinstance(t[-1], int) else t[-1]
                #return cython_pytype(t[0]) + ' {0}'.format(last)
                return self.cython_nptype(t[0])
        elif 0 < depth and self.istemplate(t):
            depth -= 1
            return [self.cython_nptype(u, depth=depth) for u in t[1:-1]]
        elif 3 == tlen and self.istemplate(t):
            return self.cython_nptype(t[1])
        else:  #elif 3 <= tlen:
            return 'np.NPY_OBJECT'

    #########################   Cython Functions   ############################

    def _cython_ctype_add_predicate(self, t, last):
        """Adds a predicate to a ctype"""
        if last == 'const':
            x, y = last, t
        else:
            x, y = t, last
        return '{0} {1}'.format(x, y)

    @memoize_method
    def cython_ctype(self, t):
        """Given a type t, returns the corresponding Cython C/C++ type declaration.
        """
        t = self.canon(t)
        if t in self.cython_ctypes:
            return self.cython_ctypes[t]
        if isinstance(t, basestring):
            if t in self.base_types:
                return self.cython_ctypes[t]
        # must be tuple below this line
        tlen = len(t)
        if 2 == tlen:
            if 0 == t[1]:
                return self.cython_ctype(t[0])
            elif self.isrefinement(t[1]):
                if t[1][0] in self.cython_ctypes:
                    subtype = self.cython_ctypes[t[1][0]]
                    if callable(subtype):
                        subtype = subtype(t[1], self)
                    return subtype
                else:
                    return self.cython_ctype(t[0])
            else:
                last = '[{0}]'.format(t[-1]) if isinstance(t[-1], int) else t[-1]
                return self._cython_ctype_add_predicate(self.cython_ctype(t[0]), last)
        elif 3 <= tlen:
            assert t[0] in self.template_types
            assert len(t) == len(self.template_types[t[0]]) + 2
            template_name = self.cython_ctypes[t[0]]
            assert template_name is not NotImplemented
            template_filling = []
            for x in t[1:-1]:
                #if isinstance(x, bool):
                #    x = _cython_ctypes[x]
                #elif isinstance(x, Number):
                if isinstance(x, Number):
                    x = str(x)
                else:
                    x = self.cython_ctype(x)
                template_filling.append(x)
            cyct = '{0}[{1}]'.format(template_name, ', '.join(template_filling))
            if 0 != t[-1]:
                last = '[{0}]'.format(t[-1]) if isinstance(t[-1], int) else t[-1]
                cyct = self._cython_ctype_add_predicate(cyct, last)
            return cyct

    @memoize_method
    def _fill_cycyt(self, cycyt, t):
        """Helper for cython_cytype()."""
        d = {}
        for key, x in zip(self.template_types[t[0]], t[1:-1]):
            if isinstance(x, basestring):
                val = self.cython_classnames[x]
            elif isinstance(x, Number):
                val = str(x)
            elif x[0] in self.base_types:
                val = self.cython_classnames[x[0]]
            else:
                val, _ = self._fill_cycyt(self.cython_classnames[x[0]], x)
            d[key] = val
        return cycyt.format(**d), t

    def _cython_cytype_add_predicate(self, t, last):
        """Adds a predicate to a cytype"""
        if last == '*':
            return '{0} {1}'.format(t, last)
        elif isinstance(last, int) and 0 < last:
            return '{0} [{1}]'.format(t, last)
        else:
            return t

    @memoize_method
    def cython_cytype(self, t):
        """Given a type t, returns the corresponding Cython type."""
        t = self.canon(t)
#        if t in self.cython_cytypes:
#            return self.cython_cytypes[t]
        if isinstance(t, basestring):
            if t in self.base_types or t in self.cython_cytypes:
                return self.cython_cytypes[t]
        # must be tuple below this line
        tlen = len(t)
        if 2 == tlen:
            if 0 == t[1]:
                return self.cython_cytype(t[0])
            elif self.isrefinement(t[1]):
                if t[1][0] in self.cython_cytypes:
                    subtype = self.cython_cytypes[t[1][0]]
                    if callable(subtype):
                        subtype = subtype(t[1], self)
                    return subtype
                else:
                    return self.cython_cytype(t[0])
            else:
                return self._cython_cytype_add_predicate(self.cython_cytype(t[0]), 
                                                         t[-1])
        elif 3 <= tlen:
            if t in self.cython_cytypes:
                return self.cython_cytypes[t]
            assert t[0] in self.template_types
            assert len(t) == len(self.template_types[t[0]]) + 2
            template_name = self.cython_cytypes[t[0]]
            assert template_name is not NotImplemented
            cycyt = self.cython_cytypes[t[0]]
            cycyt, t = self._fill_cycyt(cycyt, t)
            cycyt = self._cython_cytype_add_predicate(cycyt, t[-1])
            return cycyt

    @memoize_method
    def _fill_cypyt(self, cypyt, t):
        """Helper for cython_pytype()."""
        d = {}
        for key, x in zip(self.template_types[t[0]], t[1:-1]):
            if isinstance(x, basestring):
                val = self.cython_classnames[x]
            elif isinstance(x, Number):
                val = str(x)
            elif x[0] in self.base_types:
                val = self.cython_classnames[x[0]]
            else:
                val, _ = self._fill_cypyt(self.cython_classnames[x[0]], x)
            d[key] = val
        return cypyt.format(**d), t

    @memoize_method
    def cython_pytype(self, t):
        """Given a type t, returns the corresponding Python type."""
        if isinstance(t, Number):
            return str(t)
        t = self.canon(t)
        if t in self.cython_pytypes:
            return self.cython_pytypes[t]
        if isinstance(t, basestring):
            if t in self.base_types:
                return self.cython_pytypes[t]
        # must be tuple below this line
        tlen = len(t)
        if 2 == tlen:
            if 0 == t[1]:
                return self.cython_pytype(t[0])
            elif self.isrefinement(t[1]):
                return self.cython_pytype(t[0])
            else:
                # FIXME last is ignored for strings, but what about other types?
                #last = '[{0}]'.format(t[-1]) if isinstance(t[-1], int) else t[-1]
                #return cython_pytype(t[0]) + ' {0}'.format(last)
                return self.cython_pytype(t[0])
        elif 3 <= tlen:
            if t in self.cython_pytypes:
                return self.cython_pytypes[t]
            assert t[0] in self.template_types
            assert len(t) == len(self.template_types[t[0]]) + 2
            template_name = self.cython_pytypes[t[0]]
            assert template_name is not NotImplemented
            cypyt = self.cython_pytypes[t[0]]
            cypyt, t = self._fill_cypyt(cypyt, t)
            # FIXME last is ignored for strings, but what about other types?
            #if 0 != t[-1]:
            #    last = '[{0}]'.format(t[-1]) if isinstance(t[-1], int) else t[-1]
            #    cypyt += ' {0}'.format(last)
            return cypyt

    @memoize_method
    def cython_cimport_tuples(self, t, seen=None, inc=frozenset(['c', 'cy'])):
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
        t = self.canon(t)
        if seen is None:
            seen = set()
        if isinstance(t, basestring):
            if t in self.base_types:
                if 'c' in inc:
                    seen.update(self.cython_cimports[t])
                if 'cy' in inc:
                    seen.update(self.cython_cyimports[t])
                seen -= set((None, (None,)))
                return seen
        # must be tuple below this line
        tlen = len(t)
        if 2 == tlen:
            if 'c' in inc:
                if self.isrefinement(t[1]) and t[1][0] in self.cython_cimports:
                    f = self.cython_cimports[t[1][0]]
                    if callable(f):
                        f(t[1], self, seen)
                seen.update(self.cython_cimports.get(t[0], (None,)))
                seen.update(self.cython_cimports.get(t[1], (None,)))
            if 'cy' in inc:
                if self.isrefinement(t[1]) and t[1][0] in self.cython_cyimports:
                    f = self.cython_cyimports[t[1][0]]
                    if callable(f):
                        f(t[1], self, seen)
                seen.update(self.cython_cyimports.get(t[0], (None,)))
                seen.update(self.cython_cyimports.get(t[1], (None,)))
            seen -= set((None, (None,)))
            return self.cython_cimport_tuples(t[0], seen, inc)
        elif 3 <= tlen:
            assert t[0] in self.template_types
            if 'c' in inc:
                seen.update(self.cython_cimports[t[0]])
            if 'cy' in inc:
                seen.update(self.cython_cyimports[t[0]])
            for x in t[1:-1]:
                if isinstance(x, Number):
                    continue
                elif isinstance(x, basestring) and x not in self.cython_cimports:
                    continue
                self.cython_cimport_tuples(x, seen, inc)
            seen -= set((None, (None,)))
            return seen

    _cython_cimport_cases = {
        1: lambda tup: "cimport {0}".format(*tup),
        2: lambda tup: "from {0} cimport {1}".format(*tup),
        3: lambda tup: ("cimport {0} as {2}".format(*tup) if tup[1] == 'as' else \
                        "from {0} cimport {1} as {2}".format(*tup)),
        }

    @memoize_method
    def cython_cimport_lines(self, x, inc=frozenset(['c', 'cy'])):
        """Returns the cimport lines associated with a type or a set of seen tuples.
        """
        if not isinstance(x, Set):
            x = self.cython_cimport_tuples(x, inc=inc)
        return set([self._cython_cimport_cases[len(tup)](tup) for tup in x \
                                                              if 0 != len(tup)])

    @memoize_method
    def cython_import_tuples(self, t, seen=None):
        """Given a type t, and possibly previously seen import tuples (set),
        return the set of all seen import tuples.  These tuple have four possible
        interpretations based on the length and values:

        * ``(module-name,)`` becomes ``import {module-name}``
        * ``(module-name, var-or-mod)`` becomes 
          ``from {module-name} import {var-or-mod}``
        * ``(module-name, var-or-mod, alias)`` becomes 
          ``from {module-name} import {var-or-mod} as {alias}``
        * ``(module-name, 'as', alias)`` becomes ``import {module-name} as {alias}``

        Any of these may be used.
        """
        t = self.canon(t)
        if seen is None:
            seen = set()
        if isinstance(t, basestring):
            if t in self.base_types:
                seen.update(self.cython_pyimports[t])
                seen -= set((None, (None,)))
                return seen
        # must be tuple below this line
        tlen = len(t)
        if 2 == tlen:
            if self.isrefinement(t[1]) and t[1][0] in self.cython_pyimports:
                f = self.cython_pyimports[t[1][0]]
                if callable(f):
                    f(t[1], self, seen)
            seen.update(self.cython_pyimports.get(t[0], (None,)))
            seen.update(self.cython_pyimports.get(t[1], (None,)))
            seen -= set((None, (None,)))
            return self.cython_import_tuples(t[0], seen)
        elif 3 <= tlen:
            assert t[0] in self.template_types
            seen.update(self.cython_pyimports[t[0]])
            for x in t[1:-1]:
                if isinstance(x, Number):
                    continue
                elif isinstance(x, basestring) and x not in self.cython_cimports:
                    continue
                self.cython_import_tuples(x, seen)
            seen -= set((None, (None,)))
            return seen

    _cython_import_cases = {
        1: lambda tup: "import {0}".format(*tup),
        2: lambda tup: "from {0} import {1}".format(*tup),
        3: lambda tup: ("import {0} as {2}".format(*tup) if tup[1] == 'as' else \
                        "from {0} import {1} as {2}".format(*tup)),
        }

    @memoize_method
    def cython_import_lines(self, x):
        """Returns the import lines associated with a type or a set of seen tuples.
        """
        if not isinstance(x, Set):
            x = self.cython_import_tuples(x)
        x = [tup for tup in x if 0 < len(tup)]
        return set([self._cython_import_cases[len(tup)](tup) for tup in x])

    @memoize_method
    def cython_funcname(self, name):
        """This returns a name for a function based on its name, rather than
        its type.  The name may be either a string or a tuple of the form 
        ('name', template_arg_type1, template_arg_type2, ...).  This is not ment
        to replace cython_functionname(), but complement it.
        """
        if isinstance(name, basestring):
            return name
        fname = name[0] 
        cfs = [] 
        for x in name[1:]:
            if isinstance(x, Number):
                x = str(x).replace('-', 'Neg').replace('+', 'Pos')\
                          .replace('.', 'point')
            else:
                x = self.cython_functionname(x)[1]
            cfs.append(x) 
        fname += '' if 0 == len(cfs) else "_" + "_".join(cfs)
        return fname

    @memoize_method
    def cython_functionname(self, t, cycyt=None):
        """Computes variable or function names for cython types."""
        if cycyt is None:
            t = self.canon(t)
            if isinstance(t, basestring):
                return t, self.cython_functionnames[t]
            elif t[0] in self.base_types:
                return t, self.cython_functionnames[t[0]]
            return self.cython_functionname(t, self.cython_functionnames[t[0]])
        d = {}
        for key, x in zip(self.template_types[t[0]], t[1:-1]):
            if isinstance(x, basestring):
                val = self.cython_functionnames[x] if x in self.cython_functionnames \
                                                   else x
            elif isinstance(x, Number):
                val = str(x).replace('-', 'Neg').replace('+', 'Pos')\
                            .replace('.', 'point')
            elif x[0] in self.base_types:
                val = self.cython_functionnames[x[0]]
            else:
                _, val = self.cython_functionname(x, self.cython_functionnames[x[0]])
            d[key] = val
        return t, cycyt.format(**d)

    cython_variablename = cython_functionname

    @memoize_method
    def cython_classname(self, t, cycyt=None):
        """Computes classnames for cython types."""
        if cycyt is None:
            t = self.canon(t)
            if isinstance(t, basestring):
                return t, self.cython_classnames[t]
            elif t[0] in self.base_types:
                return t, self.cython_classnames[t[0]]
            return self.cython_classname(t, self.cython_classnames[t[0]])
        d = {}
        for key, x in zip(self.template_types[t[0]], t[1:-1]):
            if isinstance(x, basestring):
                val = self.cython_classnames[x] if x in self.cython_classnames else x
            elif isinstance(x, Number):
                val = str(x).replace('-', 'Neg').replace('+', 'Pos')\
                            .replace('.', 'point')
            elif x[0] in self.base_types:
                val = self.cython_classnames[x[0]]
            else:
                _, val = self.cython_classname(x, self.cython_classnames[x[0]])
            d[key] = val
        return t, cycyt.format(**d)

    @memoize_method
    def cython_c2py_getitem(self, t):
        """Helps find the approriate c2py value for a given concrete type key."""
        tkey = t = self.canon(t)
        while tkey not in self.cython_c2py_conv and not isinstance(tkey, basestring):
            #tkey = tkey[0]
            tkey = tkey[1] if (0 < len(tkey) and self.isrefinement(tkey[1])) else \
                                                                             tkey[0]
        if tkey not in self.cython_c2py_conv:
            tkey = t
            while tkey not in self.cython_c2py_conv and \
                       not isinstance(tkey, basestring):
                tkey = tkey[0]
        c2pyt = self.cython_c2py_conv[tkey]
        if callable(c2pyt):
            self.cython_c2py_conv[t] = c2pyt(t, self)
            c2pyt = self.cython_c2py_conv[t]
        return c2pyt

    @memoize_method
    def cython_c2py(self, name, t, view=True, cached=True, inst_name=None, 
                    proxy_name=None, cache_name=None, cache_prefix='self', 
                    existing_name=None):
        """Given a varibale name and type, returns cython code (declaration, body,
        and return statements) to convert the variable from C/C++ to Python."""
        t = self.canon(t)
        c2pyt = self.cython_c2py_getitem(t)
        ind = int(view) + int(cached)
        if cached and not view:
            raise ValueError('cached views require view=True.')
        if c2pyt is NotImplemented:
            raise NotImplementedError('conversion from C/C++ to Python for ' + \
                                      t + 'has not been implemented for when ' + \
                                      'view={0}, cached={1}'.format(view, cached))
        var = name if inst_name is None else "{0}.{1}".format(inst_name, name)
        var = existing_name or var
        cache_name = "_{0}".format(name) if cache_name is None else cache_name
        cache_name = cache_name if cache_prefix is None else "{0}.{1}".format(
                                                            cache_prefix, cache_name)
        proxy_name = "{0}_proxy".format(name) if proxy_name is None else proxy_name
        iscached = False
        tstr = self.typestr(t, self)
        template_kw = dict(var=var, cache_name=cache_name, proxy_name=proxy_name, 
                           t=tstr)
#        if callable(c2pyt):
#            import pdb; pdb.set_trace()
        if 1 == len(c2pyt) or ind == 0:
            decl = body = None
            rtn = c2pyt[0].format(**template_kw)
        elif ind == 1:
            decl = "cdef {0} {1}".format(tstr.cython_cytype, proxy_name)
            body = c2pyt[1].format(**template_kw)
            rtn = proxy_name
        elif ind == 2:
            decl = "cdef {0} {1}".format(tstr.cython_cytype, proxy_name)
            body = c2pyt[2].format(**template_kw)
            rtn = cache_name
            iscached = True
        if body is not None and 'np.npy_intp' in body:
            decl = decl or ''
            decl += "\ncdef np.npy_intp {proxy_name}_shape[1]".format(
                                                                proxy_name=proxy_name)
        if decl is not None and body is not None:
            newdecl = '\n'+"\n".join([l for l in body.splitlines() \
                                              if l.startswith('cdef')])
            body = "\n".join([l for l in body.splitlines() \
                                              if not l.startswith('cdef')])
            proxy_in_newdecl = proxy_name in [l.split()[-1] for l in \
                                              newdecl.splitlines() if 0 < len(l)]
            if proxy_in_newdecl:
                for d in decl.splitlines():
                    if d.split()[-1] != proxy_name:
                        newdecl += '\n' + d
                decl = newdecl
            else:
                decl += newdecl
        return decl, body, rtn, iscached

    @memoize_method
    def cython_py2c(self, name, t, inst_name=None, proxy_name=None):
        """Given a varibale name and type, returns cython code (declaration, body,
        and return statement) to convert the variable from Python to C/C++."""
        t = self.canon(t)
        if isinstance(t, basestring) or 0 == t[-1] or self.isrefinement(t[-1]):
            last = ''
        elif isinstance(t[-1], int):
            last = ' [{0}]'.format(t[-1])
        else:
            last = ' ' + t[-1]
        tkey = t
        tinst = None
        while tkey not in self.cython_py2c_conv and not isinstance(tkey, basestring):
            tinst = tkey
            tkey = tkey[1] if (0 < len(tkey) and self.isrefinement(tkey[1])) else tkey[0]
        if tkey not in self.cython_py2c_conv:
            tkey = t
            while tkey not in self.cython_py2c_conv and \
                       not isinstance(tkey, basestring):
                tkey = tkey[0]
        py2ct = self.cython_py2c_conv[tkey]
        if callable(py2ct):
            self.cython_py2c_conv[t] = py2ct(t, self)
            py2ct = self.cython_py2c_conv[t]
        if py2ct is NotImplemented or py2ct is None:
            raise NotImplementedError('conversion from Python to C/C++ for ' + \
                                  str(t) + ' has not been implemented.')
        body_template, rtn_template = py2ct
        var = name if inst_name is None else "{0}.{1}".format(inst_name, name)
        proxy_name = "{0}_proxy".format(name) if proxy_name is None else proxy_name
        tstr = self.typestr(t, self)
        template_kw = dict(var=var, proxy_name=proxy_name, last=last, t=tstr)
        nested = False
        if self.isdependent(tkey):
            tsig = [ts for ts in self.refined_types if ts[0] == tkey][0]
            for ts, ti in zip(tsig[1:], tinst[1:]):
                if isinstance(ts, basestring):
                    template_kw[ts] = self.cython_ctype(ti)
                else:
                    template_kw[ti[0]] = ti[2]
            vartype = self.refined_types[tsig]
            if vartype in tsig[1:]:
                vartype = tinst[tsig.index(vartype)][1]
            if self.isrefinement(vartype):
                nested = True
                vdecl, vbody, vrtn = self.cython_py2c(var, vartype)
                template_kw['var'] = vrtn
        body_filled = body_template.format(**template_kw)
        if rtn_template:
            if '{t.cython_ctype}'in body_template:
                deft = tstr.cython_ctype
            elif '{t.cython_ctype_nopred}'in body_template:
                deft = tstr.cython_ctype_nopred
            elif '{t.cython_cytype_nopred}'in body_template:
                deft = tstr.cython_cytype_nopred
            else:
                deft = tstr.cython_cytype
            decl = "cdef {0} {1}".format(deft, proxy_name)
            body = body_filled
            rtn = rtn_template.format(**template_kw)
            decl += '\n'+"\n".join([l for l in body.splitlines() \
                                            if l.startswith('cdef')])
            body = "\n".join([l for l in body.splitlines() \
                                      if not l.startswith('cdef')])
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

    #################  Some utility functions for the typesystem #############

    def register_class(self, name=None, template_args=None, cython_c_type=None,
                       cython_cimport=None, cython_cy_type=None, cython_py_type=None,
                       cython_template_class_name=None, 
                       cython_template_function_name=None, cython_cyimport=None, 
                       cython_pyimport=None, cython_c2py=None, 
                       cython_py2c=None, cpp_type=None, human_name=None, 
                       from_pytype=None):
        """Classes are user specified types.  This function will add a class to 
        the type system so that it may be used normally with the rest of the 
        type system.

        """
        # register the class name
        isbase = True
        if template_args is None:
            self.base_types.add(name)  # normal class
        elif isinstance(template_args, Sequence):
            if 0 == len(template_args):
                self.base_types.add(name)  # normal class
            elif isinstance(template_args, basestring):
                _raise_type_error(name)
            else:
                self.template_types[name] = tuple(template_args)  # templated class...
                isbase = False

        # Register with Cython C/C++ types
        if (cython_c_type is not None):
            self.cython_ctypes[name] = cython_c_type
        if (cython_cy_type is not None):
            self.cython_cytypes[name] = cython_cy_type
        if (cython_py_type is not None):
            self.cython_pytypes[name] = cython_py_type
        if (from_pytype is not None):
            self.from_pytypes[name] = from_pytype
        if cpp_type is not None:
            self.cpp_types[name] = cpp_type
        if human_name is not None:
            self.humannames[name] = human_name

        if (cython_cimport is not None):
            cython_cimport = _ensure_importable(cython_cimport)
            self.cython_cimports[name] = cython_cimport
        if (cython_cyimport is not None):
            cython_cyimport = _ensure_importable(cython_cyimport)
            self.cython_cyimports[name] = cython_cyimport
        if (cython_pyimport is not None):
            cython_pyimport = _ensure_importable(cython_pyimport)
            self.cython_pyimports[name] = cython_pyimport

        if (cython_c2py is not None):
            if isinstance(cython_c2py, basestring):
                cython_c2py = (cython_c2py,)
            cython_c2py = None if cython_c2py is None else tuple(cython_c2py)
            self.cython_c2py_conv[name] = cython_c2py
        if (cython_py2c is not None):
            if isinstance(cython_py2c, basestring):
                cython_py2c = (cython_py2c, False)
            self.cython_py2c_conv[name] = cython_py2c
        if (cython_template_class_name is not None):
            self.cython_classnames[name] = cython_template_class_name
        if (cython_template_function_name is not None):
            self.cython_functionnames[name] = cython_template_function_name

    def deregister_class(self, name):
        """This function will remove a previously registered class from 
        the type system.
        """
        isbase = name in base_types
        if not isbase and name not in self.template_types:
            _raise_type_error(name)
        if isbase:
            self.base_types.remove(name)
        else:
            self.template_types.pop(name, None)

        self.cython_ctypes.pop(name, None)
        self.cython_cytypes.pop(name, None)
        self.cython_pytypes.pop(name, None)
        self.from_pytypes.pop(name, None)
        self.cpp_types.pop(name, None)
        self.humannames.pop(name, None)
        self.cython_cimports.pop(name, None)
        self.cython_cyimports.pop(name, None)
        self.cython_pyimports.pop(name, None)

        self.cython_c2py_conv.pop(name, None)
        self.cython_py2c_conv.pop(name, None)
        self.cython_classnames.pop(name, None)

        self.clearmemo()

    def register_classname(self, classname, package, pxd_base, cpppxd_base, 
                           cpp_classname=None):
        """Registers a class with the type system from only its name, 
        and relevant header file information.

        Parameters
        ----------
        classname : str or tuple
        package : str
            Package name where headers live.
        pxd_base : str
            Base name of the pxd file to cimport.
        cpppxd_base : str
            Base name of the cpppxd file to cimport.
        cpp_classname : str or tuple, optional
            Name of class in C++, equiv. to apiname.srcname. Defaults to classname.
        """
        # target classname
        baseclassname = classname
        if isinstance(classname, basestring):
            template_args = None
            templateclassname = baseclassname
            templatefuncname = baseclassname.lower()
        else:
            template_args = ['T{0}'.format(i) for i in range(len(classname)-2)]
            template_args = tuple(template_args)
            while not isinstance(baseclassname, basestring):
                baseclassname = baseclassname[0]  # hack version of ts.basename()
            templateclassname = baseclassname
            templateclassname = templateclassname + \
                                ''.join(["{"+targ+"}" for targ in template_args])
            templatefuncname = baseclassname.lower() + '_' + \
                               '_'.join(["{"+targ+"}" for targ in template_args])

        # source classname
        if cpp_classname is None:
            cpp_classname = classname
        cpp_baseclassname = cpp_classname
        if not isinstance(cpp_classname, basestring):
            while not isinstance(cpp_baseclassname, basestring):
                cpp_baseclassname = cpp_baseclassname[0]
            if template_args is None:
                template_args = ['T{0}'.format(i) for i in range(len(cpp_classname)-2)]
                template_args = tuple(template_args)

        # register regular class
        class_c2py = ('{t.cython_pytype}({var})',
                      ('{proxy_name} = {t.cython_pytype}()\n'
                       '(<{t.cython_ctype_nopred} *> {proxy_name}._inst)[0] = {var}'),
                      ('if {cache_name} is None:\n'
                       '    {proxy_name} = {t.cython_pytype}()\n'
                       '    {proxy_name}._free_inst = False\n'
                       '    {proxy_name}._inst = &{var}\n'
                       '    {cache_name} = {proxy_name}\n')
                     )
        class_py2c = ('{proxy_name} = <{t.cython_cytype_nopred}> {var}',
                      '(<{t.cython_ctype_nopred} *> {proxy_name}._inst)[0]')
        class_cimport = ((package, cpppxd_base),)
        kwclass = dict(
            name=baseclassname,                              # FCComp
            template_args=template_args,
            cython_c_type=cpppxd_base + '.' + cpp_baseclassname, # cpp_fccomp.FCComp
            cython_cimport=class_cimport,
            cython_cy_type=pxd_base + '.' + baseclassname,      # fccomp.FCComp   
            cython_py_type=pxd_base + '.' + baseclassname,      # fccomp.FCComp   
            from_pytype=[pxd_base + '.' + baseclassname],      # fccomp.FCComp   
            cpp_type=cpp_baseclassname,
            human_name=templateclassname,
            cython_template_class_name=templateclassname,
            cython_template_function_name=templatefuncname,
            cython_cyimport=((pxd_base,),),                       # fccomp
            cython_pyimport=((pxd_base,),),                       # fccomp
            cython_c2py=class_c2py,
            cython_py2c=class_py2c,
            )
        self.register_class(**kwclass)
        canonname = classname if isinstance(classname, basestring) \
                              else self.canon(classname)
        if template_args is not None:
            specname = classname if isinstance(classname, basestring) \
                                 else self.cython_classname(classname)[1]
            kwclassspec = dict(
                name=classname,
                cython_c_type=cpppxd_base + '.' + specname,
                cython_cy_type=pxd_base + '.' + specname,
                cython_py_type=pxd_base + '.' + specname,
                cpp_type=self.cpp_type(cpp_classname),
                )
            self.register_class(**kwclassspec)
            kwclassspec['name'] = canonname
            self.register_class(**kwclassspec)
            kwclassspec['name'] = self.canon(cpp_classname)
            self.register_class(**kwclassspec)
        # register numpy type
        self.register_numpy_dtype(classname,
            cython_cimport=class_cimport,
            cython_cyimport=pxd_base,
            cython_pyimport=pxd_base,
            )
        # register vector
        class_vector_py2c = ((
            '# {var} is a {t.type}\n'
            'cdef int i{var}\n'
            'cdef int {var}_size\n'
            'cdef {t.cython_npctypes[0]} * {var}_data\n'
            '{var}_size = len({var})\n'
            'if isinstance({var}, np.ndarray) and (<np.ndarray> {var}).descr.type_num == {t.cython_nptype}:\n'
            '    {var}_data = <{t.cython_npctypes[0]} *> np.PyArray_DATA(<np.ndarray> {var})\n'
            '    {proxy_name} = {t.cython_ctype_nopred}(<size_t> {var}_size)\n'
            '    for i{var} in range({var}_size):\n'
            '        {proxy_name}[i{var}] = {var}_data[i{var}]\n'
            'else:\n'
            '    {proxy_name} = {t.cython_ctype_nopred}(<size_t> {var}_size)\n'
            '    for i{var} in range({var}_size):\n'
            '        {proxy_name}[i{var}] = (<{t.cython_npctypes_nopred[0]} *> (<{t.cython_npcytypes_nopred[0]}> {var}[i{var}])._inst)[0]\n'),
            '{proxy_name}')
        self.register_class(('vector', canonname, 0), cython_py2c=class_vector_py2c)
        self.register_class(('vector', classname, 0), cython_py2c=class_vector_py2c)
        self.register_class((('vector', canonname, 0), '&'), cython_py2c=class_vector_py2c)
        self.register_class((('vector', classname, 0), '&'), cython_py2c=class_vector_py2c)
        self.register_class(((('vector', canonname, 0), 'const'), '&'), cython_py2c=class_vector_py2c)
        self.register_class(((('vector', classname, 0), 'const'), '&'), cython_py2c=class_vector_py2c)
        # register pointer to class
        class_ptr_c2py = ('{t.cython_pytype}({var})',
                         ('cdef {t.cython_pytype} {proxy_name} = {t.cython_pytype}()\n'
                          'if {proxy_name}._free_inst:\n'
                          '    free({proxy_name}._inst)\n'
                          '(<{t.cython_ctype}> {proxy_name}._inst) = {var}'),
                         ('if {cache_name} is None:\n'
                          '    {proxy_name} = {t.cython_pytype}()\n'
                          '    {proxy_name}._free_inst = False\n'
                          '    {proxy_name}._inst = {var}\n'
                          '    {cache_name} = {proxy_name}\n')
                          )
        class_ptr_py2c = ('{proxy_name} = <{t.cython_cytype_nopred}> {var}',
                          '(<{t.cython_ctype_nopred} *> {proxy_name}._inst)')
        kwclassptr = dict(
            name=(classname, '*'),
            template_args=template_args,
            cython_py_type=pxd_base + '.' + baseclassname,
            cython_cy_type=pxd_base + '.' + baseclassname,
            cpp_type=cpp_baseclassname,
            cython_c2py=class_ptr_c2py,
            cython_py2c=class_ptr_py2c,
            cython_cimport=kwclass['cython_cimport'] ,
            cython_cyimport=kwclass['cython_cyimport'] + (('libc.stdlib','free'),),
            cython_pyimport=kwclass['cython_pyimport'],
            )
        self.register_class(**kwclassptr)
        kwclassref = dict(kwclassptr)
        # Register reference to class
        kwclassref['name'] = (classname, '&')
        kwclassref['cython_c2py'] = class_c2py
        kwclassref['cython_py2c'] = class_py2c
        #ts.register_class(**kwclassref)
        # register doublepointer to class
        class_dblptr_c2py = ('{t.cython_pytype}({var})',
                            ('{proxy_name} = {proxy_name}_obj._inst\n'
                             '(<{t.cython_ctype} *> {proxy_name}._inst) = {var}'),
                            ('if {cache_name} is None:\n'
                             '    {proxy_name} = {t.cython_pytype}()\n'
                             '    {proxy_name}._free_inst = False\n'
                             '    {proxy_name}._inst = {var}\n'
                             '    {proxy_name}_list = [{proxy_name}]\n'
                             '    {cache_name} = {proxy_name}_list\n')
                             )
        class_dblptr_py2c = ('{proxy_name} = <{t.cython_cytype_nopred}> {var}[0]',
                             '(<{t.cython_ctype_nopred} **> {proxy_name}._inst)')
        kwclassdblptr = dict(
            name=((classname, '*'), '*'),
            template_args=template_args,
            cython_c2py=class_dblptr_c2py,
            cython_py2c=class_dblptr_py2c,
            cython_cimport=kwclass['cython_cimport'],
            cython_cyimport=kwclass['cython_cyimport'],
            cython_pyimport=kwclass['cython_pyimport'],
            )
        self.register_class(**kwclassdblptr)

    def register_refinement(self, name, refinementof, cython_cimport=None, 
                            cython_cyimport=None, cython_pyimport=None, 
                            cython_c2py=None, cython_py2c=None):
        """This function will add a refinement to the type system so that it 
        may be used normally with the rest of the type system.
        """
        self.refined_types[name] = refinementof

        cyci = _ensure_importable(cython_cimport)
        self.cython_cimports[name] = cyci

        cycyi = _ensure_importable(cython_cyimport)
        self.cython_cyimports[name] = cycyi

        cypyi = _ensure_importable(cython_pyimport)
        self.cython_pyimports[name] = cypyi

        if isinstance(cython_c2py, basestring):
            cython_c2py = (cython_c2py,)
        cython_c2py = None if cython_c2py is None else tuple(cython_c2py)
        if cython_c2py is not None:
            self.cython_c2py_conv[name] = cython_c2py

        if isinstance(cython_py2c, basestring):
            cython_py2c = (cython_py2c, False)
        if cython_py2c is not None:
            self.cython_py2c_conv[name] = cython_py2c

    def deregister_refinement(self, name):
        """This function will remove a previously registered refinement from 
        the type system.
        """
        self.refined_types.pop(name, None)
        self.cython_c2py_conv.pop(name, None)
        self.cython_py2c_conv.pop(name, None)
        self.cython_cimports.pop(name, None)
        self.cython_cyimports.pop(name, None)
        self.cython_pyimports.pop(name, None)
        self.clearmemo()

    def register_specialization(self, t, cython_c_type=None, cython_cy_type=None,
                                cython_py_type=None, cython_cimport=None,
                                cython_cyimport=None, cython_pyimport=None):
        """This function will add a template specialization so that it may be used
        normally with the rest of the type system.
        """
        t = self.canon(t)
        if cython_c_type is not None:
            self.cython_ctypes[t] = cython_c_type
        if cython_cy_type is not None:
            self.cython_cytypes[t] = cython_cy_type
        if cython_py_type is not None:
            self.cython_pytypes[t] = cython_py_type
        if cython_cimport is not None:
            self.cython_cimports[t] = cython_cimport
        if cython_cyimport is not None:
            self.cython_cyimports[t] = cython_cyimport
        if cython_pyimport is not None:
            self.cython_pyimports[t] = cython_pyimport

    def deregister_specialization(self, t):
        """This function will remove previously registered template specialization."""
        t = self.canon(t)
        self.cython_ctypes.pop(t, None)
        self.cython_cytypes.pop(t, None)
        self.cython_pytypes.pop(t, None)
        self.cython_cimports.pop(t, None)
        self.cython_cyimports.pop(t, None)
        self.cython_pyimports.pop(t, None)
        self.clearmemo()

    def register_numpy_dtype(self, t, cython_cimport=None, cython_cyimport=None, 
                             cython_pyimport=None):
        """This function will add a type to the system as numpy dtype that lives in
        the stlcontainers module.
        """
        t = self.canon(t)
        if t in self.numpy_types:
            return
        varname = self.cython_variablename(t)[1]
        self.numpy_types[t] = '{stlcontainers}xd_' + varname + '.num'
        self.type_aliases[self.numpy_types[t]] = t
        self.type_aliases['xd_' + varname] = t
        self.type_aliases['xd_' + varname + '.num'] = t
        self.type_aliases['{stlcontainers}xd_' + varname] = t
        self.type_aliases['{stlcontainers}xd_' + varname + '.num'] = t
        if cython_cimport is not None:
            x = _ensure_importable(self.cython_cimports._d.get(t, None))
            x = x + _ensure_importable(cython_cimport)
            self.cython_cimports[t] = x
        # cython imports
        x = (('{stlcontainers}',),)
        x = x + _ensure_importable(self.cython_cyimports._d.get(t, None))
        x = x + _ensure_importable(cython_cyimport)
        self.cython_cyimports[t] = x
        # python imports
        x = (('{stlcontainers}',),)
        x = x + _ensure_importable(self.cython_pyimports._d.get(t, None))
        x = x + _ensure_importable(cython_pyimport)
        self.cython_pyimports[t] = x

    #################### Type system helpers ###################################

    def clearmemo(self):
        """Clears all method memoizations on this type system instance."""
        # see utils.memozie_method
        if hasattr(self, '_cache'):
            self._cache.clear()

    def delmemo(self, meth, *args, **kwargs):
        """Deletes a single key from a method on this type system instance."""
        # see utils.memozie_method
        if hasattr(self, '_cache'):
            meth = getattr(self, meth )if isinstance(meth, basestring) else meth
            del self._cache[meth.func.meth, args, tuple(sorted(kwargs.items()))]

    @contextmanager
    def swap_stlcontainers(self, s):
        """A context manager for temporarily swapping out the stlcontainer value
        with a new value and replacing the original value before exiting."""
        old = self.stlcontainers
        self.stlcontainers = s
        self.clearmemo()
        yield
        self.clearmemo()
        self.stlcontainers = old

    @contextmanager
    def local_classes(self, classnames, typesets=frozenset(['cy', 'py'])):
        """A context manager for making sure the given classes are local."""
        saved = {}
        for name in classnames:
            if 'c' in typesets and name in self.cython_ctypes:
                saved[name, 'c'] = _undot_class_name(name, self.cython_ctypes)
            if 'cy' in typesets and name in self.cython_cytypes:
                saved[name, 'cy'] = _undot_class_name(name, self.cython_cytypes)
            if 'py' in typesets and name in self.cython_pytypes:
                saved[name, 'py'] = _undot_class_name(name, self.cython_pytypes)
        self.clearmemo()
        yield
        for name in classnames:
            if 'c' in typesets and name in self.cython_ctypes:
                _redot_class_name(name, self.cython_ctypes, saved[name, 'c'])
            if 'cy' in typesets and name in self.cython_cytypes:
                _redot_class_name(name, self.cython_cytypes, saved[name, 'cy'])
            if 'py' in typesets and name in self.cython_pytypes:
                _redot_class_name(name, self.cython_pytypes, saved[name, 'py'])
        self.clearmemo()

#################### Type System Above This Line ##############################

################### Type Matching #############################################o

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
        if isinstance(t, basestring) or isinstance(t, bool) or \
           isinstance(t, int) or isinstance(t, float):
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


def matches(pattern, t):
    """Indicates whether a type t matches a pattern. See TypeMatcher for more details.
    """
    tm = TypeMatcher(pattern)
    return tm.matches(t)

#################### Lazy Configuration ###################################

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

#################### Type string for formatting ################################

class typestr(object):
    """This is class whose attributes are properties that expose various 
    string representations of a type.  This is useful for the Python string
    formatting mini-language where attributes of an object may be accessed.
    For example:

        "This is the Cython C/C++ type: {t.cython_ctype}".format(t=typestr(t, ts))

    This mechanism is used for accessing type information in conversion strings.
    """

    def __init__(self, t, ts):
        """Parameters
        ----------
        t : str or tuple
            A valid repesentation of a type in the type systems
        ts : TypeSystem
            A type system to generate the string representations with.
        """
        self.ts = ts
        self.t = ts.canon(t)
        self.t_nopred = ts.strip_predicates(t)

    _type = None

    @property
    def type(self):
        """This is a repr string of the raw type (self.t), mostly useful for 
        comments."""
        if self._type is None:
            self._type = repr(self.t)
        return self._type

    _cython_ctype = None

    @property
    def cython_ctype(self):
        """The Cython C/C++ representation of the type.
        """
        if self._cython_ctype is None:
            self._cython_ctype = self.ts.cython_ctype(self.t)
        return self._cython_ctype

    _cython_cytype = None

    @property
    def cython_cytype(self):
        """The Cython Cython representation of the type.
        """
        if self._cython_cytype is None:
            self._cython_cytype = self.ts.cython_cytype(self.t)
        return self._cython_cytype

    _cython_pytype = None

    @property
    def cython_pytype(self):
        """The Cython Python representation of the type.
        """
        if self._cython_pytype is None:
            self._cython_pytype = self.ts.cython_pytype(self.t)
        return self._cython_pytype

    _cython_nptype = None

    @property
    def cython_nptype(self):
        """The Cython NumPy representation of the type.
        """
        if self._cython_nptype is None:
            self._cython_nptype = self.ts.cython_nptype(self.t)
        return self._cython_nptype

    _cython_npctype = None

    @property
    def cython_npctype(self):
        """The Cython C/C++ representation of NumPy type.
        """
        if self._cython_npctype is None:
            npt = self.ts.cython_nptype(self.t)
            npct = self.ts.cython_ctype(npt)
            self._cython_npctype = npct
        return self._cython_npctype

    _cython_npcytype = None

    @property
    def cython_npcytype(self):
        """The Cython Cython representation of NumPy type.
        """
        if self._cython_npcytype is None:
            npt = self.ts.cython_nptype(self.t)
            npcyt = self.ts.cython_cytype(npt)
            self._cython_npcytype = npcyt
        return self._cython_npcytype

    _cython_nppytype = None

    @property
    def cython_nppytype(self):
        """The Cython Python representation of NumPy type.
        """
        if self._cython_nppytype is None:
            npt = self.ts.cython_nptype(self.t)
            nppyt = self.ts.cython_pytype(npt)
            self._cython_nppytype = nppyt
        return self._cython_nppytype

    _cython_nptypes = None

    @property
    def cython_nptypes(self):
        """The expanded Cython NumPy representation of the type.
        """
        if self._cython_nptypes is None:
            npts = self.ts.cython_nptype(self.t, depth=1)
            npts = [npts] if isinstance(npts, basestring) else npts
            self._cython_nptypes = npts
        return self._cython_nptypes

    _cython_npctypes = None

    @property
    def cython_npctypes(self):
        """The expanded Cython C/C++ representation of the NumPy types.
        """
        if self._cython_npctypes is None:
            npts = self.ts.cython_nptype(self.t, depth=1)
            npts = [npts] if isinstance(npts, basestring) else npts
            npcts = _maprecurse(self.ts.cython_ctype, npts)
            self._cython_npctypes = npcts
        return self._cython_npctypes

    _cython_npcytypes = None

    @property
    def cython_npcytypes(self):
        """The expanded Cython Cython representation of the NumPy types.
        """
        if self._cython_npcytypes is None:
            npts = self.ts.cython_nptype(self.t, depth=1)
            npts = [npts] if isinstance(npts, basestring) else npts
            npcyts = _maprecurse(self.ts.cython_cytype, npts)
            self._cython_npcytypes = npcyts
        return self._cython_npcytypes

    _cython_nppytypes = None

    @property
    def cython_nppytypes(self):
        """The expanded Cython Cython representation of the NumPy types.
        """
        if self._cython_nppytypes is None:
            npts = self.ts.cython_nptype(self.t, depth=1)
            npts = [npts] if isinstance(npts, basestring) else npts
            nppyts = _maprecurse(self.ts.cython_pytype, npts)
            self._cython_nppytypes = nppyts
        return self._cython_nppytypes

    _type_nopred = None

    @property
    def type_nopred(self):
        """This is a repr string of the raw type (self.t) without predicates."""
        if self._type_nopred is None:
            self._type_nopred = repr(self.t_nopred)
        return self._type_nopred

    _cython_ctype_nopred = None

    @property
    def cython_ctype_nopred(self):
        """The Cython C/C++ representation of the type without predicates.
        """
        if self._cython_ctype_nopred is None:
            self._cython_ctype_nopred = self.ts.cython_ctype(self.t_nopred)
        return self._cython_ctype_nopred

    _cython_cytype_nopred = None

    @property
    def cython_cytype_nopred(self):
        """The Cython Cython representation of the type without predicates.
        """
        if self._cython_cytype_nopred is None:
            self._cython_cytype_nopred = self.ts.cython_cytype(self.t_nopred)
        return self._cython_cytype_nopred

    _cython_pytype_nopred = None

    @property
    def cython_pytype_nopred(self):
        """The Cython Python representation of the type without predicates.
        """
        if self._cython_pytype_nopred is None:
            self._cython_pytype_nopred = self.ts.cython_pytype(self.t_nopred)
        return self._cython_pytype_nopred

    _cython_nptype_nopred = None

    @property
    def cython_nptype_nopred(self):
        """The Cython NumPy representation of the type without predicates.
        """
        if self._cython_nptype_nopred is None:
            self._cython_nptype_nopred = self.ts.cython_nptype(self.t_nopred)
        return self._cython_nptype_nopred

    _cython_npctype_nopred = None

    @property
    def cython_npctype_nopred(self):
        """The Cython C/C++ representation of the NumPy type without predicates.
        """
        if self._cython_npctype_nopred is None:
            npt_nopred = self.ts.cython_nptype(self.t_nopred)
            npct_nopred = self.cython_ctype(npt_nopred)            
            self._cython_npctype_nopred = npct_nopred
        return self._cython_npctype_nopred

    _cython_npcytype_nopred = None

    @property
    def cython_npcytype_nopred(self):
        """The Cython Cython representation of the NumPy type without predicates.
        """
        if self._cython_npcytype_nopred is None:
            npt_nopred = self.ts.cython_nptype(self.t_nopred)
            npcyt_nopred = self.cython_cytype(npt_nopred)            
            self._cython_npcytype_nopred = npcyt_nopred
        return self._cython_npcytype_nopred

    _cython_nppytype_nopred = None

    @property
    def cython_nppytype_nopred(self):
        """The Cython Python representation of the NumPy type without predicates.
        """
        if self._cython_nppytype_nopred is None:
            npt_nopred = self.ts.cython_nptype(self.t_nopred)
            nppyt_nopred = self.cython_pytype(npt_nopred)            
            self._cython_nppytype_nopred = nppyt_nopred
        return self._cython_nppytype_nopred

    _cython_nptypes_nopred = None

    @property
    def cython_nptypes_nopred(self):
        """The Cython NumPy representation of the types without predicates.
        """
        if self._cython_nptypes_nopred is None:
            self._cython_nptypes_nopred = self.ts.cython_nptype(self.t_nopred, depth=1)
        return self._cython_nptypes_nopred

    _cython_npctypes_nopred = None

    @property
    def cython_npctypes_nopred(self):
        """The Cython C/C++ representation of the NumPy types without predicates.
        """
        if self._cython_npctypes_nopred is None:
            npts_nopred = self.ts.cython_nptype(self.t_nopred, depth=1)
            npts_nopred = [npts_nopred] if isinstance(npts_nopred, basestring) \
                                        else npts_nopred
            npcts_nopred = _maprecurse(self.ts.cython_ctype, npts_nopred)
            self._cython_npctypes_nopred = npcts_nopred
        return self._cython_npctypes_nopred

    _cython_npcytypes_nopred = None

    @property
    def cython_npcytypes_nopred(self):
        """The Cython Cython representation of the NumPy types without predicates.
        """
        if self._cython_npcytypes_nopred is None:
            npts_nopred = self.ts.cython_nptype(self.t_nopred, depth=1)
            npts_nopred = [npts_nopred] if isinstance(npts_nopred, basestring) \
                                        else npts_nopred
            npcyts_nopred = _maprecurse(self.ts.cython_cytype, npts_nopred)
            self._cython_npcytypes_nopred = npcyts_nopred
        return self._cython_npcytypes_nopred

    _cython_nppytypes_nopred = None

    @property
    def cython_nppytypes_nopred(self):
        """The Cython Python representation of the NumPy types without predicates.
        """
        if self._cython_nppytypes_nopred is None:
            npts_nopred = self.ts.cython_nptype(self.t_nopred, depth=1)
            npts_nopred = [npts_nopred] if isinstance(npts_nopred, basestring) \
                                        else npts_nopred
            nppyts_nopred = _maprecurse(self.ts.cython_pytype, npts_nopred)
            self._cython_nppytypes_nopred = nppyts_nopred
        return self._cython_nppytypes_nopred


#################### Type system helpers #######################################

def _raise_type_error(t):
    raise TypeError("type of {0!r} could not be determined".format(t))

_ensuremod = lambda x: x if x is not None and 0 < len(x) else ''
_ensuremoddot = lambda x: x + '.' if x is not None and 0 < len(x) else ''

def _recurse_replace(x, a, b):
    if isinstance(x, basestring):
        return x.replace(a, b)
    elif isinstance(x, Sequence):
        return tuple([_recurse_replace(y, a, b) for y in x])
    else:
        return x

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

def _maprecurse(f, x):
    if not isinstance(x, list):
        return [f(x)]
    #return [_maprecurse(f, y) for y in x]
    l = []
    for y in x:
        l += _maprecurse(f, y)
    return l

def _ensure_importable(x):
    if isinstance(x, basestring) or x is None:
        r = ((x,),)
    elif isinstance(x, Iterable) and (isinstance(x[0], basestring) or x[0] is None):
        r = (x,)
    else:
        r = x
    return r
