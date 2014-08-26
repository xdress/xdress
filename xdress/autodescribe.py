"""This module creates descriptions of C/C++ classes, functions, and variables
from source code, by using external parsers (GCC-XML, Clang AST) and the type system.

This module is available as an xdress plugin by the name ``xdress.autodescribe``.

:author: Anthony Scopatz <scopatz@gmail.com>

Descriptions
============
A key component of API wrapper generation is having a a top-level, abstract
representation of the software that is being wrapped.  In C++ there are three
basic constructs which may be wrapped: variables, functions, and classes.

The abstract representation of a C++ class is known as a **description** (abbr.
*desc*).  This description is simply a Python dictionary with a specific structure.
This structure makes heavy use of the type system to declare the types of all needed
parameters.

**NOTE**: Unions are wrapped as classes. From Python they will behave much like
wrapped structs, with the addition of union memory-sharing. Also note that assigning
a union object instance to some value will not behave anything like it does in C/C++
(ie: getting the value from the union which matches the type of the l-value in
the expression).

The Name Key
------------
The *name* key is a dictionary that represents the API name of the element
being described.  This contains exactly the same keys that the utils.apiname()
type has fields.  While apiname is used for user input and validation, the values
here must truly describe the API element.  The following keys -- and only the
following keys -- are allowed in the name dictionary.

:srcname: str or tuple, the element's API name in the original source code,
    eg. MyClass.
:srcfiles: tuple of str, this is a sequence of unique strings which represents
    the file paths where the API element *may* be defined. For example, ('myfile.c',
    'myfile.h').  If the element is defined outside of these files, then the automatic
    discovery or description may fail. Since these files are parsed they must
    actually exist on the filesystem.
:tarbase: str, the base portion of all automatically generated (target) files.
    This does not include the directory or the file extension.  For example, if you
    wanted cythongen to create a file name 'mynewfile.pyx' then the value here would
    be simply 'mynewfile'.
:tarname: str or tuple, the element's API name in the automatically generated
    (target) files, e.g. MyNewClass.
:incfiles: tuple of str, this is a sequence of all files which must be #include'd
    to access the srcname at compile time.  This should be as minimal of a set as
    possible, preferably only one file.  For example, 'hdf5.h'.
:sidecars: tuple of str, this is a sequence of all sidecar files to use for
    this API element. Like srcfiles, these files must exist for xdress to run.
    For example, 'myfile.py'.
:language: str, flag for the language that the srcfiles are implemented in. Valid
    options are: 'c', 'c++', 'f', 'fortran', 'f77', 'f90', 'python', and 'cython'.

Variable Description Top-Level Keys
------------------------------------
The following are valid top-level keys in a variable description dictionary:
name, namespace, type, docstring, and extra.

:name: dict, the variable name, see above
:namespace: str or None, the namespace or module the variable lives in.
:type: str or tuple, the type of the variable
:docstring: str, optional, this is a documentation string for the variable.
:extra: dict, optional, this stores arbitrary metadata that may be used with
    different backends. It is not added by any auto-describe routine but may be
    inserted later if needed.  One example use case is that the Cython generation
    looks for the pyx, pxd, and cpppxd keys for strings of supplemental Cython
    code to insert directly into the wrapper.

Function Description Top-Level Keys
------------------------------------
The following are valid top-level keys in a function description dictionary:
name, namespace, signatures, docstring, and extra.

:name: dict, the function name, see above
:namespace: str or None, the namespace or module the function lives in.
:signatures: dict or dict-like, the keys of this dictionary are function call
    signatures and the values are dicts of non-signature information.
    The signatures themselves are tuples. The first element of these tuples is the
    method name. The remaining elements (if any) are the function arguments.
    Arguments are themselves length-2 tuples whose first elements are the argument
    names and the second element is the argument type. The values are themselves
    dicts with the following keys:

        :return: the return type of this function. Unlike class constuctors
            and destructors, the return type may not be None (only 'void' values
            are allowed).
        :defaults: a length-N tuple of length-2 tuples of the default argument
            kinds and values. N must be the number of arguments in the signature.
            In the length-2 tuples, the first element must be a member of the
            utils.Arg enum and the second element is the associated default value.
            If no default argument exists use utils.Args.NONE as the kind and
            by convention set the value to None, though this should be ignored in all
            cases.

:docstring: str, optional, this is a documentation string for the function.
:extra: dict, optional, this stores arbitrary metadata that may be used with
    different backends. It is not added by any auto-describe routine but may be
    inserted later if needed.  One example use case is that the Cython generation
    looks for the pyx, pxd, and cpppxd keys for strings of supplemental Cython
    code to insert directly into the wrapper.

Class Description Top-Level Keys
---------------------------------
The following are valid top-level keys in a class description dictionary:
name, parents, namespace, attrs, methods, docstrings, and extra.

:name: dict, the class name, see above
:parents: possibly empty list of strings, the immediate parents of the class
    (not grandparents).
:namespace: str or None, the namespace or module the class lives in.
:attrs: dict or dict-like, the names of the attributes (member variables) of the
    class mapped to their types, given in the format of the type system.
:methods: dict or dict-like, similar to the attrs except that the keys are now
    function signatures and the values are dicts of non-signature information.
    The signatures themselves are tuples. The first element of these tuples is the
    method name. The remaining elements (if any) are the function arguments.
    Arguments are themselves length-2 tuples whose first elements are the argument
    names and the second element is the argument type. The values are themselves
    dicts with the following keys:

        :return: the return type of this function. If the return type is None
            (as opposed to 'void'), then this method is assumed to be a constructor
            or destructor.
        :defaults: a length-N tuple of length-2 tuples of the default argument
            kinds and values. N must be the number of arguments in the signature.
            In the length-2 tuples, the first element must be a member of the
            utils.Arg enum and the second element is the associated default value.
            If no default argument exists use utils.Args.NONE as the kind and
            by convention set the value to None, though this should be ignored in all
            cases.

:construct: str, optional, this is a flag for how the class is implemented.
    Accepted values are 'class' and 'struct'.  If this is not present, then 'class'
    is assumed.  This is most useful from wrapping C structs as Python classes.
:docstrings: dict, optional, this dictionary is meant for storing documentation
    strings.  All values are thus either strings or dictionaries of strings.
    Valid keys include: class, attrs, and methods.  The attrs and methods
    keys are dictionaries which may include keys that mirror the top-level keys of
    the same name.
:extra: dict, optional, this stores arbitrary metadata that may be used with
    different backends. It is not added by any auto-describe routine but may be
    inserted later if needed.  One example use case is that the Cython generation
    looks for the pyx, pxd, and cpppxd keys for strings of supplemental Cython
    code to insert directly into the wrapper.

Toaster Example
---------------
Suppose we have a C++ class called Toaster that takes bread and makes delicious
toast.  A valid description dictionary for this class would be as follows::

    class_desc = {
        'name': {
            'language': 'c++',
            'incfiles': ('toaster.h',),
            'srcfiles': ('src/toaster.h', 'src/toaster.cpp'),
            'srcname': 'Toaster',
            'sidecars': ('src/toaster.py',),
            'tarbase': 'toaster',
            'tarname': 'Toaster',
            },
        'parents': ['FCComp'],
        'namespace': 'bright',
        'construct': 'class',
        'attrs': {
            'n_slices': 'int32',
            'rate': 'float64',
            'toastiness': 'str',
            },
        'methods': {
            ('Toaster',): {'return': None, 'defaults': ()},
            ('Toaster', ('name', 'str')): {'return': None,
                'defaults': ((Args.LIT, ""),)},
            ('Toaster', ('paramtrack', ('set', 'str')), ('name', 'str', '""')): {
                'return': None,
                'defaults': ((Args.NONE, None), (Args.LIT, ""))},
            ('~Toaster',): {'return': None, 'defaults': ()},
            ('tostring',): {'return': 'str', 'defaults': ()},
            ('calc',): {'return': 'Material', 'defaults': ()},
            ('calc', ('incomp', ('map', 'int32', 'float64'))): {
                'return': 'Material',
                'defaults': ((Args.NONE, None),)},
            ('calc', ('mat', 'Material')): {
                'return': 'Material',
                'defaults': ((Args.NONE, None),)},
            ('write', ('filename', 'str')): {
                'return': 'void',
                'defaults': ((Args.LIT, "toaster.txt"),)},
            ('write', ('filename', ('char' '*'), '"toaster.txt"')): {
                'return': 'void',
                'defaults': ((Args.LIT, "toaster.txt"),)},
            },
        'docstrings': {
            'class': "I am a toaster!",
            'attrs': {
                'n_slices': 'the number of slices',
                'rate': 'the toast rate',
                'toastiness': 'the toastiness level',
                },
            'methods': {
                'Toaster': "Make me a toaster!",
                '~Toaster': "Noooooo",
                'tostring': "string representation of the toaster",
                'calc': "actually makes the toast.",
                'write': "persists the toaster state."
                },
            },
        'extra': {
            'pyx': 'toaster = Toaster()  # make toaster singleton'
            },
        }

Automatic Description Generation
--------------------------------
The purpose of this module is to create description dictionaries like those
above by automatically parsing C++ classes.  In theory this parsing step may
be handled by visiting any syntax tree of C++ code.  Two options were pursued here:
GCC-XML and the Python bindings to the Clang AST.  Unfortunately, the Clang AST
bindings lack exposure for template argument types.  These are needed to use any
standard library containers.  Thus while the Clang method was pursued to a mostly
working state, the GCC-XML version is the only fully functional automatic describer
for the moment.

Automatic Descriptions API
==========================
"""
from __future__ import print_function
import os
import io
import re
import sys
import ast
from copy import deepcopy
import linecache
import subprocess
import itertools
import functools
import pickle
import collections
from hashlib import md5
from numbers import Number
from pprint import pprint, pformat
from warnings import warn
from functools import reduce

if os.name == 'nt':
    import ntpath
    import posixpath

# pycparser conditional imports
try:
    import pycparser
    PycparserNodeVisitor = pycparser.c_ast.NodeVisitor
except ImportError:
    pycparser = None
    PycparserNodeVisitor = object  # fake this for class definitions

from . import utils
from .utils import exec_file, RunControl, NotSpecified, Arg, merge_descriptions, \
    find_source, FORBIDDEN_NAMES, find_filenames, warn_forbidden_name, apiname, \
    ensure_apiname, c_literal, extra_filenames, newoverwrite, _lang_exts
from . import astparsers
from .types.system import TypeSystem

try:
    from . import clang
    from .clang.cindex import CursorKind, TypeKind, AccessKind
except ImportError:
    clang = None

if sys.version_info[0] >= 3:
    basestring = str

# d = int64, u = uint64
_GCCXML_LITERAL_INTS = re.compile('(\d+)([du]?)')
_GCCXML_LITERAL_ENUMS = re.compile('\(\w+::(\w+)\)(\d+)')

_none_arg = Arg.NONE, None
_none_return = {'return': None, 'defaults': ()}

def clearmemo():
    """Clears all function memoizations for autodescribers."""
    for x in globals().values():
        if callable(x) and hasattr(x, 'cache'):
            x.cache.clear()

def _make_c_to_xdress():
    numpy_to_c = {
        'void': 'void',
        'bool': 'bool',
        'byte': ('char','signed char'), # TODO: Don't assume char is signed in C
        'ubyte': 'unsigned char',
        'short': ('short','short int'),
        'ushort': ('unsigned short','short unsigned int'),
        'intc': 'int',
        'uintc': 'unsigned int',
        'int': ('long','long int'),
        'uint': ('unsigned long','long unsigned int'),
        'longlong': 'long long',
        'ulonglong': ('unsigned long long','long long unsigned int'),
        'int16': 'int16_t',
        'int32': 'int32_t',
        'int64': 'int64_t',
        'float32': 'float',
        'float64': 'double',
        'complex64': 'float _Complex',
        'complex128': 'double _Complex',
        'longdouble': 'long double',
        }
    # TODO: Revisit as part of https://github.com/xdress/xdress/issues/109.
    # The above table is likely fine (except for the signedness of char),
    # but we need to make sure the platform-agnostic information isn't
    # standardized away.
    c_to_xdress = {}
    for n,cs in numpy_to_c.items():
        for c in cs if isinstance(cs,tuple) else (cs,):
            c_to_xdress[c] = n
    return c_to_xdress

_c_to_xdress = _make_c_to_xdress()
_integer_types = frozenset(s+i for i in 'int16 int32 int64 short intc int longlong'.split() for s in ('','u'))
_float_types = frozenset('float double float32 float64'.split())

if clang is not None:
    _clang_base_types = {
        TypeKind.VOID       : 'void',
        TypeKind.BOOL       : 'bool',
        TypeKind.CHAR_U     : 'ubyte',
        TypeKind.UCHAR      : 'ubyte',
        TypeKind.USHORT     : 'ushort',
        TypeKind.UINT       : 'uintc',
        TypeKind.ULONG      : 'uint',
        TypeKind.ULONGLONG  : 'ulonglong',
        TypeKind.CHAR_S     : 'byte',
        TypeKind.SCHAR      : 'byte',
        TypeKind.SHORT      : 'short',
        TypeKind.INT        : 'intc',
        TypeKind.LONG       : 'int',
        TypeKind.LONGLONG   : 'longlong',
        TypeKind.FLOAT      : 'float32',
        TypeKind.DOUBLE     : 'float64',
        TypeKind.LONGDOUBLE : 'longdouble',
        }

if 1:
    # TODO: This step uses platform dependent details, and will need to be cleaned
    # if we want to generate platform independent .pyx files.
    from numpy import dtype
    def hack_type(t):
        t = dtype(t).name
        return t[:-4]+'char' if t.endswith('int8') else t
    _c_to_xdress = dict((k,hack_type(v)) for k,v in _c_to_xdress.items())
    if clang is not None:
        _clang_base_types = dict((k,hack_type(v)) for k,v in _clang_base_types.items())

# Hard code knowledge of certain templated classes, so that allocator arguments
# can be stripped.  TODO: This should be replaced by a more flexible mechanism
hack_template_args = {
    'array': ('value_type',),
    'deque': ('value_type',),
    'forward_list': ('value_type',),
    'list': ('value_type',),
    'map': ('key_type', 'mapped_type'),
    'multimap': ('key_type', 'mapped_type'),
    'set': ('key_type',),
    'multiset': ('key_type',),
    'unordered_map': ('key_type', 'mapped_type'),
    'unordered_multimap': ('key_type', 'mapped_type'),
    'unordered_set': ('key_type',),
    'unordered_multiset': ('key_type',),
    'vector': ('value_type',),
    }

#
# GCC-XML Describers
#

def gccxml_describe(filename, name, kind, includes=(), defines=('XDRESS',),
                    undefines=(), extra_parser_args=(), ts=None, verbose=False,
                    debug=False, builddir='build', onlyin=None, language='c++',
                    clang_includes=()):
    """Use GCC-XML to describe the class.

    Parameters
    ----------
    filename : str
        The path to the file.
    name : str
        The name to describe.
    kind : str
        The kind of type to describe, valid flags are 'class', 'func', and 'var'.
    includes: list of str, optional
        The list of extra include directories to search for header files.
    defines: list of str, optional
        The list of extra macro definitions to apply.
    undefines: list of str, optional
        The list of extra macro undefinitions to apply.
    extra_parser_args : list of str, optional
        Further command line arguments to pass to the parser.
    ts : TypeSystem, optional
        A type system instance.
    verbose : bool, optional
        Flag to diplay extra information while describing the class.
    debug : bool, optional
        Flag to enable/disable debug mode.
    builddir : str, optional
        Location of -- often temporary -- build files.
    onlyin: set of str
        The paths to the files that the definition is allowed to exist in.
    language : str
        Valid language flag.
    clang_includes : ignored

    Returns
    -------
    desc : dict
        A dictionary describing the class which may be used to generate
        API bindings.
    """
    # GCC-XML and/or Cygwin wants posix paths on Windows.
    posixfilename = posixpath.join(*ntpath.split(filename)) if os.name == 'nt' \
                    else filename
    root = astparsers.gccxml_parse(posixfilename, includes=includes, defines=defines,
                                   undefines=undefines,
                                   extra_parser_args=extra_parser_args,
                                   verbose=verbose, debug=debug, builddir=builddir)
    if onlyin is None:
        onlyin = set([filename])
    describers = {'class': GccxmlClassDescriber, 'func': GccxmlFuncDescriber,
                  'var': GccxmlVarDescriber}
    describer = describers[kind](name, root, onlyin=onlyin, ts=ts, verbose=verbose)
    describer.visit()
    return describer.desc


class GccxmlBaseDescriber(object):
    """Base class used to generate descriptions via GCC-XML output.
    Sub-classes need only implement a visit() method and optionally a
    constructor.  The default visitor methods are valid for classes."""

    _funckey = None
    _describes = None

    def __init__(self, name, root=None, onlyin=None, ts=None, verbose=False):
        """Parameters
        -------------
        name : str
            The name to describe.
        root : element tree node, optional
            The root element node.
        onlyin :  str, optional
            Filename the class or struct described must live in.  Prevents
            finding elements of the same name coming from other libraries.
        ts : TypeSystem, optional
            A type system instance.
        verbose : bool, optional
            Flag to display extra information while visiting.

        """
        self.desc = {'name': name, 'namespace': None}
        self.name = name
        self.ts = ts or TypeSystem()
        self.verbose = verbose
        self._root = root
        origonlyin = onlyin
        onlyin = [onlyin] if isinstance(onlyin, basestring) else onlyin
        onlyin = set() if onlyin is None else set(onlyin)
        self.onlyin = set()
        self._filemap = {}
        for fnode in root.iterfind("File"):
            fid = fnode.attrib['id']
            fname = fnode.attrib['name']

            self._filemap[fid] = fname
        for fname in onlyin:
            fnode = root.find("File[@name='{0}']".format(fname))
            if fnode is None:
                fnode = root.find("File[@name='./{0}']".format(fname))
                if fnode is None:
                    continue
            fid = fnode.attrib['id']
            self.onlyin.add(fid)
        if 0 == len(self.onlyin):
            msg = "{0!r} is not present in {1!r}; autodescribing will probably fail."
            msg = msg.format(name, origonlyin)
            warn(msg, RuntimeWarning)
        self._currfunc = []  # this must be a stack to handle nested functions
        self._currfuncsig = None
        self._currargkind = None
        self._currclass = []  # this must be a stack to handle nested classes
        self._level = -1
        self._template_args = hack_template_args
        #self._template_args.update(ts.template_types)

    def __str__(self):
        return pformat(self.desc)

    def __del__(self):
        linecache.clearcache()

    def _pprint(self, node):
        if self.verbose:
            print("{0}{1} {2}: {3}".format(self._level * "  ", node.tag,
                                       node.attrib.get('id', ''),
                                       node.attrib.get('name', None)))

    def _template_literal_enum_val(self, targ):
        """Finds the node associated with an enum value in a template"""
        m = _GCCXML_LITERAL_ENUMS.match(targ)
        if m is None:
            return None
        enumname, val = m.groups()
        query = ".//*[@name='{0}']"
        node = self._root.find(query.format(enumname))
        if node is None:
            return None
        for child in node.iterfind('EnumValue'):
            if child.attrib['init'] == val:
                return child
        return None

    def _visit_template_function(self, node):
        nodename = node.attrib['name']
        demangled = node.attrib.get('demangled', None)
        if demangled is None:
            return (nodename,)
        # we know we have a template now
        demangled = demangled[demangled.index(nodename):]
        template_args = utils.split_template_args(demangled)
        inst = [nodename]
        self._level += 1
        targ_nodes = []
        targ_islit = []
        # gross but string parsing of node name is needed.
        query = ".//*[@name='{0}']"
        for targ in template_args:
            targ_node = self._root.find(query.format(targ))
            if targ_node is None:
                try:
                    targ_node = c_literal(targ)
                    targ_islit.append(True)
                except ValueError:
                    targ_node = self._template_literal_enum_val(targ)
                    if targ_node is None:
                        continue
                    targ_islit.append(False)
            else:
                targ_islit.append(False)
            targ_nodes.append(targ_node)
        argkinds = []  # just in case it is needed
        for targ_node, targ_lit in zip(targ_nodes, targ_islit):
            if targ_lit:
                targ_kind, targ_value = Arg.LIT, targ_node
            elif 'id' in targ_node.attrib:
                targ_kind, targ_value = Arg.TYPE, self.type(targ_node.attrib['id'])
            else:
                targ_kind, targ_value = Arg.VAR, targ_node.attrib['name']
            argkinds.append(targ_kind)
            inst.append(targ_value)
        self._level -= 1
        #inst.append(0) This doesn't apply to top-level functions, only function types
        inst = tuple(inst)
        return inst

    def _visit_template_class(self, node):
        name = node.attrib['name']
        members = node.attrib.get('members', '').strip().split()
        if 0 < len(members):
            children = [child for m in members for child in \
                        self._root.iterfind(".//*[@id='{0}']".format(m))]
            tags = [child.tag for child in children]
            template_name = children[tags.index('Constructor')].attrib['name']  # 'map'
        else:
            template_name = name.split('<', 1)[0]
        if template_name == 'basic_string':
            return 'str'
        inst = [template_name]
        self._level += 1
        targ_nodes = []
        targ_islit = []
        if template_name in self._template_args:
            for targ in self._template_args[template_name]:
                possible_targ_nodes = [c for c in children if c.attrib['name'] == targ]
                targ_nodes.append(possible_targ_nodes[0])
                targ_islit.append(False)
        else:
            # gross but string parsing of node name is needed.
            targs = utils.split_template_args(name)
            query = ".//*[@name='{0}']"
            for targ in targs:
                targ_node = self._root.find(query.format(targ))
                if targ_node is None:
                    targ_node = c_literal(targ)
                    targ_islit.append(True)
                else:
                    targ_islit.append(False)
                targ_nodes.append(targ_node)
        argkinds = []
        for targ_node, targ_lit in zip(targ_nodes, targ_islit):
            if targ_lit:
                targ_kind, targ_value = Arg.LIT, targ_node
            elif 'id' in targ_node.attrib:
                targ_kind, targ_value = Arg.TYPE, self.type(targ_node.attrib['id'])
            else:
                targ_kind, targ_value = Arg.VAR, targ_node.attrib['name']
            argkinds.append(targ_kind)
            inst.append(targ_value)
        self._level -= 1
        inst.append(0)
        inst = tuple(inst)
        self.ts.register_argument_kinds(inst, tuple(argkinds))
        return inst

    def visit_class(self, node):
        """visits a class or struct."""
        self._pprint(node)
        name = node.attrib['name']
        self._currclass.append(name)
        if self._describes == 'class' and (name == self.ts.gccxml_type(self.name) or
                                           name == self._name):
            if 'bases' not in node.attrib:
                msg = ("The type {0!r} is used as part of an API element but no "
                       "declarations were made with it.  Please declare a variable "
                       "of type {0!r} somewhere in the source or header.")
                raise NotImplementedError(msg.format(name))
            bases = node.attrib['bases'].split()
            # TODO: Record whether bases are public, private, or protected
            bases = [self.type(b.replace('private:','')) for b in bases]
            self.desc['parents'] = bases
            ns = self.context(node.attrib['context'])
            if ns is not None and ns != "::":
                self.desc['namespace'] = ns
        if '<' in name and name.endswith('>'):
            name = self._visit_template_class(node)
        self._currclass.pop()
        return name

    visit_struct = visit_class

    def visit_base(self, node):
        """visits a base class."""
        self._pprint(node)
        self.visit(node)  # Walk farther down the tree

    def _visit_func(self, node):
        name = node.attrib['name']
        if name.startswith('_') or name in FORBIDDEN_NAMES:
            warn_forbidden_name(name, self.name)
            return
        demangled = node.attrib.get('demangled', "")
        demangled = demangled if name + '<' in demangled \
                                 and '>' in demangled else None
        if demangled is None:
            # normal function
            self._currfunc.append(name)
        else:
            # template function
            self._currfunc.append(self._visit_template_function(node))
        self._currfuncsig = []
        self._currargkind = []
        self._level += 1
        for child in node.iterfind('Argument'):
            self.visit_argument(child)
        self._level -= 1
        if node.tag == 'Constructor':
            rtntype = None
        elif node.tag == 'Destructor':
            rtntype = None
            if demangled is None:
                self._currfunc[-1] = '~' + self._currfunc[-1]
            else:
                self._currfunc[-1] = ('~' + self._currfunc[-1][0],) + \
                                            self._currfunc[-1][1:]
        else:
            rtntype = self.type(node.attrib['returns'])
        funcname = self._currfunc.pop()
        if self._currfuncsig is None:
            return
        key = (funcname,) + tuple(self._currfuncsig)
        self.desc[self._funckey][key] = {'return': rtntype,
                                         'defaults': tuple(self._currargkind)}
        self._currfuncsig = None
        self._currargkind = None

    def visit_constructor(self, node):
        """visits a class constructor."""
        self._pprint(node)
        self._visit_func(node)

    def visit_destructor(self, node):
        """visits a class destructor."""
        self._pprint(node)
        self._visit_func(node)

    def visit_method(self, node):
        """visits a member function."""
        self._pprint(node)
        self._visit_func(node)

    def visit_function(self, node):
        """visits a non-member function."""
        self._pprint(node)
        self._visit_func(node)
        ns = self.context(node.attrib['context'])
        if ns is not None and ns != "::":
            self.desc['namespace'] = ns

    def visit_argument(self, node):
        """visits a constructor, destructor, or method argument."""
        self._pprint(node)
        name = node.attrib.get('name', None)
        if name is None:
            self._currfuncsig = None
            self._currargkind = None
            return
        if name in FORBIDDEN_NAMES:
            rename = name + '__'
            warn_forbidden_name(name, self.name, rename)
            name = rename
        tid = node.attrib['type']
        t = self.type(tid)
        default = node.attrib.get('default', None)
        arg = (name, t)
        if default is None:
            argkind = _none_arg
        else:
            try:
                default = c_literal(default)
                islit = True
            except ValueError:
                islit = False  # Leave default as is
            argkind = (Arg.LIT if islit else Arg.VAR, default)
        self._currfuncsig.append(arg)
        self._currargkind.append(argkind)

    def visit_field(self, node):
        """visits a member variable."""
        self._pprint(node)
        context = self._root.find(".//*[@id='{0}']".format(node.attrib['context']))
        if context.attrib['name'] == self.name:
            # assert this field is member of the class we are trying to parse
            name = node.attrib['name']
            if name in FORBIDDEN_NAMES:
                warn_forbidden_name(name, self.name)
                return
            t = self.type(node.attrib['type'])
            self.desc['attrs'][name] = t

    def visit_typedef(self, node):
        """visits a type definition anywhere."""
        self._pprint(node)
        name = node.attrib.get('name', None)
        if name == 'string':
            return 'str'
        else:
            return self.type(node.attrib['type'])

    def visit_enumeration(self, node):
        self._pprint(node)
        currenum = []
        for child in node.iterfind('EnumValue'):
            currenum.append((child.attrib['name'], child.attrib['init']))
        return ('enum', node.attrib['name'], tuple(currenum))

    def visit_fundamentaltype(self, node):
        """visits a base C++ type, mapping it to the approriate type in the
        type system."""
        self._pprint(node)
        tname = node.attrib['name']
        t = _c_to_xdress.get(tname, None)
        return t

    _predicates = frozenset(['*', '&', 'const', 'volatile', 'restrict'])

    def _add_predicate(self, baset, pred):
        if isinstance(baset, basestring):
            return (baset, pred)
        last = baset[-1]
        if last in self._predicates or isinstance(last, int):
            return (baset, pred)
        else:
            return tuple(baset) + (pred,)

    def visit_arraytype(self, node):
        """visits an array type and maps it to a '*' refinement type."""
        self._pprint(node)
        baset = self.type(node.attrib['type'])
        size = int(node.attrib['size']) / 8
        t = self._add_predicate(baset, size)
        return t

    def visit_functiontype(self, node):
        """visits an function type and returns a 'function' dependent
        refinement type."""
        self._pprint(node)
        t = ['function']
        args = []
        for i, child in enumerate(node.iterfind('Argument')):
            argt = self.type(child.attrib['type'])
            args.append(('_{0}'.format(i), argt))
        t.append(tuple(args))
        rtnt = self.type(node.attrib['returns'])
        t.append(rtnt)
        return tuple(t)

    def visit_referencetype(self, node):
        """visits a reference and maps it to a '&' refinement type."""
        self._pprint(node)
        baset = self.type(node.attrib['type'])
        t = self._add_predicate(baset, '&')
        return t

    def visit_pointertype(self, node):
        """visits a pointer and maps it to a '*' refinement type."""
        self._pprint(node)
        baset = self.type(node.attrib['type'])
        if baset[0] == 'function':
            t = ('function_pointer',) + baset[1:]
        else:
            t = self._add_predicate(baset, '*')
        return t

    def visit_cvqualifiedtype(self, node):
        """visits constant, volatile, and restricted types and maps them to
        'const', 'volatile', and 'restrict' refinement types."""
        self._pprint(node)
        t = self.type(node.attrib['type'])
        if int(node.attrib.get('const', 0)):
            t = self._add_predicate(t, 'const')
        if int(node.attrib.get('volatile', 0)):
            t = self._add_predicate(t, 'volatile')
        if int(node.attrib.get('restrict', 0)):
            t = self._add_predicate(t, 'restrict')
        return t

    def type(self, id):
        """Resolves the type from its id and information in the root element tree."""
        node = self._root.find(".//*[@id='{0}']".format(id))
        tag = node.tag.lower()
        meth_name = 'visit_' + tag
        meth = getattr(self, meth_name, None)
        t = None
        if meth is not None:
            self._level += 1
            t = meth(node)
            self._level -= 1
        return t

    def visit_namespace(self, node):
        """visits the namespace that a node is defined in."""
        self._pprint(node)
        name = node.attrib['name']
        return name

    def context(self, id):
        """Resolves the context from its id and information in the element tree."""
        node = self._root.find(".//*[@id='{0}']".format(id))
        tag = node.tag.lower()
        meth_name = 'visit_' + tag
        meth = getattr(self, meth_name, None)
        c = None
        if meth is not None:
            self._level += 1
            c = meth(node)
            self._level -= 1
        return c


class GccxmlClassDescriber(GccxmlBaseDescriber):
    """Class used to generate class descriptions via GCC-XML output."""

    _funckey = 'methods'
    _describes = 'class'
    _constructvalue = 'class'

    def __init__(self, name, root=None, onlyin=None, ts=None, verbose=False):
        """Parameters
        -------------
        name : str
            The class name, this may not have a None value.
        root : element tree node, optional
            The root element node of the class or struct to describe.
        onlyin :  str, optional
            Filename the class or struct described must live in.  Prevents
            finding classes of the same name coming from other libraries.
        ts : TypeSystem, optional
            A type system instance.
        verbose : bool, optional
            Flag to display extra information while visiting the class.

        """
        super(GccxmlClassDescriber, self).__init__(name, root=root, onlyin=onlyin,
                                                   ts=ts, verbose=verbose)
        self.desc['attrs'] = {}
        self.desc[self._funckey] = {}
        self.desc['construct'] = self._constructvalue
        self.desc['type'] = ts.canon(name)
        # Gross, but it solves the problem that for uint valued template parameters
        # (ie MyClass<3>) gccxml names this MyClass<3u> or MyClass<3d> but
        # the type system has no way of distinguising this from MyClass<3> as an int.
        # maybe it should, but I think this will get us pretty far.
        self._name = None

    def _find_class_node(self):
        basename = self.name[0]
        namet = self.desc['type']
        query = "Class"
        for node in self._root.iterfind(query):
            if node.attrib['file'] not in self.onlyin:
                continue
            nodename = node.attrib['name']
            if not nodename.startswith(basename):
                continue
            if '<' not in nodename or not nodename.endswith('>'):
                continue
            nodet = self._visit_template_class(node)
            if nodet == namet:
                self._name = nodename  # gross
                break
        else:
            node = None
        return node

    def visit(self, node=None):
        """Visits the class node and all sub-nodes, generating the description
        dictionary as it goes.

        Parameters
        ----------
        node : element tree node, optional
            The element tree node to start from.  If this is None, then the
            top-level class node is found and visited.

        """
        if node is None:
            if not isinstance(self.name, basestring) and self.name not in self.ts.argument_kinds:
                node = self._find_class_node()
            if node is None:
                query = "Class[@name='{0}']".format(self.ts.gccxml_type(self.name))
                node = self._root.find(query)
            if node is None:
                query = "Struct[@name='{0}']".format(self.ts.gccxml_type(self.name))
                node = self._root.find(query)
            if node is None:
                query = "Union[@name='{0}']".format(self.ts.gccxml_type(self.name))
                node = self._root.find(query)
            if node is None and not isinstance(self.name, basestring):
                # Must be a template with some wacky argument values
                node = self._find_class_node()
            if node is None:
                raise RuntimeError("could not find class {0!r}".format(self.name))
            if node.attrib['file'] not in self.onlyin:
                msg = ("{0} autodescribing failed: found class in {1!r} ({2!r}) but "
                       "expected it in {3}.")
                fid = node.attrib['file']
                ois = ", ".join(["{0!r} ({1!r})".format(self._filemap[v], v) \
                                                 for v in sorted(self.onlyin)])
                print("ONLYIN =", self.onlyin)
                msg = msg.format(self.name, self._filemap[fid], fid, ois)
                raise RuntimeError(msg)
            self.desc['construct'] = node.tag.lower()
            self.visit_class(node)
        members = node.attrib.get('members', '').strip().split()
        children = [self._root.find(".//*[@id='{0}']".format(m)) for m in members]
        children = [c for c in children if c.attrib['access'] == 'public']
        self._level += 1
        for child in children:
            tag = child.tag.lower()
            meth_name = 'visit_' + tag
            meth = getattr(self, meth_name, None)
            if meth is not None:
                meth(child)
        self._level -= 1

class GccxmlVarDescriber(GccxmlBaseDescriber):
    """Class used to generate variable descriptions via GCC-XML output."""

    _describes = 'var'

    def __init__(self, name, root=None, onlyin=None, ts=None, verbose=False):
        """Parameters
        -------------
        name : str
            The function name, this may not have a None value.
        root : element tree node, optional
            The root element node of the function to describe.
        onlyin :  str, optional
            Filename the function described must live in.  Prevents finding
            functions of the same name coming from other libraries.
        ts : TypeSystem, optional
            A type system instance.
        verbose : bool, optional
            Flag to display extra information while visiting the function.

        """
        super(GccxmlVarDescriber, self).__init__(name, root=root, onlyin=onlyin,
                                                 ts=ts, verbose=verbose)

    def visit(self, node=None):
        """Visits the variable node and all sub-nodes, generating the description
        dictionary as it goes.

        Parameters
        ----------
        node : element tree node, optional
            The element tree node to start from.  If this is None, then the
            top-level class node is found and visited.

        """
        root = node or self._root
        for n in root.iterfind("Variable[@name='{0}']".format(self.name)):
            if n.attrib['file'] in self.onlyin:
                ns = self.context(n.attrib['context'])
                if ns is not None and ns != "::":
                    self.desc['namespace'] = ns
                self.desc['type'] = self.type(n.attrib['type'])
                break
            else:
                msg = ("{0} autodescribing failed: found variable in {1!r} but "
                       "expected it in {2!r}.")
                msg = msg.format(self.name, node.attrib['file'], self.onlyin)
                raise RuntimeError(msg)

        # Variables can also be enums
        for n in root.iterfind("Enumeration[@name='{0}']".format(self.name)):
            if n.attrib['file'] in self.onlyin:
                ns = self.context(n.attrib['context'])
                if ns is not None and ns != "::":
                    self.desc['namespace'] = ns
                # Grab the type and put it in
                self.desc['type'] = self.visit_enumeration(n)
                break
            else:
                msg = ("{0} autodescribing failed: found variable in {1!r} but "
                       "expected it in {2!r}.")
                msg = msg.format(self.name, node.attrib['file'], self.onlyin)
                raise RuntimeError(msg)


class GccxmlFuncDescriber(GccxmlBaseDescriber):
    """Class used to generate function descriptions via GCC-XML output."""

    _funckey = 'signatures'
    _describes = 'func'

    def __init__(self, name, root=None, onlyin=None, ts=None, verbose=False):
        """Parameters
        -------------
        name : str
            The function name, this may not have a None value.
        root : element tree node, optional
            The root element node of the function to describe.
        onlyin :  str, optional
            Filename the function described must live in.  Prevents finding
            functions of the same name coming from other libraries.
        ts : TypeSystem, optional
            A type system instance.
        verbose : bool, optional
            Flag to display extra information while visiting the function.

        """
        super(GccxmlFuncDescriber, self).__init__(name, root=root, onlyin=onlyin,
                                                  ts=ts, verbose=verbose)
        self.desc[self._funckey] = {}

    def visit(self, node=None):
        """Visits the function node and all sub-nodes, generating the description
        dictionary as it goes.

        Parameters
        ----------
        node : element tree node, optional
            The element tree node to start from.  If this is None, then the
            top-level class node is found and visited.

        """
        root = node or self._root
        name = self.name
        ts = self.ts
        if isinstance(name, basestring):
            basename = name
            namet = (name,)
        else:
            # Must be a template function
            basename = name[0]
            namet = [basename]
            for x in name[1:]:
                if isinstance(x, Number):
                    pass
                else:
                    x = ts.canon(x)
                namet.append(x)
            namet = tuple(namet)
        if not isinstance(name, basestring):
            pattern = re.compile(r'(?: |::)'+basename+'.*>')
        for n in root.iterfind("Function[@name='{0}']".format(basename)):
            if not isinstance(name, basestring):
                # Must be a template function
                if n.attrib['file'] not in self.onlyin:
                    continue
                nodename = n.attrib.get('demangled', '')
                if not pattern.search(nodename):
                    continue
                nodet = self._visit_template_function(n)
                if nodet != namet:
                    continue
            if n.attrib['file'] not in self.onlyin:
                msg = ("{0} autodescribing failed: found function in {1!r} but "
                       "expected it in {2!r}.")
                msg = msg.format(name, node.attrib['file'], self.onlyin)
                raise RuntimeError(msg)
            self.visit_function(n)

#
# Clang Describers
#

def clang_describe(filename, name, kind, includes=(), defines=('XDRESS',),
                   undefines=(), extra_parser_args=(), ts=None, verbose=False,
                   debug=False, builddir=None, onlyin=None, language='c++',
                   clang_includes=()):
    """Use Clang to describe the class.

    Parameters
    ----------
    filename : str
        The path to the file.
    name : str
        The name to describe.
    kind : str
        The kind of type to describe, valid flags are 'class', 'func', and 'var'.
    includes: list of str, optional
        The list of extra include directories to search for header files.
    defines: list of str, optional
        The list of extra macro definitions to apply.
    undefines: list of str, optional
        The list of extra macro undefinitions to apply.
    extra_parser_args : list of str, optional
        Further command line arguments to pass to the parser.
    ts : TypeSystem, optional
        A type system instance.
    verbose : bool, optional
        Flag to diplay extra information while describing the class.
    debug : bool, optional
        Flag to enable/disable debug mode.  Currently ignored.
    builddir : str, optional
        Ignored.  Exists only for compatibility with gccxml_describe.
    onlyin : set of str, optional
        The paths to the files that the definition is allowed to exist in.
    language : str
        Valid language flag.

    Returns
    -------
    desc : dict
        A dictionary describing the class which may be used to generate
        API bindings.
    """
    tu = astparsers.clang_parse(filename, includes=includes, defines=defines,
                                undefines=undefines,
                                extra_parser_args=extra_parser_args, verbose=verbose,
                                debug=debug, language=language,
                                clang_includes=clang_includes)
    ts = ts or TypeSystem()
    if onlyin is None:
        onlyin = None if filename is None else frozenset([filename])
    onlyin = clang_fix_onlyin(onlyin)
    if kind == 'class':
        cls = clang_find_class(tu, name, ts=ts, filename=filename, onlyin=onlyin)
        desc = clang_describe_class(cls)
    elif kind == 'func':
        fns = clang_find_function(tu, name, ts=ts, filename=filename, onlyin=onlyin)
        desc = clang_describe_functions(fns)
    elif kind == 'var':
        var = clang_find_var(tu, name, ts=ts, filename=filename, onlyin=onlyin)
        desc = clang_describe_var(var)
    else:
        raise ValueError('bad description kind {0}, name {1}'.format(kind,name))
    linecache.clearcache() # Clean up results of clang_range_str
    return desc

def clang_fix_onlyin(onlyin):
    '''Make sure onlyin is a set and add ./path versions for each relative path'''
    if onlyin is not None:
        onlyin = set(onlyin)
        for f in tuple(onlyin):
            if not os.path.isabs(f):
                onlyin.add('./'+f)
                if os.sep != '/': # I'm not sure if clang lists paths with / or \ on windows
                    onlyin.add(os.path.join('.',f))
        onlyin = frozenset(onlyin)
    return onlyin

def clang_range_str(source_range):
    """Get the text present on a source range."""
    start = source_range.start
    stop = source_range.end
    filename = start.file.name
    if filename != stop.file.name:
        msg = 'range spans multiple files: {0!r} & {1!r}'
        msg = msg.format(filename, stop.file.name)
        raise ValueError(msg)
    lines = [linecache.getline(filename, n) for n in range(start.line, stop.line+1)]
    lines[-1] = lines[-1][:stop.column-1]  # stop slice must come first for
    lines[0] = lines[0][start.column-1:]   # len(lines) == 1
    s = "".join(lines)
    return s

def clang_find_scopes(tu, onlyin, namespace=None):
    """Find all 'toplevel' scopes, optionally restricting to a given namespace"""
    namespace_kind = CursorKind.NAMESPACE
    if namespace is None:
        def all_namespaces(node):
            for n in node.get_children():
                if n.kind == namespace_kind:
                    if onlyin is None or n.location.file.name in onlyin:
                        yield n
                    for c in all_namespaces(n):
                        yield c
        return (tu.cursor,)+tuple(all_namespaces(tu.cursor))
    else:
        scopes = []
        for n in tu.cursor.get_children():
            if n.kind == namespace_kind and n.spelling == namespace:
                if onlyin is None or n.location.file.name in onlyin:
                    scopes.append(n)
        return scopes

def clang_find_decls(tu, name, kinds, onlyin, namespace=None):
    """Find all declarations of the given name and kind in the given scopes."""
    scopes = clang_find_scopes(tu, onlyin, namespace=namespace)
    decls = []
    for s in scopes[::-1]:
        for c in s.get_children():
            if c.kind in kinds and c.spelling == name:
                if onlyin is None or c.location.file.name in onlyin:
                    decls.append(c)
    return decls

# TODO: This functionality belongs in TypeSystem
def canon_template_arg(ts, kind, arg):
    if kind == Arg.TYPE:
        return ts.canon(arg)
    return arg

def clang_where(namespace, filename):
    where = ''
    if namespace is not None:
        where += " in namespace {0}".format(namespace)
    if filename is not None:
        where += " in file {0}".format(filename)
    return where

def clang_find_class(tu, name, ts, namespace=None, filename=None, onlyin=None):
    """Find the node for a given class in the given translation unit."""
    templated = isinstance(name, tuple)
    if templated:
        basename = name[0]
        if name[-1] != 0:
            raise NotImplementedError('no predicate support in clang class description')
        args = name[1:-1]
        kinds = CursorKind.CLASS_TEMPLATE,
    else:
        basename = name
        kinds = CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL, CursorKind.UNION_DECL
    decls = clang_find_decls(tu, basename, kinds=kinds, onlyin=onlyin, namespace=namespace)
    decls = frozenset(c.get_definition() or c for c in decls) # Use definitions if available
    if len(decls)==1:
        decl, = decls
        if not templated:
            # No templates, so we're done
            return decl
        else:
            # Search for the desired template specialization
            kinds = clang_template_param_kinds(decl)
            args = tuple(canon_template_arg(ts,k,a) for k,a in zip(kinds, args))
            args = clang_expand_template_args(decl, args)
            for spec in decl.get_specializations():
                if args == tuple(canon_template_arg(ts,k,a) for k,a in zip(kinds, clang_describe_template_args(spec))):
                    return spec

    # Nothing found, time to complain
    where = clang_where(namespace, filename)
    if not decls:
        raise ValueError("class '{0}' could not be found{1}".format(name, where))
    elif len(decls)>1:
        raise ValueError("class '{0}' found more than once ({2} times) {1}".format(name, len(decls), where))
    else:
        raise ValueError("class '{0}' found, but specialization {1} not found{2}".format(basename, name, where))

def clang_find_function(tu, name, ts, namespace=None, filename=None, onlyin=None):
    """Find all nodes corresponding to a given function.  If there is a separate declaration
    and definition, they will be returned as separate nodes, in the order given in the file."""
    templated = isinstance(name, tuple)
    if templated:
        basename = name[0]
        args = name[1:]
        kinds = CursorKind.FUNCTION_TEMPLATE,
    else:
        basename = name
        kinds = CursorKind.FUNCTION_DECL,
    decls = clang_find_decls(tu, basename, kinds=kinds, onlyin=onlyin, namespace=namespace)
    if decls:
        if not templated:
            # No templates, so we're done
            return decls
        else:
            # Search for the desired function specialization
            decl, = decls # TODO: Support multiple decl case
            kinds = clang_template_param_kinds(decl)
            args = tuple(canon_template_arg(ts,k,a) for k,a in zip(kinds, args))
            args = clang_expand_template_args(decl, args)
            for spec in decl.get_specializations():
                if args == tuple(canon_template_arg(ts,k,a) for k,a in zip(kinds, clang_describe_template_args(spec))):
                    return [spec]

    # Nothing found, time to complain
    where = clang_where(namespace, filename)
    if not decls:
        raise ValueError("function '{0}' could not be found{1}".format(name, where))
    elif len(decls)>1:
        raise ValueError("function '{0}' found more than once ({2} times) {1}".format(name, len(decls), where))
    else:
        raise ValueError("function '{0}' found, but specialization {1} not found{1}".format(basename, name, where))

def clang_find_var(tu, name, ts, namespace=None, filename=None, onlyin=None):
    """Find the node for a given var."""
    assert isinstance(name, basestring)
    kinds = CursorKind.ENUM_DECL,
    decls = clang_find_decls(tu, name, kinds=kinds, onlyin=onlyin, namespace=namespace)
    decls = list(set(c.get_definition() or c for c in decls)) # Use definitions if available
    if len(decls)==1:
        return decls[0]

    # Nothing found, time to complain
    where = clang_where(namespace, filename)
    if not decls:
        raise ValueError("var '{0}' could not be found{1}".format(name, where))
    else:
        raise ValueError("var '{0}' found more than once ({2} times) {1}".format(name, len(decls), where))

def clang_dump(node, indent=0, onlyin=None, file=sys.stdout):
    try:
        spelling = node.spelling
    except AttributeError:
        spelling = '<unknown-spelling>'
    s = '%*s%s %s'%(indent,'',node.kind.name,spelling)
    if node.kind == CursorKind.CXX_ACCESS_SPEC_DECL:
        s += ' '+node.access.name
    r = node.extent
    if r.start.line == r.end.line:
        try:
            s += ' : '+clang_range_str(r)
        except AttributeError:
            pass
    print(s,file=file)
    if onlyin is None:
        for c in node.get_children():
            clang_dump(c,indent+2,file=file)
    else:
        for c in node.get_children():
            f = c.extent.start.file
            if f and f.name in onlyin:
                clang_dump(c,indent+2,file=file)

def clang_parent_namespace(node):
    if node.semantic_parent.kind == CursorKind.NAMESPACE:
        return node.semantic_parent.spelling
    # Otherwise, return none

_operator_pattern = re.compile(r'^operator\W')

def clang_describe_class(cls):
    """Describe the class at the given clang AST node"""
    if cls.get_definition() is None:
        raise ValueError("can't describe undefined class '{0}' at {1}"
            .format(cls.spelling, clang_str_location(cls.location)))
    parents = []
    attrs = {}
    methods = {}
    if cls.kind == CursorKind.CLASS_DECL:
        construct = 'class'
    elif cls.kind == CursorKind.STRUCT_DECL:
        construct = 'struct'
    elif cls.kind == CursorKind.UNION_DECL:
        construct = 'union'
    else:
        raise ValueError('bad class kind {0}'.format(cls.kind.name))
    typ = cls.spelling
    dest = '~'+typ
    templated = cls.has_template_args()
    if templated:
        cons = (typ,) + clang_describe_template_args(cls)
        dest = (dest,) + cons[1:]
        typ = cons + (0,)
    else:
        cons = typ
    for kid in cls.get_children(all_spec_bodies=1):
        kind = kid.kind
        if kind == CursorKind.CXX_BASE_SPECIFIER:
            parents.append(clang_describe_type(kid.type, kid.location))
        elif kid.access == AccessKind.PUBLIC:
            if kind == CursorKind.CXX_METHOD:
                # TODO: For now, we ignore operators
                if not _operator_pattern.match(kid.spelling):
                    sig, defaults = clang_describe_args(kid)
                    methods[sig] = {'return': clang_describe_type(kid.result_type, kid.location),
                                    'defaults': defaults}
            elif kind == CursorKind.CONSTRUCTOR:
                sig, defaults = clang_describe_args(kid)
                methods[(cons,)+sig[1:]] = {'return': None, 'defaults': defaults}
            elif kind == CursorKind.DESTRUCTOR:
                methods[(dest,)] = _none_return
            elif kind == CursorKind.FIELD_DECL:
                attrs[kid.spelling] = clang_describe_type(kid.type, kid.location)
    # Make sure defaulted methods are described
    if cls.has_default_constructor():
        # Check if any user defined constructors act as a default constructor
        for sig, info in methods.items():
            if sig[0] == cons:
                for _,default in info['defaults']:
                    if default is None:
                        break
                else:
                    # All arguments have defaults, so no need to generate a default manually
                    break
        else:
            methods[(cons,)] = _none_return
    if cls.has_simple_destructor():
        methods[(dest,)] = _none_return
    # Put everything together
    return {'name': typ, 'type': typ, 'namespace': clang_parent_namespace(cls),
            'parents': parents, 'attrs': attrs, 'methods': methods, 'construct': construct}

def clang_describe_var(var):
    """Describe the var at the given clang AST node"""
    if var.kind == CursorKind.ENUM_DECL:
        return {'name': var.spelling, 'namespace': clang_parent_namespace(var),
                'type': clang_describe_enum(var)}
    else:
        raise NotImplementedError('var kind {0}: {1}'.format(var.kind, var.spelling))

def clang_str_location(loc):
    s = '%d:%d'%(loc.line,loc.column)
    return '%s:%s'%(loc.file.name,s) if loc.file else s

def clang_describe_functions(funcs):
    """Describe the function at the given clang AST nodes.  If more than one
    node is given, we verify that they match and find argument names where we can."""
    descs = tuple(map(clang_describe_function,funcs))
    if len(descs)==1:
        return descs[0]
    def merge(d0,d1):
        """Merge two descriptions, checking that they describe the same function
        except possibly for argument name differences.  If argument names conflict,
        the name from d0 is kept."""
        def check(name,v0,v1):
            if v0 != v1:
                raise ValueError("{0} mismatch: {1} != {2}".format(name,v0,v1))
        for s in 'name','namespace':
            check(s,d0[s],d1[s])
        (args0,r0), = d0['signatures'].items()
        (args1,r1), = d1['signatures'].items()
        check('return type',r0,r1)
        check('name',args0[0],args1[0])
        check('arity',len(args0)-1,len(args1)-1)
        args = [args0[0]]
        for i,(a0,a1) in enumerate(zip(args0[1:],args1[1:])):
            check('argument %d type'%i,a0[1],a1[1])
            a = [a0[0] or a1[0],a0[1]]
            if len(a0)>2:
                if len(a1)>2:
                    check('argument %d default'%i,a0[2],a1[2])
                a.append(a0[2])
            elif len(a1)>2:
                a.append(a1[2])
            args.append(tuple(a))
        return {'name':d0['name'],'namespace':d0['namespace'],'signatures':{tuple(args):r0}}
    try:
        return reduce(merge,descs)
    except ValueError:
        for j in xrange(len(funcs)):
            for i in xrange(j):
                try:
                    merge(descs[i],descs[j])
                except ValueError as e:
                    pprint(descs[i])
                    pprint(descs[j])
                    raise ValueError("mismatch between declarations of '{0}' at {1} and {2}: {3}"
                        .format(descs[0]['name'],clang_str_location(funcs[i].location),
                                                 clang_str_location(funcs[j].location),e))
        raise

def clang_describe_function(func):
    """Describe the function at the given clang AST node."""
    assert func.kind == CursorKind.FUNCTION_DECL
    sig, defaults = clang_describe_args(func)
    signatures = {sig: {'return': clang_describe_type(func.result_type, func.location),
                        'defaults': defaults}}
    name = next(iter(signatures))[0]
    return {'name': name, 'namespace': clang_parent_namespace(func), 'signatures': signatures}

def clang_describe_args(func):
    if func.has_template_args():
        descs = [(func.spelling,) + clang_describe_template_args(func)]
    else:
        descs = [func.spelling]
    defaults = []
    for arg in func.get_arguments():
        descs.append((arg.spelling, clang_describe_type(arg.type, arg.location)))
        default = arg.default_argument
        defaults.append(_none_arg if default is None else clang_describe_expression(default))
    return tuple(descs), tuple(defaults)

def clang_describe_type(typ, loc):
    """Describe the type reference at the given cursor"""
    typ = typ.get_canonical()
    kind = typ.kind
    try:
        desc = _clang_base_types[kind]
    except KeyError:
        if kind == TypeKind.RECORD:
            decl = typ.get_declaration()
            cls = decl.spelling
            if cls == 'basic_string':
                desc = 'str'
            elif decl.has_template_args():
                desc = (cls,) + clang_describe_template_args(decl) + (0,)
            else:
                desc = cls
        elif kind == TypeKind.LVALUEREFERENCE:
            desc = (clang_describe_type(typ.get_pointee(), loc), '&')
        elif kind == TypeKind.POINTER:
            p = typ.get_pointee()
            if p.kind == TypeKind.FUNCTIONPROTO:
                desc = ('function_pointer',
                        tuple(('_{0}'.format(i),clang_describe_type(arg, loc)) for i,arg in enumerate(p.argument_types())),
                        clang_describe_type(p.get_result(), loc))
            else:
                desc = (clang_describe_type(p, loc), '*')
        elif kind == TypeKind.FUNCTIONPROTO:
            desc = ('function',
                    tuple(('_{0}'.format(i),clang_describe_type(arg, loc)) for i,arg in enumerate(typ.argument_types())),
                    clang_describe_type(typ.get_result(), loc))
        elif kind == TypeKind.ENUM:
            return clang_describe_enum(typ.get_declaration())
        elif kind == TypeKind.CONSTANTARRAY:
            array_size = typ.get_array_size()
            element_type = typ.get_array_element_type()
            desc = (clang_describe_type(element_type, loc), array_size)
        else:
            raise NotImplementedError('type kind {0}: {1} at {2}'
                .format(typ.kind, typ.spelling, clang_str_location(loc)))
    if typ.is_const_qualified():
        return (desc, 'const')
    else:
        return desc

def clang_describe_enum(decl):
    options = tuple((v.spelling, str(v.enum_value)) for v in decl.get_children())
    return ('enum', decl.spelling, options)

if 0:
    # TODO: Automatic support for default template arguments doesn't work yet
    def clang_template_arg_info(node):
        count = 0
        defaults = []
        kinds = CursorKind.TEMPLATE_TYPE_PARAMETER, CursorKind.TEMPLATE_NON_TYPE_PARAMETER
        for kid in node.get_children():
            if kid.kind in kinds:
                count += 1
                default = kid.default_argument
                if default is not None:
                    defaults.append(default)
            else:
                break
        return count,tuple(defaults)

def clang_template_param_kinds(node):
    '''Find the Arg kind of each template argument of node'''
    kinds = []
    for kid in node.get_children():
        if kid.kind == CursorKind.TEMPLATE_TYPE_PARAMETER:
            kinds.append(Arg.TYPE)
        elif kid.kind == CursorKind.TEMPLATE_NON_TYPE_PARAMETER:
            typ = clang_describe_type(kid.type, kid.location)
            kinds.append(Arg.VAR if isinstance(typ, tuple) and typ[0]=='enum' else Arg.LIT)
        else:
            # Template arguments come first, so we're done
            break
    return kinds

def clang_describe_template_args(node):
    """TODO: Broken version handling defaults
    automatically::

        _, defaults = clang_template_arg_info(node.specialized_template)
        args = [clang_describe_template_arg(a) for a in node.get_template_args()]
        for i in xrange(len(defaults)):
            if defaults[-1-i] == args[-1]:
                args.pop()
        return tuple(args)

    TODO: Needs a better docstring.
    """
    loc = node.location
    args = tuple(clang_describe_template_arg(a, loc) for a in node.get_template_args())
    if node.spelling in hack_template_args:
        return args[:len(hack_template_args[node.spelling])]
    else:
        return args

def clang_expand_template_args(node, args):
    """TODO: Broken version handling defaults
    automatically::

        count,defaults = clang_template_arg_info(node)
        print('SP %s, COUNT %s, %s, %s'%(node.spelling,count,defaults,args))
        if len(args) < count:
            return tuple(args) + defaults[(count-len(args)):]
        return tuple(args)+defaults[count-len(args)]

    TODO: Needs a better docstring.
    """
    return args

def clang_describe_template_arg(arg, loc):
    '''Describe a template argument'''
    kind = arg.kind
    if kind == CursorKind.TYPE_TEMPLATE_ARG:
        return clang_describe_type(arg.type, loc)
    try:
        s = arg.spelling.strip()
        lit = c_literal(s)
    except:
        pass
    else:
        if kind == CursorKind.INTEGRAL_TEMPLATE_ARG:
            typ = clang_describe_type(arg.type, loc)
            if isinstance(typ, tuple) and typ[0]=='enum':
                # Convert integers to enum names
                lit = str(lit)
                for n,v in typ[2]:
                    if v == lit:
                        return n
                else:
                    raise RuntimeError('template argument {0} is invalid, expected one of {1} at {2}'
                        .format(lit, ', '.join('%s=%s'%(n,v) for n,v in typ[2]), clang_str_location(loc)))
        return lit
    if kind == CursorKind.EXPRESSION_TEMPLATE_ARG:
        exp, = arg.get_children()
        if exp.referenced:
            exp = exp.referenced
        if exp.kind == CursorKind.ENUM_CONSTANT_DECL:
            return s
    # Nothing worked, so bail
    raise NotImplementedError('template argument {0}, kind {1} at {2}'
        .format(s, kind.name, clang_str_location(loc)))

def clang_describe_expression(exp):
    # For now, we just use clang_range_str to pull the expression out of the file.
    # This is because clang doesn't seem to have any mechanism for printing expressions.
    s = clang_range_str(exp.extent)
    try:
        return Arg.LIT, c_literal(s)
    except:
        pass
    if exp.referenced:
        exp = exp.referenced
    if exp.kind == CursorKind.ENUM_CONSTANT_DECL:
        return Arg.VAR, s.strip()
    # Nothing worked, so bail
    kind = exp.kind.name
    raise NotImplementedError('unhandled expression "{0}" of kind {1} at {2}'
        .format(s, kind, clang_str_location(exp.location)))


#
# pycparser Describers
#


class PycparserBaseDescriber(PycparserNodeVisitor):

    _funckey = None

    def __init__(self, name, root, onlyin=None, ts=None, verbose=False):
        """Parameters
        -------------
        name : str
            The name to describe.
        root : pycparser AST
            The root of the abstract syntax tree.
        onlyin :  str, optional
            Filename the class or struct described must live in.  Prevents
            finding classes of the same name coming from other libraries.
        ts : TypeSystem, optional
            A type system instance.
        verbose : bool, optional
            Flag to display extra information while visiting the class.

        """
        super(PycparserBaseDescriber, self).__init__()
        self.desc = {'name': name, 'namespace': None}
        self.name = name
        self.ts = ts or TypeSystem()
        self.verbose = verbose
        self._root = root
        self._currfunc = []  # this must be a stack to handle nested functions
        self._currfuncsig = None
        self._currargkind = None
        self._currclass = []  # this must be a stack to handle nested classes
        self._level = -1
        self._currtype = None
        self._currenum = None
        self._loading_bases = False
        self._basetypes = {
            'char': 'char',
            'signed char': 'char',
            'unsigned char': 'uchar',
            'short': 'int16',
            'short int': 'int16',
            'signed short': 'int16',
            'signed short int': 'int16',
            'int': 'int32',
            'signed int': 'int32',
            'long' : 'int32',
            'long int' : 'int32',
            'signed long' : 'int32',
            'signed long int' : 'int32',
            'long long' : 'int64',
            'long long int' : 'int64',
            'signed long long' : 'int64',
            'signed long long int' : 'int64',
            'unsigned short': 'uint16',
            'unsigned short int': 'uint16',
            'unsigned': 'uint32',
            'unsigned int': 'uint32',
            'unsigned long': 'uint32',
            'unsigned long int': 'uint32',
            'long unsigned int': 'uint32',
            'unsigned long long' : 'uint64',
            'unsigned long long int' : 'uint64',
            'float': 'float32',
            'double': 'float64',
            'long double': 'float128',
            'void': 'void',
            }

    def _pprint(self, node):
        if self.verbose:
            node.show()

    def load_basetypes(self):
        self._loading_bases = True
        for child_name, child in self._root.children():
            if isinstance(child, pycparser.c_ast.Typedef):
                self._basetypes[child.name] = self.type(child)
        if self.verbose:
            print("Base type mapping = ")
            pprint(self._basetypes)
        self._loading_bases = False

    def visit_FuncDef(self, node):
        self._pprint(node)
        name = node.decl.name
        ftype = node.decl.type
        if name.startswith('_') or name in FORBIDDEN_NAMES:
            warn_forbidden_name(name, self.name)
            return
        self._currfunc.append(name)
        self._currfuncsig = []
        self._currargkind = []
        self._level += 1
        children = () if ftype.args is None else ftype.args.children()
        for _, child in children:
            if isinstance(child, pycparser.c_ast.EllipsisParam):
                continue
            arg = (child.name, self.type(child))
            if arg == (None, 'void'):
                # skip foo(void) cases, since no arg name is given
                continue
            if arg[0] in FORBIDDEN_NAMES:
                rename = arg[0] + '__'
                warn_forbidden_name(arg[0], self.name, rename)
                arg = (rename, arg[1])
            self._currfuncsig.append(arg)
            self._currargkind.append(_none_arg)
        self._level -= 1
        rtntype = self.type(ftype.type)
        funcname = self._currfunc.pop()
        if self._currfuncsig is None:
            self._currargkind = None
            return
        key = (funcname,) + tuple(self._currfuncsig)
        self.desc[self._funckey][key] = {'return': rtntype,
                                         'defaults': tuple(self._currargkind)}
        self._currfuncsig = None
        self._currargkind = None

    def visit_IdentifierType(self, node):
        self._pprint(node)
        t = " ".join(node.names)
        t = self._basetypes.get(t, t)
        self._currtype = t

    def visit_Decl(self, node):
        self._pprint(node)
        if isinstance(node.type, pycparser.c_ast.FuncDecl):
            self.visit(node.type)
            key = (node.name,) + self._currtype[1]
            self.desc[self._funckey][key] = {'return': self._currtype[2],
                'defaults': (_none_arg,) * len(self._currtype[1])}
            self._currtype = None
        else:
            self.visit(node.type)

    def visit_TypeDecl(self, node):
        self._pprint(node)
        if hasattr(node.type, 'children'):
            self.visit(node.type)

    def _enumint(self, value):
        base = 10
        if value.startswith('0x') or value.startswith('-0x') or \
           value.startswith('+0x'):
            base = 16
        return int(value, base)

    def visit_Enumerator(self, node):
        self._pprint(node)
        if node.value is None:
            if len(self._currenum) == 0:
                value = 0
            else:
                value = self._currenum[-1][-1] + 1
        elif isinstance(node.value, pycparser.c_ast.Constant):
            value = self._enumint(node.value.value)
        elif isinstance(node.value, pycparser.c_ast.UnaryOp):
            if not isinstance(node.value.expr, pycparser.c_ast.Constant):
                raise ValueError("non-contant enum values not yet supported")
            value = self._enumint(node.value.op + node.value.expr.value)
        else:
            value = node.value
        self._currenum.append((node.name, value))

    def visit_Enum(self, node):
        self._pprint(node)
        self._currenum = []
        for _, child in node.children():
            self.visit(child)
        self._currtype = ('enum', node.name, tuple(self._currenum))
        self._currenum = None

    def visit_PtrDecl(self, node):
        self._pprint(node)
        self.visit(node.type)
        if self._currtype is not None and self._currtype[0] == 'function':
            self._currtype = ('function_pointer',) + self._currtype[1:]
        else:
            self._currtype = (self._currtype, '*')

    def visit_ArrayDecl(self, node):
        self._pprint(node)
        self.visit(node.type)
        predicate = '*' if node.dim is None else int(node.dim.value)
        self._currtype = (self._currtype, predicate)

    def visit_FuncDecl(self, node):
        self._pprint(node)
        args = []
        params = () if node.args is None else node.args.params
        for i, arg in enumerate(params):
            if isinstance(arg, pycparser.c_ast.EllipsisParam):
                continue
            argname = arg.name or '_{0}'.format(i)
            argtype = self.type(arg, safe=True)
            args.append((argname, argtype))
        rtntype = self.type(node.type, safe=True)
        self._currtype = ('function', tuple(args), rtntype)

    def visit_Struct(self, node):
        self._pprint(node)
        name = node.name
        if name is None:
            name = "<name-not-found>"
        self._currtype = name

    def visit_Typedef(self, node):
        self._pprint(node)
        self._currtype = None
        if not hasattr(node.type, 'children'):
            return
        self.visit(node.type)
        name = self._currtype
        self._currtype = None
        if name is None:
            return
        if name == "<name-not-found>":
            name = node.name
        self._currtype = name


    def type(self, node, safe=False):
        self._pprint(node)
        if safe:
            stashtype = self._currtype
            self._currtype = None
        self.visit(node)
        t = self._currtype
        self._currtype = None
        if safe:
            self._currtype = stashtype
        return t

    def visit_members(self, node):
        self._pprint(node)
        for _, child in node.children():
            name = child.name
            if name.startswith('_') or name in FORBIDDEN_NAMES:
                warn_forbidden_name(name, self.name)
                continue
            t = self.type(child)
            if t == "<name-not-found>":
                msg = ("autodescribe: warning: anonymous struct members not "
                       "yet supported, found {0}.{1}")
                print(msg.format(self.name, name))
                continue
            elif t == ("<name-not-found>", '*'):
                msg = ("autodescribe: warning: anonymous struct pointer members "
                       "not yet supported, found {0}.{1}")
                print(msg.format(self.name, name))
                continue
            self.desc['attrs'][name] = t

class PycparserVarDescriber(PycparserBaseDescriber):

    _type_error_msg = "{0} is a {1}, use {2} instead."

    def __init__(self, name, root, onlyin=None, ts=None, verbose=False):
        """Parameters
        -------------
        name : str
            The variable name.
        root : pycparser AST
            The root of the abstract syntax tree.
        onlyin :  str, optional
            Filename the variable described must live in.  Prevents
            finding variables of the same name coming from other libraries.
        ts : TypeSystem, optional
            A type system instance.
        verbose : bool, optional
            Flag to display extra information while visiting the class.

        """
        super(PycparserVarDescriber, self).__init__(name, root, onlyin=onlyin,
                                                    ts=ts, verbose=verbose)

    def visit(self, node=None):
        """Visits the variable definition node and all sub-nodes, generating
        the description dictionary as it goes.

        Parameters
        ----------
        node : element tree node, optional
            The element tree node to start from.  If this is None, then the
            top-level variable node is found and visited.

        """
        if node is None:
            self.load_basetypes()
            for child_name, child in self._root.children():
                if getattr(child, 'name', None) == self.name:
                    if isinstance(child, pycparser.c_ast.FuncDef):
                        raise TypeError(self._type_error_msg.format(
                            self.name, 'function', 'PycparserFuncDescriber'))
                    if isinstance(child, pycparser.c_ast.Struct):
                        raise TypeError(self._type_error_msg.format(
                            self.name, 'struct', 'PycparserClassDescriber'))
                    self.desc['type'] = self.type(child)
                    break
        else:
            super(PycparserVarDescriber, self).visit(node)

class PycparserFuncDescriber(PycparserBaseDescriber):

    _funckey = 'signatures'

    def __init__(self, name, root, onlyin=None, ts=None, verbose=False):
        """Parameters
        -------------
        name : str
            The function name.
        root : pycparser AST
            The root of the abstract syntax tree.
        onlyin :  str, optional
            Filename the class or struct described must live in.  Prevents
            finding classes of the same name coming from other libraries.
        ts : TypeSystem, optional
            A type system instance.
        verbose : bool, optional
            Flag to display extra information while visiting the class.

        """
        super(PycparserFuncDescriber, self).__init__(name, root, onlyin=onlyin,
                                                     ts=ts, verbose=verbose)
        self.desc[self._funckey] = {}

    def visit(self, node=None):
        """Visits the function node and all sub-nodes, generating the description
        dictionary as it goes.

        Parameters
        ----------
        node : element tree node, optional
            The element tree node to start from.  If this is None, then the
            top-level class node is found and visited.

        """
        if node is None:
            self.load_basetypes()
            for child_name, child in self._root.children():
                if isinstance(child, pycparser.c_ast.FuncDef) and \
                   child.decl.name == self.name:
                    self.visit(child)
                elif isinstance(child, pycparser.c_ast.Decl) and \
                     child.name == self.name:
                    self.visit(child)
        else:
            super(PycparserFuncDescriber, self).visit(node)

class PycparserClassDescriber(PycparserBaseDescriber):

    _funckey = 'methods'

    def __init__(self, name, root, onlyin=None, ts=None, verbose=False):
        """Parameters
        -------------
        name : str
            The name to describe.
        root : pycparser AST
            The root of the abstract syntax tree.
        onlyin :  str, optional
            Filename the class or struct described must live in.  Prevents
            finding classes of the same name coming from other libraries.
        ts : TypeSystem, optional
            A type system instance.
        verbose : bool, optional
            Flag to display extra information while visiting the class.

        Notes
        -----
        It is impossible for C structs to have true member functions, only
        function pointers.

        """
        super(PycparserClassDescriber, self).__init__(name, root, onlyin=onlyin,
                                                      ts=ts, verbose=verbose)
        self.desc['attrs'] = {}
        self.desc[self._funckey] = {}
        self.desc['parents'] = []
        self.desc['type'] = ts.canon(name)

    def visit(self, node=None):
        """Visits the struct (class) node and all sub-nodes, generating the
        description dictionary as it goes.

        Parameters
        ----------
        node : element tree node, optional
            The element tree node to start from.  If this is None, then the
            top-level struct (class) node is found and visited.

        """
        construct_typemap = {
            pycparser.c_ast.Struct: 'struct',
            pycparser.c_ast.Union: 'union',
        }
        construct_types = tuple(construct_typemap.keys())
        if node is None:
            self.load_basetypes()
            for child_name, child in self._root.children():
                if isinstance(child, pycparser.c_ast.Typedef) and \
                   isinstance(child.type, pycparser.c_ast.TypeDecl) and \
                   isinstance(child.type.type, construct_types):
                    child = child.type.type
                if not isinstance(child, construct_types):
                    continue
                if child.name != self.name:
                    continue
                self.desc['construct'] = construct_typemap[type(child)]
                self.visit_members(child)
        else:
            super(PycparserClassDescriber, self).visit(node)

_pycparser_describers = {
    'var': PycparserVarDescriber,
    'func': PycparserFuncDescriber,
    'class': PycparserClassDescriber,
    }

def pycparser_describe(filename, name, kind, includes=(), defines=('XDRESS',),
                       undefines=(), extra_parser_args=(), ts=None, verbose=False,
                       debug=False, builddir='build', onlyin=None, language='c',
                       clang_includes=()):
    """Use pycparser to describe the fucntion or struct (class).

    Parameters
    ----------
    filename : str
        The path to the file.
    name : str
        The name to describe.
    kind : str
        The kind of type to describe, valid flags are 'class', 'func', and 'var'.
    includes: list of str, optional
        The list of extra include directories to search for header files.
    defines: list of str, optional
        The list of extra macro definitions to apply.
    undefines: list of str, optional
        The list of extra macro undefinitions to apply.
    extra_parser_args : list of str, optional
        Further command line arguments to pass to the parser.
    ts : TypeSystem, optional
        A type system instance.
    verbose : bool, optional
        Flag to diplay extra information while describing the class.
    debug : bool, optional
        Flag to enable/disable debug mode.
    builddir : str, optional
        Location of -- often temporary -- build files.
    onlyin : set of str
        The paths to the files that the definition is allowed to exist in.
    language : str
        Must be 'c'.
    clang_includes : ignored

    Returns
    -------
    desc : dict
        A dictionary describing the class which may be used to generate
        API bindings.
    """
    assert language=='c'
    root = astparsers.pycparser_parse(filename, includes=includes, defines=defines,
                                      undefines=undefines,
                                      extra_parser_args=extra_parser_args,
                                      verbose=verbose, debug=debug, builddir=builddir)
    if onlyin is None:
        onlyin = set([filename])
    describer = _pycparser_describers[kind](name, root, onlyin=onlyin, ts=ts,
                                            verbose=verbose)
    describer.visit()
    return describer.desc


#
#  General utilities
#

def _make_includer(filenames, builddir, language, verbose=False):
    """Creates a source file made up of #include pre-processor statements for
    all of the files in filenames.  Returns the path to the newly made file.
    """
    newfile = ""
    newnames = []
    for filename in filenames:
        newnames.append(filename.replace(os.path.sep, '_'))
        newfile += '#include "{0}"\n'.format(filename)
    newnames = "-".join(newnames)
    if len(newnames) > 250:
        # this is needed to prevent 'IOError: [Errno 36] File name too long'
        newnames = md5(newnames).hexdigest()
    newname = os.path.join(builddir, newnames + '.' + _lang_exts[language])
    newoverwrite(newfile, newname, verbose=verbose)
    return newname

_describers = {
    'clang': clang_describe,
    'gccxml': gccxml_describe,
    'pycparser': pycparser_describe,
    }

def describe(filename, name=None, kind='class', includes=(), defines=('XDRESS',),
             undefines=(), extra_parser_args=(), parsers='gccxml', ts=None,
             verbose=False, debug=False, builddir='build', language='c++',
             clang_includes=()):
    """Automatically describes an API element in a file.  This is the main entry point.

    Parameters
    ----------
    filename : str or container of strs
        The path to the file or a list of file paths.  If this is a list to many
        files, a temporary file will be created that #includes all of the files
        in this list in order.  This temporary file is the one which will be
        parsed.
    name : str
        The name to describe.
    kind : str, optional
        The kind of type to describe, valid flags are 'class', 'func', and 'var'.
    includes: list of str, optional
        The list of extra include directories to search for header files.
    defines: list of str, optional
        The list of extra macro definitions to apply.
    undefines: list of str, optional
        The list of extra macro undefinitions to apply.
    extra_parser_args : list of str, optional
        Further command line arguments to pass to the parser.
    parsers : str, list, or dict, optional
        The parser / AST to use to use for the file.  Currently 'clang', 'gccxml',
        and 'pycparser' are supported, though others may be implemented in the
        future.  If this is a string, then this parser is used.  If this is a list,
        this specifies the parser order to use based on availability.  If this is
        a dictionary, it specifies the order to use parser based on language, i.e.
        ``{'c' ['pycparser', 'gccxml'], 'c++': ['gccxml', 'pycparser']}``.
    ts : TypeSystem, optional
        A type system instance.
    verbose : bool, optional
        Flag to diplay extra information while describing the class.
    debug : bool, optional
        Flag to enable/disable debug mode.
    builddir : str, optional
        Location of -- often temporary -- build files.
    language : str
        Valid language flag.
    clang_includes : list of str, optional
        clang-specific include paths.

    Returns
    -------
    desc : dict
        A dictionary describing the class which may be used to generate
        API bindings.
    """
    if isinstance(filename, basestring):
        onlyin = set([filename])
    else:
        onlyin = set(filename)
        filename = filename[0] if len(filename) == 0 \
                   else _make_includer(filename, builddir, language, verbose=verbose)
    if name is None:
        name = os.path.split(filename)[-1].rsplit('.', 1)[0].capitalize()
    parser = astparsers.pick_parser(language, parsers)
    describer = _describers[parser]
    desc = describer(filename, name, kind, includes=includes, defines=defines,
                     undefines=undefines, extra_parser_args=extra_parser_args, ts=ts,
                     verbose=verbose, debug=debug, builddir=builddir, onlyin=onlyin,
                     language=language, clang_includes=clang_includes)
    return desc


#
# Plugin
#

class XDressPlugin(astparsers.ParserPlugin):
    """This plugin creates automatic description dictionaries of all souce and
    target files."""

    def __init__(self):
        super(XDressPlugin, self).__init__()
        self.pysrcenv = {}

    def defaultrc(self):
        """This plugin adds the env dictionary to the rc."""
        rc = RunControl()
        rc._update(super(XDressPlugin, self).defaultrc)
        # target enviroment made up of module dicts made up of descriptions
        rc.env = {}
        return rc

    def rcdocs(self):
        """This plugin adds the env dictionary to the rc."""
        docs = {}
        docs.update(super(XDressPlugin, self).rcdocs)
        docs['env'] = "The target environment computed by the autodescriber."
        return docs

    def setup(self, rc):
        """Expands variables, functions, and classes in the rc based on
        copying src filenames to tar filename."""
        super(XDressPlugin, self).setup(rc)
        for i, var in enumerate(rc.variables):
            rc.variables[i] = ensure_apiname(var)
        for i, fnc in enumerate(rc.functions):
            rc.functions[i] = ensure_apiname(fnc)
        for i, cls in enumerate(rc.classes):
            rc.classes[i] = cls = ensure_apiname(cls)
            if not isinstance(cls.srcname, basestring) and cls.srcname[-1] is not 0:
                # ensure the predicate is a scalar for template specializations
                rc.classes[i] = cls = cls._replace(srcname=tuple(cls.srcname) + (0,))
            if not isinstance(cls.tarname, basestring) and cls.tarname[-1] is not 0:
                # ensure the predicate is a scalar for template specializations
                rc.classes[i] = cls = cls._replace(tarname=tuple(cls.tarname) + (0,))
        if 'make_dtypes' not in rc:
            rc.make_dtypes = False
        self.register_classes(rc)

    def execute(self, rc):
        print("autodescribe: scraping C/C++ APIs from source")
        self.load_sidecars(rc)
        self.compute_classes(rc)
        self.compute_functions(rc)
        self.compute_variables(rc)

    def report_debug(self, rc):
        super(XDressPlugin, self).report_debug(rc)

    # Helper methods below

    def register_classes(self, rc):
        """Registers classes with the type system.  This can and should be done
        trying to describe the class."""
        ts = rc.ts
        for i, cls in enumerate(rc.classes):
            print("autodescribe: registering {0}".format(cls.srcname))
            fnames = extra_filenames(cls)
            ts.register_classname(cls.srcname, rc.package, fnames['pxd_base'],
                                  fnames['cpppxd_base'], make_dtypes=rc.make_dtypes)
            if cls.srcname != cls.tarname:
                ts.register_classname(cls.tarname, rc.package, fnames['pxd_base'],
                                      fnames['cpppxd_base'], cpp_classname=cls.srcname,
                                      make_dtypes=rc.make_dtypes)

    def load_pysrcmod(self, sidecar, rc):
        """Loads a module dictionary from a sidecar file into the pysrcenv cache."""
        if sidecar in self.pysrcenv:
            return
        if os.path.isfile(sidecar):
            glbs = globals()
            locs = {}
            exec_file(sidecar, glbs, locs)
            if 'mod' not in locs:
                pymod = {}
            elif callable(locs['mod']):
                pymod = eval('mod()', glbs, locs)
            else:
                pymod = locs['mod']
            if 'ts' in locs:
                rc.ts.update(locs['ts'])
            elif 'type_system' in locs:
                rc.ts.update(locs['type_system'])
        else:
            pymod = {}
        self.pysrcenv[sidecar] = pymod

    def load_sidecars(self, rc):
        """Loads all sidecar files."""
        sidecars = set()
        for x in rc.variables:
            sidecars.update(x.sidecars)
        for x in rc.functions:
            sidecars.update(x.sidecars)
        for x in rc.classes:
            sidecars.update(x.sidecars)
        for sidecar in sidecars:
            self.load_pysrcmod(sidecar, rc)

    def compute_desc(self, name, kind, rc):
        """Returns a description dictionary for a class or function
        implemented in a source file and bound into a target file.

        Parameters
        ----------
        name : apiname
            API element name to describe.
        kind : str
            The kind of type to describe, valid flags are 'class', 'func', and 'var'.
        rc : xdress.utils.RunControl
            Run contoler for this xdress execution.

        Returns
        -------
        desc : dict
            Description dictionary.

        """
        cache = rc._cache
        if cache.isvalid(name, kind):
            srcdesc = cache[name, kind]
        else:
            srcdesc = describe(name.srcfiles, name=name.srcname, kind=kind,
                               includes=rc.includes, defines=rc.defines,
                               undefines=rc.undefines,
                               extra_parser_args=rc.extra_parser_args,
                               parsers=rc.parsers, ts=rc.ts, verbose=rc.verbose,
                               debug=rc.debug, builddir=rc.builddir,
                               language=name.language,
                               clang_includes=rc.clang_includes)
            srcdesc['name'] = dict(zip(name._fields, name))
            cache[name, kind] = srcdesc
        descs = [srcdesc]
        descs += [self.pysrcenv[s].get(name.srcname, {}) for s in name.sidecars]
        descs.append({'extra': extra_filenames(name)})
        desc = merge_descriptions(descs)
        return desc

    _extrajoinkeys = ['pxd_header', 'pxd_footer', 'pyx_header', 'pyx_footer',
                      'cpppxd_header', 'cpppxd_footer']

    def adddesc2env(self, desc, env, name):
        """Adds a description to environment."""
        # Add to target environment
        # docstrings overwrite, extras accrete
        docs = [self.pysrcenv[s].get(name.srcname, {}).get('docstring', '') \
                for s in name.sidecars]
        docs = "\n\n".join([d for d in docs if len(d) > 0])
        mod = {name.tarname: desc,
               'docstring': docs,
               'srcpxd_filename': desc['extra']['srcpxd_filename'],
               'pxd_filename': desc['extra']['pxd_filename'],
               'pyx_filename': desc['extra']['pyx_filename'],
               'language': name.language,
               }
        tarbase = name.tarbase
        extrajoinkeys = self._extrajoinkeys
        if tarbase not in env:
            env[tarbase] = mod
            env[tarbase]["name"] = tarbase
            env[tarbase]['extra'] = modextra = dict(zip(extrajoinkeys,
                                                        ['']*len(extrajoinkeys)))
        else:
            #env[tarbase].update(mod)
            env[tarbase][name.tarname] = desc
            modextra = env[tarbase]['extra']
        for sidecar in name.sidecars:
            pyextra = self.pysrcenv[sidecar].get(name.srcname, {}).get('extra', {})
            for key in extrajoinkeys:
                modextra[key] += pyextra.get(key, '')

    def compute_variables(self, rc):
        """Computes variables descriptions and loads them into the environment."""
        ts = rc.ts
        env = rc.env
        cache = rc._cache
        for i, var in enumerate(rc.variables):
            print("autodescribe: describing {0}".format(var.srcname))
            desc = self.compute_desc(var, 'var', rc)
            if rc.verbose:
                pprint(desc)
            cache.dump()
            self.adddesc2env(desc, env, var)
            ts.register_variable_namespace(desc['name']['srcname'], desc['namespace'],
                                           desc['type'])
            if 0 == i%rc.clear_parser_cache_period:
                astparsers.clearmemo()

    def compute_functions(self, rc):
        """Computes function descriptions and loads them into the environment."""
        env = rc.env
        cache = rc._cache
        for i, fnc in enumerate(rc.functions):
            print("autodescribe: describing {0}".format(fnc.srcname))
            desc = self.compute_desc(fnc, 'func', rc)
            if rc.verbose:
                pprint(desc)
            cache.dump()
            self.adddesc2env(desc, env, fnc)
            if 0 == i%rc.clear_parser_cache_period:
                astparsers.clearmemo()

    def compute_classes(self, rc):
        """Computes class descriptions and loads them into the environment."""
        # compute all class descriptions first
        cache = rc._cache
        env = rc.env  # target environment, not source one
        for i, cls in enumerate(rc.classes):
            print("autodescribe: describing {0}".format(cls.srcname))
            desc = self.compute_desc(cls, 'class', rc)
            cache.dump()
            if rc.verbose:
                pprint(desc)
            self.adddesc2env(desc, env, cls)
            if 0 == i%rc.clear_parser_cache_period:
                astparsers.clearmemo()

