"""This module creates abstract syntax trees using external tools 
(GCC-XML, pycparser) of C/C++ code.

:author: Anthony Scopatz <scopatz@gmail.com>

AST Parsers API
==========================
"""
from __future__ import print_function
import os
import io
import sys
from copy import deepcopy
import linecache
import subprocess
import itertools
import tempfile
import functools
import collections
from pprint import pprint, pformat
from warnings import warn
import gzip
try:
    import cPickle as pickle
except ImportError:
    import pickle

if os.name == 'nt':
    import ntpath
    import posixpath

# GCC-XML conditional imports
HAVE_LXML = False
try:
    from lxml import etree
    HAVE_LXML = True
except ImportError:
    try:
        # Python 2.5
        import xml.etree.cElementTree as etree
    except ImportError:
        try:
          # Python 2.5
          import xml.etree.ElementTree as etree
        except ImportError:
            try:
                # normal cElementTree install
                import cElementTree as etree
            except ImportError:
                try:
                  # normal ElementTree install
                  import elementtree.ElementTree as etree
                except ImportError:
                    pass

# pycparser conditional imports
try:
    import pycparser
    PycparserNodeVisitor = pycparser.c_ast.NodeVisitor
except ImportError:
    pycparser = None
    PycparserNodeVisitor = object  # fake this for class definitions

from . import utils
from .utils import guess_language, RunControl, NotSpecified
from .plugins import Plugin

PARSERS_AVAILABLE = {
    'clang': False, 
    'pycparser': pycparser is not None,
    }
with tempfile.NamedTemporaryFile() as f:
    # If gccxml is not availble, an OSError is raised.  Otherwise, it will
    # return 0 (typically indicates successful invocation).
    try:
        retcode = subprocess.call(['gccxml'], stdout=f, stderr=f)
        if retcode == 0:
            PARSERS_AVAILABLE['gccxml'] = True
        else:
            PARSERS_AVAILABLE['gccxml'] = False
    except OSError:
        PARSERS_AVAILABLE['gccxml'] = False
del f

if sys.version_info[0] >= 3: 
    basestring = str

def _makekey(obj):
    if isinstance(obj, basestring):
        return obj
    elif isinstance(obj, collections.Sequence):
        return tuple([_makekey(o) for o in obj])
    elif isinstance(obj, collections.Set):
        return frozenset([_makekey(o) for o in obj])
    elif isinstance(obj, collections.Mapping):
        return tuple([(_makekey(k), _makekey(v)) for k, v in sorted(obj.items())])
    else:
        return obj

def _memoize_parser(f):
    # based off code from http://wiki.python.org/moin/PythonDecoratorLibrary
    cache = f.cache = {}
    @functools.wraps(f)
    def memoizer(*args, **kwargs):
        key = _makekey(args) + _makekey(kwargs)
        if key in cache:
            value = cache[key]
        else:
            value = f(*args, **kwargs) 
            try:
                cache[key] = value
            except TypeError:
                pass
        return value
    return memoizer

def clearmemo():
    """Clears all function memoizations for autodescribers."""
    for x in globals().values():
        if callable(x) and hasattr(x, 'cache'):
            x.cache.clear()

def not_implemented(obj):
    if not isinstance(obj, type):
        if obj.__doc__ is None:
            obj.__doc__ = ''
        obj.__doc__ += ("\n\n.. warning:: This has not yet been implemented "
                        "fully or at all.\n\n")
    @functools.wraps(obj)
    def func(*args, **kwargs):
        msg = "The functionality in {0} has not been implemented fully or at all"
        msg = msg.format(obj)
        raise NotImplementedError(msg)
    return func

#
# GCC-XML Describers
#

@_memoize_parser
def gccxml_parse(filename, includes=(), defines=('XDRESS',), undefines=(), 
                  verbose=False, debug=False, builddir='build'):
    """Use GCC-XML to parse a file. This function is automatically memoized.

    Parameters
    ----------
    filename : str
        The path to the file.
    includes: list of str, optional
        The list of extra include directories to search for header files.
    defines: list of str, optional
        The list of extra macro definitions to apply.
    undefines: list of str, optional
        The list of extra macro undefinitions to apply.
    verbose : bool, optional
        Flag to diplay extra information while describing the class.
    debug : bool, optional
        Flag to enable/disable debug mode.
    builddir : str, optional
        Location of -- often temporary -- build files.

    Returns
    -------
    root : XML etree
        An in memory tree representing the parsed file.
    """
    xmlname = filename.replace(os.path.sep, '_').rsplit('.', 1)[0] + '.xml'
    xmlname = os.path.join(builddir, xmlname)
    cmd = ['gccxml', filename, '-fxml=' + xmlname]
    cmd += ['-I' + i for i in includes]
    cmd += ['-D' + d for d in defines]
    cmd += ['-U' + u for u in undefines]
    if verbose:
        print(" ".join(cmd))
    if os.path.isfile(xmlname):
        f = io.open(xmlname, 'r+b')
    else:
        f = io.open(xmlname, 'w+b')
        subprocess.call(cmd)
    f.seek(0)
    root = etree.parse(f)
    f.close()
    return root

#
# pycparser Describers
#

@_memoize_parser
def pycparser_parse(filename, includes=(), defines=('XDRESS',), undefines=(), 
                    verbose=False, debug=False, builddir='build'):
    """Use pycparser to parse a file.  This functions is automatically memoized.

    Parameters
    ----------
    filename : str
        The path to the file.
    includes: list of str, optional
        The list of extra include directories to search for header files.
    defines: list of str, optional
        The list of extra macro definitions to apply.
    undefines: list of str, optional
        The list of extra macro undefinitions to apply.
    verbose : bool, optional
        Flag to diplay extra information while describing the class.
    debug : bool, optional
        Flag to enable/disable debug mode.
    builddir : str, optional
        Location of -- often temporary -- build files.

    Returns
    -------
    root : AST
        A pycparser abstract syntax tree.

    """
    pklgzname = filename.replace(os.path.sep, '_').rsplit('.', 1)[0] + '.pkl.gz'
    pklgzname = os.path.join(builddir, pklgzname)
    if os.path.isfile(pklgzname):
        with gzip.open(pklgzname, 'rb') as f:
            root = pickle.loads(f.read())
        return root
    kwargs = {'cpp_args': [r'-D__attribute__(x)=',  # Workaround for GNU libc
                r'-D__asm__(x)=', r'-D__const=', 
                r'-D__builtin_va_list=int', # just fake this
                r'-D__restrict=', r'-D__extension__=', 
                r'-D__inline__=', r'-D__inline=',
                ]}
    kwargs['cpp_args'] += ['-I' + i for i in includes]
    kwargs['cpp_args'] += ['-D' + d for d in defines]
    kwargs['cpp_args'] += ['-U' + d for u in undefines]
    root = pycparser.parse_file(filename, use_cpp=True, **kwargs)
    with gzip.open(pklgzname, 'wb') as f:
        f.write(pickle.dumps(root, pickle.HIGHEST_PROTOCOL))
    return root

#
#  General utilities
#

def pick_parser(filename, parsers):
    """Determines the parse to use for a file.

    Parameters
    ----------
    filename : str
        The path to the file.
    parsers : str, list, or dict, optional
        The parser / AST to use to use for the file.  Currently 'clang', 'gccxml', 
        and 'pycparser' are supported, though others may be implemented in the 
        future.  If this is a string, then this parser is used.  If this is a list, 
        this specifies the parser order to use based on availability.  If this is
        a dictionary, it specifies the order to use parser based on language, i.e.
        ``{'c' ['pycparser', 'gccxml'], 'c++': ['gccxml', 'pycparser']}``.

    Returns
    -------
    parser : str
        The name of the parser to use.

    """
    if isinstance(parsers, basestring):
        parser = parsers
    elif isinstance(parsers, collections.Sequence):
        ps = [p for p in parsers if PARSERS_AVAILABLE[p.lower()]]
        if len(ps) == 0:
            msg = "Parsers not available: {0}".format(", ".join(parsers))
            raise RuntimeError(msg)
        parser = ps[0].lower()
    elif isinstance(parsers, collections.Mapping):
        lang = guess_language(filename)
        ps = parsers[lang]
        ps = [p for p in ps if PARSERS_AVAILABLE[p.lower()]]
        if len(ps) == 0:
            msg = "{0} parsers not available: {1}"
            msg = msg.format(lang.capitalize(), ", ".join(parsers))
            raise RuntimeError(msg)
        parser = ps[0].lower()
    else:
        raise ValueError("type of parsers not intelligible")
    return parser

def _pformat_etree_inplace(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            _pformat_etree_inplace(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def dumpast(filename, parsers, sourcedir, includes=(), defines=('XDRESS',), 
            undefines=(), verbose=False, debug=False, builddir='build'):
    """Prints an abstract syntax tree to stdout."""
    if not os.path.isfile(filename):
        filename = os.path.join(sourcedir, filename)
        if not os.path.isfile(filename):
            sys.exit(filename + " is not a regular file")
    parser = pick_parser(filename, parsers)
    if parser == 'pycparser':
        root = pycparser_parse(filename, includes=includes, defines=defines, 
                               undefines=undefines, verbose=verbose, debug=debug, 
                               builddir=builddir)
        root.show()
    elif parser == 'gccxml':
        root = gccxml_parse(filename, includes=includes, defines=defines, 
                            undefines=undefines, verbose=verbose, debug=debug, 
                            builddir=builddir)
        if HAVE_LXML:
            print(etree.tostring(root, pretty_print=True))
        else:
            _pformat_etree_inplace(root)
            print(etree.tostring(root))
    else:
        sys.exit(parser + " is not a valid parser")
    
    

#
# Plugin
#

class ParserPlugin(Plugin):
    """This is a base plugin for tools that wish to wrap parsing.
    It should not be used directly."""

    requires = ('xdress.base',)
    """This plugin requires 'xdress.base'."""
    
    defaultrc = utils.RunControl(
        includes=[],
        defines=["XDRESS"],
        undefines=[],
        variables=(),
        functions=(),
        classes=(),
        parsers={'c': ['pycparser', 'gccxml', 'clang'],
                 'c++':['gccxml', 'clang', 'pycparser']},
        clear_parser_cache_period=50,
        dumpast=NotSpecified,
        )

    rcupdaters = {'includes': lambda old, new: list(new) + list(old)}

    rcdocs = {
        'includes': "Additional include directories", 
        'defines': "Set additional macro definitions",
        'undefines': "Unset additional macro definitions",
        'variables': ("A list of variable names in sequence, mapping, "
                      "or apiname format"),
        'functions': ("A list of function names in sequence, mapping, "
                      "or apiname format"),
        'classes': ("A list of class names in sequence, mapping, "
                    "or apiname format"),
        'parsers': "Parser(s) name, list, or dict",
        'clear_parser_cache_period': ("Number of parser calls to perform before "
                                      "clearing the internal cache.  This prevents "
                                      "nasty memory overflow issues."),
        'dumpast': "Prints the abstract syntax tree of a file.", 
        }

    def update_argparser(self, parser):
        rcdocs = self.rcdocs() if callable(self.rcdocs) else self.rcdocs
        parser.add_argument('-I', '--includes', action='store', dest='includes', 
                            nargs="+", help=rcdocs["includes"])
        parser.add_argument('-D', '--defines', action='append', dest='defines', 
                            nargs="+", help=rcdocs["defines"])
        parser.add_argument('-U', '--undefines', action='append', dest='undefines',
                            nargs="+", type=str, help=rcdocs["undefines"])
        parser.add_argument('-p', action='store', dest='parsers', 
                            help=rcdocs["parsers"])
        parser.add_argument('--clear-parser-cache-period', action='store', 
                            dest='clear_parser_cache_period', type=int,
                            help=rcdocs["clear_parser_cache_period"])
        parser.add_argument('--dumpast', action='store', dest='dumpast', 
                            metavar="FILE", help=rcdocs["dumpast"])

    def setup(self, rc):
        """Remember to call super() on subclasses!"""
        if isinstance(rc.parsers, basestring):
            if '[' in rc.parsers or '{' in  rc.parsers:
                rc.parsers = eval(rc.parsers)
        # This should go last
        if rc.dumpast is not NotSpecified:
            dumpast(rc.dumpast, rc.parsers, rc.sourcedir, includes=rc.includes, 
                    defines=rc.defines, undefines=rc.undefines,
                    verbose=rc.verbose, debug=rc.debug, builddir=rc.builddir)
            sys.exit()

    def execute(self, rc):
        raise TypeError("ParserPlugin is not a complete plugin.  Do not use directly")

    def report_debug(self, rc):
        """Remember to call super() on subclasses!"""
        msg = 'Autodescriber parsers available:\n\n{0}\n\n'
        msg = msg.format(pformat(PARSERS_AVAILABLE))
        return msg
