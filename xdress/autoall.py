"""This module is used to scrape the all of the APIs from a given source file
and return their name and kind.  These include classes, structs, functions,
and certain variable types.  It is not used to actually describe these elements.
That is the job of the autodescriber.

This module is available as an xdress plugin by the name ``xdress.autoall``.

Including this plugin enables the ``classes``, ``functions``, and ``variables``  
run control parameters to have an asterix ('*') in the name positon (index 0).
For example, rather than writing::

    classes = [
        ('People', 'people'),
        ('JoanOfArc', 'people'),
        ('JEdgaHoover', 'people'),
        ('Leslie', 'people'),
        ('HuaMulan', 'people'),
        ]

we can instead simply write::

    classes = [('*', 'people')]

Isn't this grand?!

:author: Anthony Scopatz <scopatz@gmail.com>

Automatic Finder API
====================
"""
from __future__ import print_function
import os
import io
import re
import sys
from hashlib import md5
from pprint import pprint, pformat
from warnings import warn
try:
    import cPickle as pickle
except ImportError:
    import pickle

try:
    import pycparser
except ImportError:
    pycparser = None

try:
    from . import clang
    from .clang.cindex import CursorKind
except ImportError:
    clang = None

from . import utils
from . import astparsers

from .utils import find_source, FORBIDDEN_NAMES, NotSpecified, RunControl, apiname, \
    ensure_apiname

if os.name == 'nt':
    import ntpath
    import posixpath

if sys.version_info[0] >= 3:
    basestring = str

class GccxmlFinder(object):
    """Class used for discovering APIs using an etree representation of 
    the GCC-XML AST."""

    def __init__(self, root=None, onlyin=None, verbose=False):
        """Parameters
        -------------
        root : element tree node, optional
            The root element node of the AST.  
        onlyin :  str, optional
            Filename to search, prevents finding APIs coming from other libraries.
        verbose : bool, optional
            Flag to display extra information while visiting the file.

        """
        self.verbose = verbose
        self._root = root
        origonlyin = onlyin
        onlyin = [onlyin] if isinstance(onlyin, basestring) else onlyin
        onlyin = set() if onlyin is None else set(onlyin)
        onlyin = [root.find("File[@name='{0}']".format(oi)) for oi in onlyin]
        self.onlyin = set([oi.attrib['id'] for oi in onlyin if oi is not None])
        if 0 == len(self.onlyin):
            msg = ("None of these files are present: {0!r}; "
                   "autodescribing will probably fail.")
            msg = msg.format(origonlyin)
            warn(msg, RuntimeWarning)
        self.variables = []
        self.functions = []
        self.classes = []

    def __str__(self):
        return ("vars = " + pformat(self.variables) + "\n" + 
                "funcs = " + pformat(self.functions) + "\n" +
                "classes = " + pformat(self.classes) + "\n")

    def _pprint(self, node):
        if self.verbose:
            print("Auto-Found: {0} {1} {2}".format(node.tag,
                                        node.attrib.get('id', ''),
                                        node.attrib.get('name', None)))

    def visit(self, node=None):
        """Visits the node and all sub-nodes, filling the API names
        as it goes.

        Parameters
        ----------
        node : element tree node, optional
            The element tree node to start from.  If this is None, then the 
            top-level node is found and visited.

        """
        node = node or self._root
        self.variables += self.visit_kinds(node, "Enumeration")
        self.functions += self.visit_kinds(node, "Function")
        self.classes += self.visit_kinds(node, ["Class", "Struct"])

    def visit_kinds(self, node, kinds):
        """Visits the node and all sub-nodes, finding instances of the kinds 
        and recording the names as it goes.

        Parameters
        ----------
        node : element tree node
            The element tree node to start from.  
        kinds : str or sequence of str
            The API elements to find.

        Returns
        -------
        names : list of str
            Names of the API elements in this file that match the kinds provided.

        """
        if not isinstance(kinds, basestring):
            names = []
            for k in kinds:
                names += self.visit_kinds(node, k)
            names = [n for n in names if n not in FORBIDDEN_NAMES]
            return names
        names = set()
        for child in node.iterfind(".//" + kinds):
            if child.attrib.get('file', None) not in self.onlyin:
                continue
            name = child.attrib.get('name', '_')
            if name.startswith('_'):
                continue
            if name in FORBIDDEN_NAMES:
                continue
            names.add(utils.parse_template(name))
            self._pprint(child)
        return sorted(names)
            

def gccxml_findall(filename, includes=(), defines=('XDRESS',), undefines=(),
                   extra_parser_args=(), verbose=False, debug=False, 
                   builddir='build', language='c++', clang_includes=()):
    """Automatically finds all API elements in a file via GCC-XML.

    Parameters
    ----------
    filename : str
        The path to the file
    includes : list of str, optional
        The list of extra include directories to search for header files.
    defines : list of str, optional
        The list of extra macro definitions to apply.
    undefines : list of str, optional
        The list of extra macro undefinitions to apply.
    extra_parser_args : list of str, optional
        Further command line arguments to pass to the parser.
    verbose : bool, optional
        Flag to diplay extra information while describing the class.
    debug : bool, optional
        Flag to enable/disable debug mode.
    builddir : str, optional
        Location of -- often temporary -- build files.
    language : str
        Valid language flag.
    clang_includes : ignored

    Returns
    -------
    variables : list of strings
        A list of variable names to wrap from the file.
    functions : list of strings
        A list of function names to wrap from the file.
    classes : list of strings
        A list of class names to wrap from the file.

    """
    if os.name == 'nt':
        # GCC-XML and/or Cygwin wants posix paths on Windows.
        filename = posixpath.join(*ntpath.split(filename))
    root = astparsers.gccxml_parse(filename, includes=includes, defines=defines,
            undefines=undefines, extra_parser_args=extra_parser_args, verbose=verbose, 
            debug=debug, builddir=builddir)
    basename = filename.rsplit('.', 1)[0]
    onlyin = set([filename] + 
                 [basename + '.' + h for h in utils._hdr_exts if h.startswith('h')])
    finder = GccxmlFinder(root, onlyin=onlyin, verbose=verbose)
    finder.visit()
    return finder.variables, finder.functions, finder.classes


def clang_findall(filename, includes=(), defines=('XDRESS',), undefines=(),
                  extra_parser_args=(), verbose=False, debug=False, builddir='build', 
                  language='c++', clang_includes=()):
    """Automatically finds all API elements in a file via clang.

    Parameters
    ----------
    filename : str
        The path to the file
    includes : list of str, optional
        The list of extra include directories to search for header files.
    defines : list of str, optional
        The list of extra macro definitions to apply.
    undefines : list of str, optional
        The list of extra macro undefinitions to apply.
    extra_parser_args : list of str, optional
        Further command line arguments to pass to the parser.
    language : str
        Valid language flag.
    verbose : Ignored
    debug : Ignored
    builddir : Ignored
    clang_includes : list of str, optional
        clang-specific include paths.

    Returns
    -------
    variables : list of strings
        A list of variable names to wrap from the file.
    functions : list of strings
        A list of function names to wrap from the file.
    classes : list of strings
        A list of class names to wrap from the file.

    """
    tu = astparsers.clang_parse(filename, includes=includes, defines=defines,
                                undefines=undefines, 
                                extra_parser_args=extra_parser_args, verbose=verbose, 
                                debug=debug, language=language, 
                                clang_includes=clang_includes)
    basename = filename.rsplit('.', 1)[0]
    onlyin = frozenset([filename] +
                       [basename + '.' + h for h in utils._hdr_exts if h.startswith('h')])
    variables, functions, classes = [],[],[]
    def visit(node):
        kind = node.kind
        if kind == CursorKind.NAMESPACE:
            for kid in node.get_children():
                visit(kid)
        elif kind == CursorKind.ENUM_DECL:
            variables.append(node.spelling)
        elif kind == CursorKind.FUNCTION_DECL:
            functions.append(node.spelling)
        elif kind in (CursorKind.CLASS_DECL,CursorKind.STRUCT_DECL):
            classes.append(node.spelling)
    for node in tu.cursor.get_children():
        file = node.extent.start.file
        if file and file.name in onlyin:
            visit(node)
    return variables, functions, classes


class PycparserFinder(astparsers.PycparserNodeVisitor):
    """Class used for discovering APIs using the pycparser AST."""

    def __init__(self, root=None, onlyin=None, verbose=False):
        """Parameters
        -------------
        root : element tree node, optional
            The root element node of the AST.  
        onlyin :  str, optional
            Filename to search, prevents finding APIs coming from other libraries.
        verbose : bool, optional
            Flag to display extra information while visiting the file.

        """
        super(PycparserFinder, self).__init__()
        self.verbose = verbose
        self._root = root
        self.onlyin = onlyin
        self.variables = []
        self.functions = []
        self.classes = []

    def __str__(self):
        return ("vars = " + pformat(self.variables) + "\n" + 
                "funcs = " + pformat(self.functions) + "\n" +
                "classes = " + pformat(self.classes) + "\n")

    def _pprint(self, node):
        if self.verbose:
            node.show()

    def visit(self, node=None):
        """Visits the node and all sub-nodes, filling the API names
        as it goes.

        Parameters
        ----------
        node : element tree node, optional
            The element tree node to start from.  If this is None, then the 
            top-level node is found and visited.

        """
        node = node or self._root
        super(PycparserFinder, self).visit(node)

    def visit_Enum(self, node):
        if node.coord.file not in self.onlyin:
            return
        name = node.name
        if name.startswith('_'):
            return
        if name in FORBIDDEN_NAMES:
            return
        self._pprint(node)
        self.variables.append(name)

    def visit_FuncDecl(self, node):
        if node.coord.file not in self.onlyin:
            return
        if isinstance(node.type, pycparser.c_ast.PtrDecl):
            name = node.type.type.declname
        else:
            name = node.type.declname
        if name is None or name.startswith('_'):
            return
        if name in FORBIDDEN_NAMES:
            return
        self._pprint(node)
        self.functions.append(name)

    def visit_Struct(self, node):
        if node.coord.file not in self.onlyin:
            return
        name = node.name
        if name is None:
            self._status = "<name-not-found>"
            return
        if name.startswith('_'):
            return 
        if name in FORBIDDEN_NAMES:
            return
        self._pprint(node)
        self.classes.append(name)

    def visit_Typedef(self, node):
        if node.coord.file not in self.onlyin:
            return
        self._pprint(node)
        self._status = None
        self.visit(node.type)
        stat = self._status
        self._status = None
        if stat is None:
            return
        if stat == "<name-not-found>":
            name = node.name
        if name is None or name.startswith('_'):
            return
        if name in FORBIDDEN_NAMES:
            return
        self.classes.append(name)


def pycparser_findall(filename, includes=(), defines=('XDRESS',), undefines=(),
                      extra_parser_args=(), verbose=False, debug=False, 
                      builddir='build', language='c', clang_includes=()):
    """Automatically finds all API elements in a file via GCC-XML.

    Parameters
    ----------
    filename : str
        The path to the file
    includes : list of str, optional
        The list of extra include directories to search for header files.
    defines : list of str, optional
        The list of extra macro definitions to apply.
    undefines : list of str, optional
        The list of extra macro undefinitions to apply.
    extra_parser_args : list of str, optional
        Further command line arguments to pass to the parser.
    verbose : bool, optional
        Flag to diplay extra information while describing the class.
    debug : bool, optional
        Flag to enable/disable debug mode.
    builddir : str, optional
        Location of -- often temporary -- build files.
    language : str
        Valid language flag.
    clang_includes : ignored

    Returns
    -------
    variables : list of strings
        A list of variable names to wrap from the file.
    functions : list of strings
        A list of function names to wrap from the file.
    classes : list of strings
        A list of class names to wrap from the file.

    """
    root = astparsers.pycparser_parse(filename, includes=includes, defines=defines,
                undefines=undefines, extra_parser_args=extra_parser_args, 
                verbose=verbose, debug=debug, builddir=builddir)
    basename = filename.rsplit('.', 1)[0]
    onlyin = set([filename, basename + '.h'])
    finder = PycparserFinder(root, onlyin=onlyin, verbose=verbose)
    finder.visit()
    return finder.variables, finder.functions, finder.classes


#
# Top-level function
#

_finders = {
    'clang': clang_findall,
    'gccxml': gccxml_findall,
    'pycparser': pycparser_findall,
    }

def findall(filename, includes=(), defines=('XDRESS',), undefines=(), 
            extra_parser_args=(), parsers='gccxml', verbose=False, debug=False, 
            builddir='build', language='c++', clang_includes=()):
    """Automatically finds all API elements in a file.  This is the main entry point.

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
    extra_parser_args : list of str, optional
        Further command line arguments to pass to the parser.
    parsers : str, list, or dict, optional
        The parser / AST to use to use for the file.  Currently 'clang', 'gccxml', 
        and 'pycparser' are supported, though others may be implemented in the 
        future.  If this is a string, then this parser is used.  If this is a list, 
        this specifies the parser order to use based on availability.  If this is
        a dictionary, it specifies the order to use parser based on language, i.e.
        ``{'c' ['pycparser', 'gccxml'], 'c++': ['gccxml', 'pycparser']}``.
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
    variables : list of strings
        A list of variable names to wrap from the file.
    functions : list of strings
        A list of function names to wrap from the file.
    classes : list of strings
        A list of class names to wrap from the file.

    """
    parser = astparsers.pick_parser(language, parsers)
    finder = _finders[parser]
    rtn = finder(filename, includes=includes, defines=defines, undefines=undefines, 
                 extra_parser_args=extra_parser_args, verbose=verbose, debug=debug, 
                 builddir=builddir, language=language, clang_includes=clang_includes)
    return rtn


#
# Persisted Cache for great speed up
#

class AutoNameCache(object):
    """A quick persistent cache for name lists automatically found in files.  
    The keys are (classname, filename, kind) tuples.  The values are 
    (hashes-of-the-file, finder-results) tuples."""

    def __init__(self, cachefile=os.path.join('build', 'autoname.cache')):
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

    def isvalid(self, filename):
        """Boolean on whether the cach value for a filename matches the state 
        of the file on the system."""
        key = filename
        if key not in self.cache:
            return False
        cachehash = self.cache[key][0]
        with io.open(filename, 'rb') as f:
            filebytes = f.read()
        currhash = md5(filebytes).hexdigest()
        return cachehash == currhash

    def __getitem__(self, key):
        return self.cache[key][1]  # return the results of the finder only

    def __setitem__(self, key, value):
        filename = key
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

#
# Plugin
#

class XDressPlugin(astparsers.ParserPlugin):
    """This plugin resolves the '*' syntax in wrapper types by parsing the 
    source files prio to describing them. 
    """
    allsrc = varhasstar = fnchasstar = clshasstar = None

    def defaultrc(self):
        rc = RunControl()
        rc._update(super(XDressPlugin, self).defaultrc)
        return rc

    def report_debug(self, rc):
        msg = super(XDressPlugin, self).report_debug(rc)
        msg += "Autoall:\n\n"
        msg += "allsrc = {0}\n\n".format(pformat(self.allsrc))
        msg += "varhasstar = {0}\n\n".format(pformat(self.varhasstar))
        msg += "fnchasstar = {0}\n\n".format(pformat(self.fnchasstar))
        msg += "clshasstar = {0}\n\n".format(pformat(self.clshasstar))
        return msg

    def setup(self, rc):
        """Expands variables, functions, and classes in the rc based on 
        copying src filenames to tar filename and the special '*' all syntax."""
        super(XDressPlugin, self).setup(rc)
        self.setup_basic(rc)
        self.setup_heavy(rc)

    def execute(self, rc):
        # dummy
        pass

    # Helper methods

    def setup_basic(self, rc):
        """Does the easy part of setting up an autodecsibe environment"""
        # first pass -- gather and expand target
        allsrc = {}
        varhasstar = False
        for i, var in enumerate(rc.variables):
            rc.variables[i] = var = ensure_apiname(var)
            if var.srcname == '*':
                allsrc.update(zip(var.srcfiles, [var.language]*len(var.srcfiles)))
                varhasstar = True
        fnchasstar = False
        for i, fnc in enumerate(rc.functions):
            rc.functions[i] = fnc = ensure_apiname(fnc)
            if fnc.srcname == '*':
                allsrc.update(zip(fnc.srcfiles, [fnc.language]*len(fnc.srcfiles)))
                #allsrc.update(fnc.srcfiles)
                fnchasstar = True
        clshasstar = False
        for i, cls in enumerate(rc.classes):
            rc.classes[i] = cls = ensure_apiname(cls)
            if cls.srcname == '*':
                allsrc.update(zip(cls.srcfiles, [cls.language]*len(cls.srcfiles)))
                #allsrc.update(cls.srcfiles)
                clshasstar = True
        self.allsrc = allsrc
        self.varhasstar = varhasstar
        self.fnchasstar = fnchasstar
        self.clshasstar = clshasstar

    def setup_heavy(self, rc):
        """Does the hard work of actually searching the source files."""
        print("autoall: discovering API names")
        if not self.varhasstar and not self.fnchasstar and not self.clshasstar:
            print("autoall: no API names to discover!")
            return
        allsrc = self.allsrc
        kinds = ['variables', 'functions', 'classes']

        # second pass -- find all
        allfiles = {}
        cachefile = os.path.join(rc.builddir, 'autoname.cache')
        autonamecache = AutoNameCache(cachefile=cachefile)
        for i, (srcfile, lang) in enumerate(allsrc.items()):
            print("autoall: searching {0}".format(srcfile))
            if autonamecache.isvalid(srcfile):
                found = autonamecache[srcfile]
            else:
                found = findall(srcfile, includes=rc.includes, defines=rc.defines, 
                                undefines=rc.undefines, 
                                extra_parser_args=rc.extra_parser_args, 
                                parsers=rc.parsers, verbose=rc.verbose, 
                                debug=rc.debug, builddir=rc.builddir, language=lang,
                                clang_includes=rc.clang_includes)
                autonamecache[srcfile] = found
                autonamecache.dump()
            allfiles[srcfile] = found
            for k, kind in enumerate(kinds):
                if 0 < len(found[k]):
                    fstr = ", ".join([str(_) for _ in found[k]])
                    print("autoall: found {0}: {1}".format(kind, fstr))
            if 0 == i%rc.clear_parser_cache_period:
                astparsers.clearmemo()

        # third pass -- replace *s
        if self.varhasstar:
            newvars = []
            for var in rc.variables:
                if var.srcname == '*':
                    for srcfile in var.srcfiles:
                        for x in allfiles[srcfile][0]:
                            newvars.append(var._replace(srcname=x, tarname=x))
                else:
                    newvars.append(var)
            rc.variables = newvars
        if self.fnchasstar:
            newfncs = []
            for fnc in rc.functions:
                if fnc.srcname == '*':
                    for srcfile in fnc.srcfiles:
                        for x in allfiles[srcfile][1]:
                            newfncs.append(fnc._replace(srcname=x, tarname=x))
                else:
                    newfncs.append(fnc)
            rc.functions = newfncs
        if self.clshasstar:
            newclss = []
            for cls in rc.classes:
                if cls.srcname == '*':
                    for srcfile in cls.srcfiles:
                        for x in allfiles[srcfile][2]:
                            newclss.append(cls._replace(srcname=x, tarname=x))
                else:
                    newclss.append(cls)
            rc.classes = newclss

