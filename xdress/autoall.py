"""This module is used to scrape the all of the APIs from a given source file
and return thier name and kind.  These include classes, structs, functions, 
and certain variable types.  It is not used to actually describe these elements.
That is the job of the autodescriber.

:author: Anthony Scopatz <scopatz@gmail.com>

"""
from __future__ import print_function

from . import utils
from . import autodescribe


@autodescribe.not_implemented
def gccxml_findall(filename, includes=(), defines=('XDRESS',), undefines=(),
            parsers='gccxml', verbose=False, debug=False,  builddir='build'):
    """Automatically finds all API elements in a file via GCC-XML.

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
    root = autodescribe.gccxml_parse(filename, includes=includes, defines=defines,
            undefines=undefines, verbose=verbose, debug=debug, builddir=builddir)
    basename = filename.rsplit('.', 1)[0]
    onlyin = set([filename] + 
                 [basename + '.' + h for h in utils._hdr_exts if h.startswith('h')])
    finder = GccxmlFinder(name, root, onlyin=onlyin, verbose=verbose)
    finder.visit()
    return finder.variables, finder.functions, finder.classes

@autodescribe.not_implemented
def clang_findall(*args, **kwargs):
    pass

@autodescribe.not_implemented
def pycparser_findall(*args, **kwargs):
    pass


#
# Top-level function
#

_finders = {
    'clang': clang_findall,
    'gccxml': gccxml_findall,
    'pycparser': pycparser_findall,
    }


def findall(filename, includes=(), defines=('XDRESS',), undefines=(), 
            parsers='gccxml', verbose=False, debug=False,  builddir='build'):
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

    Returns
    -------
    variables : list of strings
        A list of variable names to wrap from the file.
    functions : list of strings
        A list of function names to wrap from the file.
    classes : list of strings
        A list of class names to wrap from the file.

    """
    parser = autodescribe.pick_parser(filename, parsers)
    finder = _finders[parser]
    rtn = finder(filename, includes=includes, defines=defines, undefines=undefines, 
                 verbose=verbose, debug=debug, builddir=builddir)
    return rtn
