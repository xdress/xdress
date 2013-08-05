"""Top-level xdress entry point.  

:author: Anthony Scopatz <scopatz@gmail.com>

XDress Comand Line Interface
============================
This module is where all of the API generation routines reside.
Until now the type system, automatic description, and cython code generation have
all be independent of what classes they are wrapping.  
The main module is normally run via the command line interface as follows:

.. code-block:: bash

    path/to/proj/ $ xdress

This has the following usage::

    path/to/proj/ $ xdress -h 
    usage: Generates XDress API [-h] [--rc RC] [--plugins PLUGINS [PLUGINS ...]]
                                [--debug] [-v] [--dumpdesc] [--package PACKAGE]
                                [--packagedir PACKAGEDIR] [--sourcedir SOURCEDIR]
                                [--builddir BUILDDIR] [--bash-completion]
                                [--no-bash-completion]
                                [-I INCLUDES [INCLUDES ...]]
                                [-D DEFINES [DEFINES ...]]
                                [-U UNDEFINES [UNDEFINES ...]] [-p PARSERS]
                                [--clear-parser-cache-period CLEAR_PARSER_CACHE_PERIOD]
                                [--dumpast FILE] [--max-callbacks MAX_CALLBACKS]
                                [--extra-types EXTRA_TYPES] [--make-extra-types]
                                [--no-make-extra-types]
                                [--stlcontainers-module STLCONTAINERS_MODULE]
                                [--make-stlcontainers] [--no-make-stlcontainers]

    optional arguments:
      -h, --help            show this help message and exit
      --rc RC               Path to run control file
      --plugins PLUGINS [PLUGINS ...]
                            Plugins to include
      --debug               Build in debugging mode
      -v, --verbose         Print more output.
      --dumpdesc            Print the description cache
      --package PACKAGE     The Python package name for the generated wrappers
      --packagedir PACKAGEDIR
                            Path to package directory, same as 'package' if not
                            specified
      --sourcedir SOURCEDIR
                            Path to source directory
      --builddir BUILDDIR   Path to build directory
      --bash-completion     enable bash completion
      --no-bash-completion  disable bash completion
      -I INCLUDES [INCLUDES ...], --includes INCLUDES [INCLUDES ...]
                            Additional include directories
      -D DEFINES [DEFINES ...], --defines DEFINES [DEFINES ...]
                            Set additional macro definitions
      -U UNDEFINES [UNDEFINES ...], --undefines UNDEFINES [UNDEFINES ...]
                            Unset additional macro definitions
      -p PARSERS            Parser(s) name, list, or dict
      --clear-parser-cache-period CLEAR_PARSER_CACHE_PERIOD
                            Number of parser calls to perform before clearing the
                            internal cache. This prevents nasty memory overflow
                            issues.
      --dumpast FILE        Prints the abstract syntax tree of a file.
      --max-callbacks MAX_CALLBACKS
                            The maximum number of callbacks for function pointers
      --extra-types EXTRA_TYPES
                            Module and header file name for xdress extra types.
      --make-extra-types    make extra types wrapper
      --no-make-extra-types
                            don't make extra types wrapper
      --stlcontainers-module STLCONTAINERS_MODULE
                            Module name for C++ standard library container
                            wrappers.
      --make-stlcontainers  make C++ STL container wrappers
      --no-make-stlcontainers
                            don't make C++ STL container wrappers


Sidecar Description Files
=========================
One main advantage of xdress is that every source file may have its own sidecar 
Python file. This file includes additional information (docstrings, additional APIs, 
etc.) about the classes and functions in the source file that it is next to.  For
example, given header and source files named src/my_code.h and src/my_code.cpp then
the sidecar is called src/my_code.py.  

These sidecars may contain a variable named ``mod`` which describes module elements
(classes and functions) coming from this source file.  The mod variable may either
be a dictionary or a callable which returns a dictionary.  This dictionary has the 
following top-level keys:

:<names>: str, names of function and classes that are present in this source file.
    The value for each name key is a partial or full description dictionary for 
    this module variable.  See ``autodescribe.py`` for more details on description
    dictionaries.
:docstring: str, optional, this is a documentation string for the module.  
:extra: dict, optional, this stores arbitrary metadata that may be used with 
    different backends. It is not added by any auto-describe routine but may be
    inserted later if needed.  One example use case is that the Cython generation
    looks for the pyx, pxd, and cpppxd keys for strings of supplemental Cython 
    code to insert directly into the wrapper.  This is not generally used at the
    module level.

Sidecar files are guaranteed to be executed only once.  (Note that they are 
execfile'd rather than imported.)  Furthermore, the description dictionaries
that live under the name keys are merged with the automatically generated 
descriptions.  The sidecar descriptions take precedence over the automatically
generated ones.

Awesome abuses/hacks/side effects are possible since these sidecar files are 
pure Python code. One major use case is to modify the type system from within a 
sidecar.  This is useful for adding refinement types or type specializations that
are pertinent to just that source code.  This is done by creating a TypeSystem
instance in the sidecar, naming it ``ts``, and adding any user-modifications.
The sidecar's ``ts`` TypeSystem will automatically update the default run control
TypeSystem (``rc.ts``).  For this reason, its often useful for the sidecar file's
type system to be empty.

Sidecar Example
---------------
The following example displays the contents of a sample sidecar file which adds 
a refinement type (implemented elsewhere) to the type system so that it may be
used to declare additional attr and method APIs.

.. code-block:: python

    # Start by adding refinement type hooks for 'sepeff_t', which is a type of map
    from xdress.typesystem import TypeSystem
    ts = TypeSystem.empty()
    ts.register_refinement('sepeff_t', ('map', 'int32', 'float64'),
        cython_cyimport='bright.typeconverters', 
        cython_pyimport='bright.typeconverters',
        cython_py2c='bright.typeconverters.sepeff_py2c({var})',)

    # Define a partial class description, will be merged into a full description
    # by main() with the description coming from autodescriber.
    desc = {
        'docstrings': {
            'class': "I am reprocess class",
            'attrs': {
                'sepeff': "I am an attr",
                },
            'methods': {
                'initialize': "init with me",
                'calc_params': "make some params",
                'calc': "run me",
                },
            },
        'attrs': {
            'sepeff': 'sepeff_t',  # redefine attr as refinement type
            },
        'methods': {
            # Add a constructor which takes as its first argument the new refinement
            ('Reprocess', ('sepeff', 'sepeff_t'), ('name', 'str', '""')): None,
            },
        }

    # Module description
    mod = {'Reprocess': desc,
           'docstring': "Python wrapper for Reprocess.",}

Main API
========
"""
from __future__ import print_function
import os
import io
import sys
import argparse
from pprint import pprint, pformat

try:
    import argcomplete
except ImportError:
    argcomplete = None

from .utils import NotSpecified, RunControl, DEFAULT_RC_FILE, DEFAULT_PLUGINS, \
    exec_file

from .plugins import Plugins

if sys.version_info[0] >= 3:
    basestring = str


def main():
    """Entry point for xdress API generation."""
    # Preprocess plugin names, which entails preprocessing the rc file
    preparser = argparse.ArgumentParser("XDress Pre-processor", add_help=False)
    preparser.add_argument('--rc', default=NotSpecified, 
                           help="path to run control file")
    preparser.add_argument('--plugins', default=NotSpecified, nargs="+",
                           help="plugins to include")
    preparser.add_argument('--bash-completion', default=True, action='store_true',
                           help="enable bash completion", dest="bash_completion")
    preparser.add_argument('--no-bash-completion', action='store_false',
                           help="disable bash completion", dest="bash_completion")
    prens = preparser.parse_known_args()[0]
    predefaultrc = RunControl(rc=DEFAULT_RC_FILE, plugins=DEFAULT_PLUGINS)
    prerc = RunControl()
    prerc._update(predefaultrc)
    prerc.rc = prens.rc
    rcdict = {}
    if os.path.isfile(prerc.rc):
        exec_file(prerc.rc, rcdict, rcdict)
        prerc.rc = rcdict['rc'] if 'rc' in rcdict else NotSpecified
        prerc.plugins = rcdict['plugins'] if 'plugins' in rcdict else NotSpecified
    prerc._update([(k, v) for k, v in prens.__dict__.items()])    

    # run plugins
    plugins = Plugins(prerc.plugins)
    parser = plugins.build_cli()
    if argcomplete is not None and prerc.bash_completion:
        argcomplete.autocomplete(parser)
    ns = parser.parse_args()
    rc = plugins.merge_rcs()
    rc._update(rcdict)
    rc._update([(k, v) for k, v in ns.__dict__.items()])
    plugins.setup()
    plugins.execute()
    plugins.teardown()

if __name__ == '__main__':
    main()
