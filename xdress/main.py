"""Top-level automatic API generators entry point.  

:author: Anthony Scopatz <scopatz@gmail.com>

API Generation
==============
This module is where all of the API generation routines reside.
Until now the type system, automatic description, and cython code generation have
all be independent of what classes they are wrapping.  
The main module is normally run via the command line interface as follows:

.. code-block:: bash

    path/to/proj/ $ xdress

This has the following usage::

    path/to/proj/ $ xdress -h 
    usage: Generates XDress API [-h] [--rc RC] [-v] [--debug] [--make-extratypes]
                                [--no-make-extratypes] [--make-stlcontainers]
                                [--no-make-stlcontainers] [--make-cythongen]
                                [--no-make-cythongen] [--make-cyclus]
                                [--no-make-cyclus] [--dumpdesc]
                                [-I INCLUDES [INCLUDES ...]]
                                [-D DEFINES [DEFINES ...]]
                                [-U UNDEFINES [UNDEFINES ...]]
                                [--builddir BUILDDIR]

    optional arguments:
      -h, --help            show this help message and exit
      --rc RC               path to run control file
      -v, --verbose         print more output
      --debug               build in debugging mode
      --make-extratypes     make extra types wrapper
      --no-make-extratypes  don't make extra types wrapper
      --make-stlcontainers  make STL container wrappers
      --no-make-stlcontainers
                            don't make STL container wrappers
      --make-cythongen      make cython bindings
      --no-make-cythongen   don't make cython bindings
      --make-cyclus         make cyclus bindings
      --no-make-cyclus      don't make cyclus bindings
      --dumpdesc            print description cache
      -I INCLUDES [INCLUDES ...], --includes INCLUDES [INCLUDES ...]
                            additional include dirs
      -D DEFINES [DEFINES ...], --defines DEFINES [DEFINES ...]
                            set additional macro definitions
      -U UNDEFINES [UNDEFINES ...], --undefines UNDEFINES [UNDEFINES ...]
                            unset additional macro definitions
      --builddir BUILDDIR   path to build directory


.. warning:: 

    Known Limitation: Currently only files header files ending in .h and 
    implementation files ending in .cpp are seen by xdress.  These could easily
    be abstracted to more extensions.  Pull requests always welcome :).


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
    this module variable.
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
are pertinent to just that source code.  (The modifications will affect the type 
system globally, but may not be relevant anywhere except for that source file.)

Sidecar Example
---------------
The following example displays the contents of a sample sidecar file which adds 
a refinement type (implemented elsewhere) to the type system so that it may be
used to declare additional attr and method APIs.

.. code-block:: python

    # Start by adding refinement type hooks for 'sepeff_t', which is a type of map
    import xdress.typesystem as ts
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

from .utils import newoverwrite, newcopyover, ensuredirs, writenewonly, exec_file, \
    NotSpecified, RunControl, guess_language, find_source
from . import typesystem as ts
from . import stlwrap
from .cythongen import gencpppxd, genpxd, genpyx
from . import autodescribe 
from . import autoall

from .plugins import Plugins

if sys.version_info[0] >= 3:
    basestring = str


pysrcenv = {}

def load_pysrcmod(srcname, rc):
    """Loads a module dictionary from a src file into the pysrcenv cache."""
    if srcname in pysrcenv:
        return 
    pyfilename = os.path.join(rc.sourcedir, srcname + '.py')
    if os.path.isfile(pyfilename):
        glbs = globals()
        locs = {}
        exec_file(pyfilename, glbs, locs)
        if 'mod' not in locs:
            pymod = {}
        elif callable(locs['mod']):
            pymod = eval('mod()', glbs, locs)
        else:
            pymod = locs['mod']
    else:
        pymod = {}
    pysrcenv[srcname] = pymod


def compute_desc(name, srcname, tarname, kind, rc):
    """Returns a description dictionary for a class or function
    implemented in a source file and bound into a target file.

    Parameters
    ----------
    name : str
        Class or function name to describe.
    srcname : str
        File basename of implementation.  
    tarname : str
        File basename where the bindings will be generated.  
    kind : str
        The kind of type to describe, valid flags are 'class', 'func', and 'var'.
    verbose : bool, optional
        Flag for printing extra information during description process.

    Returns
    -------
    desc : dict
        Description dictionary.

    """
    # description
    srcfname, hdrfname, lang, ext = find_source(srcname, sourcedir=rc.sourcedir)
    filename = os.path.join(rc.sourcedir, srcfname)
    cache = rc._cache
    if cache.isvalid(name, filename, kind):
        srcdesc = cache[name, filename, kind]
    else:
        srcdesc = autodescribe.describe(filename, name=name, kind=kind,
                                        includes=rc.includes, defines=rc.defines,
                                        undefines=rc.undefines, parsers=rc.parsers,
                                        verbose=rc.verbose, 
                                        debug=rc.debug, builddir=rc.builddir)
        cache[name, filename, kind] = srcdesc

    # python description
    pydesc = pysrcenv[srcname].get(name, {})

    desc = autodescribe.merge_descriptions([srcdesc, pydesc])
    desc['source_filename'] = srcfname
    desc['header_filename'] = hdrfname
    desc['metadata_filename'] = '{0}.py'.format(srcname)
    if tarname is None:
        desc['pxd_filename'] = desc['pyx_filename'] = desc['srcpxd_filename'] = None
    else:
        desc['pxd_filename'] = '{0}.pxd'.format(tarname)
        desc['pyx_filename'] = '{0}.pyx'.format(tarname)
        desc['srcpxd_filename'] = '{0}_{1}.pxd'.format(ext, tarname)
    return desc

def _adddesc2env(desc, env, name, srcname, tarname):
    """Adds a description to environment"""
    # Add to target environment
    # docstrings overwrite, extras accrete 
    mod = {name: desc, 'docstring': pysrcenv[srcname].get('docstring', ''),
           'srcpxd_filename': desc['srcpxd_filename'],
           'pxd_filename': desc['pxd_filename'], 
           'pyx_filename': desc['pyx_filename']}
    if tarname not in env:
        env[tarname] = mod
        env[tarname]["name"] = tarname
        env[tarname]['extra'] = pysrcenv[srcname].get('extra', '')
    else:
        env[tarname].update(mod)
        env[tarname]['extra'] += pysrcenv[srcname].get('extra', '')

def expand_apis(rc):
    """Expands variables, functions, and classes in the rc based on 
    copying src filenames to tar filename and the special '*' all syntax."""
    # first pass -- gather and expand tar
    allsrc = set()
    varhasstar = False
    for i, var in enumerate(rc.variables):
        if var[0] == '*':
            allsrc.add(var[1])
            varhasstar = True
        if len(var) == 2:
            rc.variables[i] = (var[0], var[1], var[1])
    fnchasstar = False
    for i, fnc in enumerate(rc.functions):
        if fnc[0] == '*':
            allsrc.add(fnc[1])
            fnchasstar = True
        if len(fnc) == 2:
            rc.functions[i] = (fnc[0], fnc[1], fnc[1])
    clshasstar = False
    for i, cls in enumerate(rc.classes):
        if cls[0] == '*':
            allsrc.add(cls[1])
            clshasstar = True
        if len(cls) == 2:
            rc.classes[i] = (cls[0], cls[1], cls[1])
    if not varhasstar and not fnchasstar and not clshasstar:
        return 
    # second pass -- find all
    allnames = {}
    for srcname in allsrc:
        srcfname, hdrfname, lang, ext = find_source(srcname, sourcedir=rc.sourcedir)
        filename = os.path.join(rc.sourcedir, srcfname)
        found = autoall.findall(filename, includes=rc.includes, defines=rc.defines,
                    undefines=rc.undefines, parsers=rc.parsers, verbose=rc.verbose, 
                    debug=rc.debug, builddir=rc.builddir)
        allnames[srcname] = found
    # third pass -- replace *s
    if varhasstar:
        newvars = []
        for var in rc.variables:
            if var[0] == '*':
                newvars += [(x, var[1], var[2]) for x in allnames[var[1]][0]]
            else:
                newvars.append(var)
        rc.variables = newvars
    if fnchasstar:
        newfncs = []
        for fnc in rc.functions:
            if fnc[0] == '*':
                newfncs += [(x, fnc[1], fnc[2]) for x in allnames[fnc[1]][1]]
            else:
                newfncs.append(fnc)
        rc.functions = newfncs
    if clshasstar:
        newclss = []
        for cls in rc.classes:
            if cls[0] == '*':
                newclss += [(x, cls[1], cls[2]) for x in allnames[cls[1]][2]]
            else:
                newclss.append(cls)
        rc.classes = newclss


def genbindings(rc):
    """Generates bidnings using the command line setting specified in rc.
    """
    print("cythongen: scraping C/C++ APIs from source")
    cache = rc._cache
    rc.make_cyclus = False  # FIXME cyclus bindings don't exist yet!
    expand_apis(rc)
    srcnames = set([x[1] for x in rc.variables])
    srcnames |= set([x[1] for x in rc.functions])
    srcnames |= set([x[1] for x in rc.classes])
    for x in srcnames:
        load_pysrcmod(x, rc)
    # register dtypes
    for t in rc.stlcontainers:
        if t[0] == 'vector':
            ts.register_numpy_dtype(t[1])

    # compute all class descriptions first 
    classes = {}
    env = {}  # target environment, not source one
    for classname, srcname, tarname in rc.classes:
        print("parsing " + classname)
        desc = classes[classname] = compute_desc(classname, srcname, tarname, 
                                                 'class', rc)
        if rc.verbose:
            pprint(desc)

        print("registering " + classname)
        pxd_base = desc['pxd_filename'].rsplit('.', 1)[0]         # eg, fccomp
        #pxd_base = tarname or srcname  # eg, fccomp
        cpppxd_base = desc['srcpxd_filename'].rsplit('.', 1)[0]   # eg, cpp_fccomp
        #cpppxd_base = 'cpp_' + (tarname or srcname)   # eg, cpp_fccomp
        class_c2py = ('{pytype}({var})', 
                      ('{proxy_name} = {pytype}()\n'
                       '(<{ctype} *> {proxy_name}._inst)[0] = {var}'),
                      ('if {cache_name} is None:\n'
                       '    {proxy_name} = {pytype}()\n'
                       '    {proxy_name}._free_inst = False\n'
                       '    {proxy_name}._inst = &{var}\n'
                       '    {cache_name} = {proxy_name}\n')
                     )
        class_py2c = ('{proxy_name} = <{cytype_nopred}> {var}', 
                      '(<{ctype_nopred} *> {proxy_name}._inst)[0]')
        class_cimport = (rc.package, cpppxd_base) 
        kwclass = dict(
            name=classname,                                       # FCComp
            cython_c_type=cpppxd_base + '.' + classname,          # cpp_fccomp.FCComp
            cython_cimport=class_cimport,  
            cython_cy_type=pxd_base + '.' + classname,            # fccomp.FCComp   
            cython_py_type=pxd_base + '.' + classname,            # fccomp.FCComp   
            cython_template_class_name=classname.replace('_', '').capitalize(),
            cython_cyimport=pxd_base,                             # fccomp
            cython_pyimport=pxd_base,                             # fccomp
            cython_c2py=class_c2py,
            cython_py2c=class_py2c,
            )
        ts.register_class(**kwclass)
        class_ptr_c2py = ('{pytype}({var})', 
                          ('{proxy_name} = {pytype}()\n'
                           '(<{ctype} *> {proxy_name}._inst) = {var}'),
                          ('if {cache_name} is None:\n'
                           '    {proxy_name} = {pytype}()\n'
                           '    {proxy_name}._free_inst = False\n'
                           '    {proxy_name}._inst = {var}\n'
                           '    {cache_name} = {proxy_name}\n')
                         )
        class_ptr_py2c = ('{proxy_name} = <{cytype_nopred}> {var}', 
                          '(<{ctype_nopred} *> {proxy_name}._inst)')
        kwclassptr = dict(
            name=(classname, '*'), 
            cython_c2py=class_ptr_c2py,
            cython_py2c=class_ptr_py2c,
            cython_cimport=kwclass['cython_cimport'],
            cython_cyimport=kwclass['cython_cyimport'],
            cython_pyimport=kwclass['cython_pyimport'],
            )
        ts.register_class(**kwclassptr)
        cache.dump()
        _adddesc2env(desc, env, classname, srcname, tarname)

    # then compute all function descriptions
    for funcname, srcname, tarname in rc.functions:
        print("parsing " + funcname)
        desc = compute_desc(funcname, srcname, tarname, 'func', rc)
        if rc.verbose:
            pprint(desc)
        cache.dump()
        _adddesc2env(desc, env, funcname, srcname, tarname)

    # then compute all variable descriptions
    for varname, srcname, tarname in rc.variables:
        print("parsing " + varname)
        desc = compute_desc(varname, srcname, tarname, 'var', rc)
        if rc.verbose:
            pprint(desc)
        cache.dump()
        _adddesc2env(desc, env, varname, srcname, tarname)

    # next, make cython bindings
    # generate first, then write out to ensure this is atomic per-class
    if rc.make_cythongen:
        print("cythongen: creating C/C++ API wrappers")
        cpppxds = gencpppxd(env)
        pxds = genpxd(env, classes)
        pyxs = genpyx(env, classes)
        for key, cpppxd in cpppxds.items():
            newoverwrite(cpppxd, os.path.join(rc.package, env[key]['srcpxd_filename']), rc.verbose)
        for key, pxd in pxds.items():
            newoverwrite(pxd, os.path.join(rc.package, env[key]['pxd_filename']), rc.verbose)
        for key, pyx in pyxs.items():
            newoverwrite(pyx, os.path.join(rc.package, env[key]['pyx_filename']), rc.verbose)

    # next, make cyclus bindings
    if rc.make_cyclus:
        print("making cyclus bindings")


def setuprc(rc):
    """Makes and validates a run control object and the environment it specifies."""
    if isinstance(rc.parsers, basestring):
        if '[' in rc.parsers or '{' in  rc.parsers:
            rc.parsers = eval(rc.parsers)

#defaultrc = RunControl(
#    rc="xdressrc.py", 
#    debug=False,
#    make_extratypes=True,
##    make_stlcontainers=True,
##    make_cythongen=True,
##    make_cyclus=False,
#    dumpdesc=False,
    includes=[],
    defines=["XDRESS"],
    undefines=[],
#    verbose=False,
#    package=NotSpecified,
#    packagedir=NotSpecified,
#    sourcedir='src',
#    builddir='build',
#    extra_types='xdress_extra_types',
#    stlcontainers=[],
#    stlcontainers_module='stlcontainers',
    variables=(),
    functions=(),
    classes=(),
    parsers={'c': ['pycparser', 'gccxml', 'clang'], 
             'c++':['gccxml', 'clang', 'pycparser'] },
    )

def _old_main_setup():
    """Setup xdress API generation."""
    parser.add_argument('-I', '--includes', action='store', dest='includes', nargs="+",
                        default=NotSpecified, help="additional include dirs")
    parser.add_argument('-D', '--defines', action='append', dest='defines', nargs="+",
                        default=NotSpecified, help="set additional macro definitions")
    parser.add_argument('-U', '--undefines', action='append', dest='undefines', 
                        nargs="+", default=NotSpecified, type=str,
                        help="unset additional macro definitions")
    parser.add_argument('-p', action='store', dest='parsers', 
                        default=NotSpecified, help="parser(s) name, list, or dict")
    ns = parser.parse_args()

    rc = RunControl()
    rc._update(defaultrc)
    rc.rc = ns.rc
    if os.path.isfile(rc.rc):
        d = {}
        exec_file(rc.rc, d, d)
        rc._update(d)
    rc._update([(k, v) for k, v in ns.__dict__.items() if not k.startswith('_')])
    rc._cache = DescriptionCache(cachefile=os.path.join(rc.builddir, 'desc.cache'))

    setuprc(rc)
    return rc


def main_body(rc):
    """Body for xdress API generation."""
    # set typesystem defaults

    if rc.make_cythongen:
        genbindings(rc)

def _old_main():
    """Entry point for xdress API generation."""
    rc = main_setup()
    try:
        main_body(rc)
    except Exception as e:
        if rc.debug:
            import traceback
            sep = r'~\_/' * 17 + '~=[,,_,,]:3\n\n'
            with io.open(os.path.join(rc.builddir, 'debug.txt'), 'a+b') as f:
                msg = '\n{0}Autodescriber parsers available:\n\n{1}\n\n'
                f.write(msg.format(sep, pformat(autodescribe.PARSERS_AVAILABLE)))
            raise 
        else:
            sys.exit(str(e))

def main():
    """Entry point for xdress API generation."""
    # Preprocess plugin names, which entails preprocessing the rc filename
    preparser = argparse.ArgumentParser("XDress Pre-processor", add_help=False)
    preparser.add_argument('--rc', default=NotSpecified, 
                           help="path to run control file")
    preparser.add_argument('--plugins', default=NotSpecified, nargs="+",
                           help="plugins to include")
    prens = preparser.parse_known_args()
    predefaultrc = RunControl(rc="xdressrc.py", plugins=["xdress.base"])
    prerc = RunControl()
    prerc._update(predefaultrc)
    prerc.rc = prens.rc
    if os.path.isfile(prerc.rc):
        rcdict = {}
        exec_file(prerc.rc, rcdict, rcdict)
        prerc.rc = rcdict['rc'] if 'rc' in rcdict else NotSpecified
        prerc.plugins = rcdict['plugins'] if 'plugins' in rcdict else NotSpecified
    prerc._update([(k, v) for k, v in prens.__dict__.items()])    

    # run plugins
    plugins = Plugins(prerc.plugins)
    parser = plugins.build_cli()
    ns = parser.parse_args()
    rc = plugins.merge_defaultrcs()
    rc._update(rcdict)
    rc._update([(k, v) for k, v in ns.__dict__.items()])
    plugins.setup()
    plugins.execute()
    plugins.teardown()

if __name__ == '__main__':
    main()
