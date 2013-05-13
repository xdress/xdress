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
                                [-I INCLUDES [INCLUDES ...]] [--builddir BUILDDIR]

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
from hashlib import md5
try:
    import cPickle as pickle
except ImportError:
    import pickle

from .utils import newoverwrite, newcopyover, ensuredirs, writenewonly, exec_file, \
    NotSpecified, RunControl, guess_language, find_source
from . import typesystem as ts
from . import stlwrap
from .cythongen import gencpppxd, genpxd, genpyx
from . import autodescribe 

if sys.version_info[0] >= 3:
    basestring = str

class DescriptionCache(object):
    """A quick persistent cache for descriptions from files.  
    The keys are (classname, filename) tuples.  The values are 
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

    def isvalid(self, name, filename, kind):
        """Boolean on whether the cach value for a (name, filename, kind)
        tuple matches the state of the file on the system."""
        key = (name, filename, kind)
        if key not in self.cache:
            return False
        cachehash = self.cache[key][0]
        with io.open(filename, 'r') as f:
            filestr = f.read().encode()
        currhash = md5(filestr).hexdigest()
        return cachehash == currhash

    def __getitem__(self, key):
        return self.cache[key][1]  # return the description only

    def __setitem__(self, key, value):
        name, filename, kind = key
        with io.open(filename, 'r') as f:
            filestr = f.read().encode()
        currhash = md5(filestr).hexdigest()
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
        The kind of type to describe, currently valid flags are 'class' and 'func'.
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
        cppdesc = cache[name, filename, kind]
    else:
        cppdesc = autodescribe.describe(filename, name=name, kind=kind,
                                        includes=rc.includes, parser=rc.parsers,
                                        verbose=rc.verbose, debug=rc.debug, 
                                        builddir=rc.builddir)
        cache[name, filename, kind] = cppdesc

    # python description
    pydesc = pysrcenv[srcname].get(name, {})

    #if tarname is None:
    #    tarname = "<dont-build>"

    desc = autodescribe.merge_descriptions([cppdesc, pydesc])
    desc['cpp_filename'] = '{0}.cpp'.format(srcname)
    desc['header_filename'] = '{0}.h'.format(srcname)
    desc['metadata_filename'] = '{0}.py'.format(srcname)
    if tarname is None:
        desc['pxd_filename'] = desc['pyx_filename'] = desc['cpppxd_filename'] = None
    else:
        desc['pxd_filename'] = '{0}.pxd'.format(tarname)
        desc['pyx_filename'] = '{0}.pyx'.format(tarname)
        desc['cpppxd_filename'] = 'cpp_{0}.pxd'.format(tarname)
    return desc

def genextratypes(rc):
    d = os.path.split(__file__)[0]
    srcs = [os.path.join(d, 'xdress_extra_types.h'), 
            os.path.join(d, 'xdress_extra_types.pxd'), 
            os.path.join(d, 'xdress_extra_types.pyx')]
    tars = [os.path.join(rc.sourcedir, rc.extra_types + '.h'), 
            os.path.join(rc.packagedir, rc.extra_types + '.pxd'), 
            os.path.join(rc.packagedir, rc.extra_types + '.pyx')]
    for src, tar in zip(srcs, tars):
        with io.open(src, 'r') as f:
            s = f.read()
            s = s.format(extra_types=rc.extra_types)
            newoverwrite(s, tar, rc.verbose)

def genstlcontainers(rc):
    print("stlwrap: generating C++ standard library wrappers & converters")
    fname = os.path.join(rc.packagedir, rc.stlcontainers_module)
    ensuredirs(fname)
    testname = os.path.join(rc.packagedir, 'tests', 'test_' + rc.stlcontainers_module)
    ensuredirs(testname)
    stlwrap.genfiles(rc.stlcontainers, fname=fname, testname=testname, 
                     package=rc.package, verbose=rc.verbose)


def _adddesc2env(desc, env, name, srcname, tarname):
    """Adds a description to environment"""
    # Add to target environment
    # docstrings overwrite, extras accrete 
    mod = {name: desc, 'docstring': pysrcenv[srcname].get('docstring', ''),
           'cpppxd_filename': desc['cpppxd_filename'],
           'pxd_filename': desc['pxd_filename'], 
           'pyx_filename': desc['pyx_filename']}
    if tarname not in env:
        env[tarname] = mod
        env[tarname]['extra'] = pysrcenv[srcname].get('extra', '')
    else:
        env[tarname].update(mod)
        env[tarname]['extra'] += pysrcenv[srcname].get('extra', '')

def genbindings(rc):
    """Generates bidnings using the command line setting specified in rc.
    """
    print("cythongen: scraping C/C++ APIs from source")
    cache = rc._cache
    rc.make_cyclus = False  # FIXME cyclus bindings don't exist yet!
    for i, cls in enumerate(rc.classes):
        if len(cls) == 2:
            rc.classes[i] = (cls[0], cls[1], cls[1])
        load_pysrcmod(cls[1], rc)        
    for i, fnc in enumerate(rc.functions):
        if len(fnc) == 2:
            rc.functions[i] = (fnc[0], fnc[1], fnc[1])
        load_pysrcmod(fnc[1], rc)
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
        #pxd_base = desc['pxd_filename'].rsplit('.', 1)[0]         # eg, fccomp
        pxd_base = tarname or srcname  # eg, fccomp
        #cpppxd_base = desc['cpppxd_filename'].rsplit('.', 1)[0]   # eg, cpp_fccomp
        cpppxd_base = 'cpp_' + (tarname or srcname)   # eg, cpp_fccomp
        class_c2py = ('{pytype}({var})', 
                      ('{proxy_name} = {pytype}()\n'
                       '(<{ctype} *> {proxy_name}._inst)[0] = {var}'),
                      ('if {cache_name} is None:\n'
                       '    {proxy_name} = {pytype}()\n'
                       '    {proxy_name}._free_inst = False\n'
                       '    {proxy_name}._inst = &{var}\n'
                       '    {cache_name} = {proxy_name}\n')
                     )
        class_py2c = ('{proxy_name} = <{cytype}> {var}', '(<{ctype} *> {proxy_name}._inst)[0]')
        class_cimport = (rc.package, cpppxd_base) 
        ts.register_class(classname,                              # FCComp
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

    # next, make cython bindings
    # generate first, then write out to ensure this is atomic per-class
    if rc.make_cythongen:
        print("cythongen: creating C/C++ API wrappers")
        cpppxds = gencpppxd(env)
        pxds = genpxd(env)
        pyxs = genpyx(env, classes)
        for key, cpppxd in cpppxds.items():
            newoverwrite(cpppxd, os.path.join(rc.package, env[key]['cpppxd_filename']), rc.verbose)
        for key, pxd in pxds.items():
            newoverwrite(pxd, os.path.join(rc.package, env[key]['pxd_filename']), rc.verbose)
        for key, pyx in pyxs.items():
            newoverwrite(pyx, os.path.join(rc.package, env[key]['pyx_filename']), rc.verbose)

    # next, make cyclus bindings
    if rc.make_cyclus:
        print("making cyclus bindings")

def dumpdesc(rc):
    """Prints the current contents of the description cache using rc.
    """
    print(str(rc._cache))

def setuprc(rc):
    """Makes and validates a run control object and the environment it specifies."""
    if rc.package is NotSpecified:
        sys.exit("no package name given; please add 'package' to {0}".format(rc.rc))
    if isinstance(rc.parsers, basestring):
        if '[' in rc.parsers or '{' in  rc.parsers:
            rc.parsers = eval(rc.parsers)
    if rc.packagedir is NotSpecified:
        rc.packagedir = rc.package.replace('.', os.path.sep)
    if not os.path.isdir(rc.packagedir):
        os.makedirs(rc.packagedir)
    if not os.path.isdir(rc.sourcedir):
        os.makedirs(rc.sourcedir)
    if not os.path.isdir(rc.builddir):
        os.makedirs(rc.builddir)
    writenewonly("", os.path.join(rc.packagedir, '__init__.py'), rc.verbose)
    writenewonly("", os.path.join(rc.packagedir, '__init__.pxd'), rc.verbose)

defaultrc = RunControl(
    rc="xdressrc.py", 
    debug=False,
    make_extratypes=True,
    make_stlcontainers=True,
    make_cythongen=True,
    make_cyclus=False,
    dumpdesc=False,
    includes=[],
    verbose=False,
    package=NotSpecified,
    packagedir=NotSpecified,
    sourcedir='src',
    builddir='build',
    extra_types='xdress_extra_types',
    stlcontainers=[],
    stlcontainers_module='stlcontainers',
    parsers={'c': ['pycparser', 'gccxml', 'clang'], 
             'cpp':['gccxml', 'clang', 'pycparser'] },
    )

def main_setup():
    """Setup xdress API generation."""
    parser = argparse.ArgumentParser("Generates XDress API")
    parser.add_argument('--rc', default=NotSpecified, 
                        help="path to run control file")
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', 
                        default=NotSpecified, help="print more output")
    parser.add_argument('--debug', action='store_true', default=NotSpecified, 
                        help='build in debugging mode')    
    parser.add_argument('--make-extratypes', action='store_true', 
                        dest='make_extratypes', default=NotSpecified, 
                        help="make extra types wrapper")
    parser.add_argument('--no-make-extratypes', action='store_false', 
                        dest='make_extratypes', default=NotSpecified, 
                        help="don't make extra types wrapper")
    parser.add_argument('--make-stlcontainers', action='store_true', 
                        dest='make_stlcontainers', default=NotSpecified,
                        help="make STL container wrappers")
    parser.add_argument('--no-make-stlcontainers', action='store_false', 
                        dest='make_stlcontainers', default=NotSpecified,
                        help="don't make STL container wrappers")
    parser.add_argument('--make-cythongen', action='store_true', 
                        dest='make_cythongen', default=NotSpecified,
                        help="make cython bindings")
    parser.add_argument('--no-make-cythongen', action='store_false', 
                        dest='make_cythongen', default=NotSpecified,
                        help="don't make cython bindings")
    parser.add_argument('--make-cyclus', action='store_true', 
                        dest='make_cyclus', default=NotSpecified, 
                        help="make cyclus bindings")
    parser.add_argument('--no-make-cyclus', action='store_false', 
                        dest='make_cyclus', default=NotSpecified, 
                        help="don't make cyclus bindings")
    parser.add_argument('--dumpdesc', action='store_true', dest='dumpdesc', 
                        default=NotSpecified, help="print description cache")
    parser.add_argument('-I', '--includes', action='store', dest='includes', nargs="+",
                        default=NotSpecified, help="additional include dirs")
    parser.add_argument('--builddir', action='store', dest='builddir', 
                        default=NotSpecified, help="path to build directory")
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

    if rc.dumpdesc:
        dumpdesc(rc)
        sys.exit()

    setuprc(rc)
    return rc

def main_body(rc):
    """Body for xdress API generation."""
    # set typesystem defaults
    ts.EXTRA_TYPES = rc.extra_types
    ts.STLCONTAINERS = rc.stlcontainers_module

    if rc.make_extratypes:
        genextratypes(rc)

    if rc.make_stlcontainers:
        genstlcontainers(rc)

    if rc.make_cythongen:
        genbindings(rc)

def main():
    """Entry point for xdress API generation."""
    rc = main_setup()
    try:
        main_body(rc)
    except Exception as e:
        if rc.debug:
            import traceback
            sep = r'~\_/' * 17 + '~=[,,_,,]:3\n\n'
            with io.open(os.path.join(rc.builddir, 'debug.txt'), 'a+b') as f:
                f.write('{0}xdress failed with the following error:\n\n'.format(sep))
                traceback.print_exc(None, f)
                msg = '\n{0}Run control run-time contents:\n\n{1}\n\n'
                f.write(msg.format(sep, rc._pformat()))
                msg = '\n{0}Current descripton cache contents:\n\n{1}\n\n'
                f.write(msg.format(sep, str(rc._cache)))
            raise 
        else:
            sys.exit(str(e))

if __name__ == '__main__':
    main()
