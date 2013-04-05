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
    usage: Generates XDress API [-h] [--rc RC] [--debug] [--no-extratypes]
                                [--no-stlcont] [--no-cython] [--no-cyclus]
                                [--dump-desc] [-I INCLUDES [INCLUDES ...]] [-v]

    optional arguments:
      -h, --help            show this help message and exit
      --rc RC               path to run control file.
      --debug               build with debugging flags
      --no-extratypes       don't make extr types wrapper
      --no-stlcont          don't make STL container wrappers
      --no-cython           don't make cython bindings
      --no-cyclus           don't make cyclus bindings
      --dump-desc           print description cache
      -I INCLUDES [INCLUDES ...]
                            additional include dirs
      -v, --verbose         print more output

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
import os
import sys
import argparse
from pprint import pprint
from hashlib import md5
try:
    import cPickle as pickle
except ImportError:
    import pickle

from utils import newoverwrite, newcopyover, ensuredirs, writenewonly
import typesystem as ts
import stlwrap
from cythongen import gencpppxd, genpxd, genpyx
import autodescribe 


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
            with open(cachefile, 'r') as f:
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
        with open(filename, 'r') as f:
            filestr = f.read()
        currhash = md5(filestr).hexdigest()
        return cachehash == currhash

    def __getitem__(self, key):
        return self.cache[key][1]  # return the description only

    def __setitem__(self, key, value):
        name, filename, kind = key
        with open(filename, 'r') as f:
            filestr = f.read()
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
        with open(self.cachefile, 'w') as f:
            pickle.dump(self.cache, f, pickle.HIGHEST_PROTOCOL)

    def __str__(self):
        from pprint import pformat
        return pformat(self.cache)


# singleton
cache = DescriptionCache()

pysrcenv = {}

def load_pysrcmod(srcname, ns, rc):
    """Loads a module dictionary from a src file into the pysrcenv cache."""
    if srcname in pysrcenv:
        return 
    pyfilename = os.path.join(rc.sourcedir, srcname + '.py')
    if os.path.isfile(pyfilename):
        glbs = globals()
        locs = {}
        execfile(pyfilename, glbs, locs)
        if 'mod' not in locs:
            pymod = {}
        elif callable(locs['mod']):
            pymod = eval('mod()', glbs, locs)
        else:
            pymod = locs['mod']
    else:
        pymod = {}
    pysrcenv[srcname] = pymod


def compute_desc(name, srcname, tarname, kind, ns, rc):
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
    # C++ description
    cppfilename = os.path.join(rc.sourcedir, srcname + '.cpp')
    if cache.isvalid(name, cppfilename, kind):
        cppdesc = cache[name, cppfilename, kind]
    else:
        cppdesc = autodescribe.describe(cppfilename, name=name, kind=kind,
                                        includes=ns.includes + rc.includes,
                                        verbose=ns.verbose)
        cache[name, cppfilename, kind] = cppdesc

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

def genextratypes(ns, rc):
    d = os.path.split(__file__)[0]
    srcs = [os.path.join(d, 'xdress_extra_types.h'), 
            os.path.join(d, 'xdress_extra_types.pxd'), 
            os.path.join(d, 'xdress_extra_types.pyx')]
    tars = [os.path.join(rc.sourcedir, rc.extra_types + '.h'), 
            os.path.join(rc.packagedir, rc.extra_types + '.pxd'), 
            os.path.join(rc.packagedir, rc.extra_types + '.pyx')]
    for src, tar in zip(srcs, tars):
        with open(src, 'r') as f:
            s = f.read()
            s = s.format(extra_types=rc.extra_types)
            newoverwrite(s, tar, ns.verbose)

def genstlcontainers(ns, rc):
    print "stlwrap: generating C++ standard library wrappers & converters"
    fname = os.path.join(rc.packagedir, rc.stlcontainers_module)
    ensuredirs(fname)
    testname = os.path.join(rc.packagedir, 'tests', 'test_' + rc.stlcontainers_module)
    ensuredirs(testname)
    stlwrap.genfiles(rc.stlcontainers, fname=fname, testname=testname, 
                     package=rc.package, verbose=ns.verbose)


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

def genbindings(ns, rc):
    """Generates bidnings using the command line setting specified in ns.
    """
    print("cythongen: scraping C/C++ APIs from source")

    ns.cyclus = False  # FIXME cyclus bindings don't exist yet!
    for i, cls in enumerate(rc.classes):
        if len(cls) == 2:
            rc.classes[i] = (cls[0], cls[1], cls[1])
        load_pysrcmod(cls[1], ns, rc)        
    for i, fnc in enumerate(rc.functions):
        if len(fnc) == 2:
            rc.functions[i] = (fnc[0], fnc[1], fnc[1])
        load_pysrcmod(fnc[1], ns, rc)
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
                                                 'class', ns, rc)
        if ns.verbose:
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
        desc = compute_desc(funcname, srcname, tarname, 'func', ns, rc)
        if ns.verbose:
            pprint(desc)
        cache.dump()
        _adddesc2env(desc, env, funcname, srcname, tarname)

    # next, make cython bindings
    # generate first, then write out to ensure this is atomic per-class
    if ns.cython:
        print("cythongen: creating C/C++ API wrappers")
        cpppxds = gencpppxd(env)
        pxds = genpxd(env)
        pyxs = genpyx(env, classes)
        for key, cpppxd in cpppxds.iteritems():
            newoverwrite(cpppxd, os.path.join(rc.package, env[key]['cpppxd_filename']), ns.verbose)
        for key, pxd in pxds.iteritems():
            newoverwrite(pxd, os.path.join(rc.package, env[key]['pxd_filename']), ns.verbose)
        for key, pyx in pyxs.iteritems():
            newoverwrite(pyx, os.path.join(rc.package, env[key]['pyx_filename']), ns.verbose)

    # next, make cyclus bindings
    if ns.cyclus:
        print("making cyclus bindings")

def dumpdesc(ns):
    """Prints the current contents of the description cache using ns.
    """
    print str(DescriptionCache())

def setuprc(ns):
    """Makes and validates a run control namespace."""
    rc = dict(
        package=None,
        packagedir=None,
        sourcedir='src',
        extra_types='xdress_extra_types',
        stlcontainers=[],
        stlcontainers_module='stlcontainers',
        )
    execfile(ns.rc, rc, rc)
    rc = argparse.Namespace(**rc)
    rc.includes = list(rc.includes) if hasattr(rc, 'includes') else []
    if rc.package is None:
        sys.exit("no package name given; please add 'package' to xdressrc.py")
    if rc.packagedir is None:
        rc.packagedir = rc.package.replace('.', os.path.sep)
    if not os.path.isdir(rc.packagedir):
        os.makedirs(rc.packagedir)
    if not os.path.isdir(rc.sourcedir):
        os.makedirs(rc.sourcedir)
    writenewonly("", os.path.join(rc.packagedir, '__init__.py'), ns.verbose)
    writenewonly("", os.path.join(rc.packagedir, '__init__.pxd'), ns.verbose)
    return rc

def main():
    """Entry point for xdress API generation."""
    parser = argparse.ArgumentParser("Generates XDress API")
    parser.add_argument('--rc', default="xdressrc.py", 
                        help="path to run control file.")
    parser.add_argument('--debug', action='store_true', default=False, 
                        help='build with debugging flags')    
    parser.add_argument('--no-extratypes', action='store_false', dest='extratypes', 
                        default=True, help="don't make extr types wrapper")
    parser.add_argument('--no-stlcont', action='store_false', dest='stlcont', 
                        default=True, help="don't make STL container wrappers")
    parser.add_argument('--no-cython', action='store_false', dest='cython', 
                        default=True, help="don't make cython bindings")
    parser.add_argument('--no-cyclus', action='store_false', dest='cyclus', 
                        default=True, help="don't make cyclus bindings")
    parser.add_argument('--dump-desc', action='store_true', dest='dumpdesc', 
                        default=False, help="print description cache")
    parser.add_argument('-I', action='store', dest='includes', nargs="+",
                        default=[], help="additional include dirs")
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', 
                        default=False, help="print more output")
    ns = parser.parse_args()

    if ns.dumpdesc:
        dumpdesc(ns)
        return 

    rc = setuprc(ns)

    # set typesystem defaults
    ts.EXTRA_TYPES = rc.extra_types
    ts.STLCONTAINERS = rc.stlcontainers_module

    if ns.extratypes:
        genextratypes(ns, rc)

    if ns.stlcont:
        genstlcontainers(ns, rc)

    if ns.cython or ns.cyclus:
        genbindings(ns, rc)


if __name__ == '__main__':
    main()
