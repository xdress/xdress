"""Top-level automatic API generators for Bright.  

:author: Anthony Scopatz <scopatz@gmail.com>

API Generation
==============
This module is where all of the Bright-specific API generation routines reside.
Until now the type system, automatic description, and cython code generation have
all be independent of what classes they are wrapping.  The functions and classes
here are specifically set up to be executed on the Bright code base.  Thus, 
attempting to wrap other codes with the tools developed here would only need to fork
this module. 

The main module is normally run from the base bright directory as follows:

.. code-block:: bash

    ~ $ cd bright
    ~/bright $ python bright/apigen/main.py

The function here has the following command line interface::

    usage: Generates Bright API [-h] [--debug] [--no-cython] [--no-cyclus]
                                [--dump-desc] [-v]

    optional arguments:
      -h, --help     show this help message and exit
      --debug        build with debugging flags
      --no-cython    don't make cython bindings
      --no-cyclus    don't make cyclus bindings
      --dump-desc    print description cache
      -v, --verbose  print more output

Main API
========
"""
import os
import argparse
from pprint import pprint
from hashlib import md5
try:
    import cPickle as pickle
except ImportError:
    import pickle

from utils import newoverwrite, newcopyover, ensuredirs
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

    def isvalid(self, classname, filename):
        """Boolean on whether the cach value for a (classname, filename)
        tuple matches the state of the file on the system."""
        key = (classname, filename)
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
        classname, filename = key
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


def describe_class(classname, srcname, tarname, rc, verbose=False):
    """Returns a description dictionary for a class (called classname) 
    living in a file (called filename).  

    Parameters
    ----------
    classname : str
        Class to describe.
    srcname : str
        File basename where the class is implemented.  
    tarname : str
        File basename where the class basenames will be generated.  
    verbose : bool, optional
        Flag for printing extra information during description process.

    Returns
    -------
    desc : dict
        Description dictionary of the class.

    """
    # C++ description
    cppfilename = os.path.join(rc.sourcedir, srcname + '.cpp')
    if cache.isvalid(classname, cppfilename):
        cppdesc = cache[classname, cppfilename]
    else:
        cppdesc = autodescribe.describe(cppfilename, classname=classname, 
                                        verbose=verbose)
        cache[classname, cppfilename] = cppdesc

    # python description
    pyfilename = os.path.join(rc.sourcedir, srcname + '.py')
    if os.path.isfile(pyfilename):
        glbs = globals()
        locs = {}
        execfile(pyfilename, glbs, locs)
        if 'desc' not in locs:
            pydesc = {}
        elif callable(locs['desc']):
            pydesc = eval('desc()', glbs, locs)
        else:
            pydesc = locs['desc']
    else:
        pydesc = {}

    desc = autodescribe.merge_descriptions([cppdesc, pydesc])
    desc['cpp_filename'] = '{0}.cpp'.format(srcname)
    desc['header_filename'] = '{0}.h'.format(srcname)
    desc['metadata_filename'] = '{0}.py'.format(srcname)
    desc['pxd_filename'] = '{0}.pxd'.format(tarname)
    desc['pyx_filename'] = '{0}.pyx'.format(tarname)
    desc['cpppxd_filename'] = 'cpp_{0}.pxd'.format(tarname)
    return desc

def genextratypes(ns, rc):
    d = os.path.split(__file__)[0]
    srcs = [os.path.join(d, 'xdress_extra_types.pxd'), 
            os.path.join(d, 'xdress_extra_types.pyx')]
    tars = [os.path.join(rc.packagedir, rc.extra_types + '.pxd'), 
            os.path.join(rc.packagedir, rc.extra_types + '.pyx')]
    newcopyover(srcs[0], tars[0])
    with open(srcs[1], 'r') as f:
        s = f.read()
    s = s.format(extra_types=rc.extra_types)
    newoverwrite(s, tars[1])

def genstlcontainers(ns, rc):
    print "generating C++ standard library wrappers & converters"
    fname = os.path.join(rc.packagedir, rc.stlcontainers_module)
    ensuredirs(fname)
    testname = os.path.join(rc.packagedir, 'tests', 'test_' + rc.stlcontainers_module)
    ensuredirs(testname)
    stlwrap.genfiles(rc.stlcontainers, fname=fname, testname=testname, 
                     package=rc.package)

def genbindings(ns, rc):
    """Generates bidnings using the command line setting specified in ns.
    """
    genextratypes(ns, rc)
    ns.cyclus = False  # FIXME cyclus bindings don't exist yet!
    for i, cls in enumerate(rc.classes):
        if len(cls) == 2:
            rc.classes[i] = (cls[0], cls[1], cls[1])

    # compute all descriptions first 
    env = {}
    for classname, srcname, tarname in rc.classes:
        print("parsing " + classname)
        desc = env[classname] = describe_class(classname, srcname, tarname, rc, 
                                               verbose=ns.verbose)
        if ns.verbose:
            pprint(env[classname])

        print("registering " + classname)
        pxd_base = desc['pxd_filename'].rsplit('.', 1)[0]         # eg, fccomp
        cpppxd_base = desc['cpppxd_filename'].rsplit('.', 1)[0]   # eg, cpp_fccomp
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


    # next, make cython bindings
    for classname, srcname, tarname in rc.classes:
        if not ns.cython:
            continue
        print("making cython bindings for " + classname)
        # generate first, then write out to ensure this is atomic per-class
        desc = env[classname]
        cpppxd = gencpppxd(desc)
        pxd = genpxd(desc)
        pyx = genpyx(desc, env)
        newoverwrite(cpppxd, os.path.join(rc.package, desc['cpppxd_filename']))
        newoverwrite(pxd, os.path.join(rc.package, desc['pxd_filename']))
        newoverwrite(pyx, os.path.join(rc.package, desc['pyx_filename']))

    # next, make cyclus bindings
    for classname, srcname, tarname in rc.classes:
        if not ns.cyclus:
            continue
        print("making cyclus bindings for " + classname)

def dumpdesc(ns):
    """Prints the current contents of the description cache using ns.
    """
    print str(DescriptionCache())


defaultrc = dict(
    package='<xdtest-pkg>',
    packagedir='<xdtest-pkgdir>',
    extra_types='xdress_extra_types',
    stlcontainers=[],
    stlcontainers_module='stlcontainers',
    )

def main():
    """Entry point for xdress API generation."""
    parser = argparse.ArgumentParser("Generates xdress API")
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
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', 
                        default=False, help="print more output")
    ns = parser.parse_args()

    rc = dict(defaultrc)
    execfile(ns.rc, rc, rc)
    rc = argparse.Namespace(**rc)

    # set typesystem defaults
    ts.EXTRA_TYPES = rc.extra_types
    ts.STLCONTAINERS = rc.stlcontainers_module

    if ns.dumpdesc:
        dumpdesc(ns)
        return 

    if ns.extratypes:
        genextratypes(ns, rc)

    if ns.stlcont:
        genstlcontainers(ns, rc)

    if ns.cython or ns.cyclus:
        genbindings(ns, rc)


if __name__ == '__main__':
    main()
