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

import typesystem as ts
import stlwrap
from cythongen import gencpppxd, genpxd, genpyx
from autodescribe import describe, merge_descriptions


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


def describe_class(classname, filename, verbose=False):
    """Returns a description dictionary for a class (called classname) 
    living in a file (called filename).  

    Parameters
    ----------
    classname : str
        Class to describe.
    filename : str
        File where the class is implemented.  Will remove leading ``'bright_'`` from
        filename for the puprose of generating other files.
    verbose : bool, optional
        Flag for printing extra information during description process.

    Returns
    -------
    desc : dict
        Description dictionary of the class.

    """
    # C++ description
    cppfilename = filename + '.cpp'
    if cache.isvalid(classname, cppfilename):
        cppdesc = cache[classname, cppfilename]
    else:
        cppdesc = describe(cppfilename, classname=classname, verbose=verbose)
        cache[classname, cppfilename] = cppdesc

    # python description
    if os.path.isfile(filename + '.py'):
        glbs = globals()
        locs = {}
        execfile(filename + '.py', glbs, locs)
        if 'desc' not in locs:
            pydesc = {}
        elif callable(locs['desc']):
            pydesc = eval('desc()', glbs, locs)
        else:
            pydesc = locs['desc']
    else:
        pydesc = {}

    desc = merge_descriptions([cppdesc, pydesc])
    basefilename = os.path.split(filename)[-1]
    dimfilename = basefilename[7:] if basefilename.startswith('bright_') else basefilename
    desc['cpp_filename'] = '{0}.cpp'.format(basefilename)
    desc['header_filename'] = '{0}.h'.format(basefilename)
    desc['metadata_filename'] = '{0}.py'.format(basefilename)
    desc['pxd_filename'] = '{0}.pxd'.format(dimfilename)
    desc['pyx_filename'] = '{0}.pyx'.format(dimfilename)
    desc['cpppxd_filename'] = 'cpp_{0}.pxd'.format(dimfilename)
    return desc

# Classes and type to preregister with the typesyetem prior to doing any code 
# generation.  
PREREGISTER_KEYS = ['name', 'cython_c_type', 'cython_cimport', 'cython_cy_type',
                    'cython_py_type', 'cython_template_class_name', 
                    'cython_cyimport', 'cython_pyimport', 'cython_c2py', 'cython_py2c']
PREREGISTER_CLASSES = [
    ('Material', 'cpp_material.Material', ('pyne', 'cpp_material'), 
     'material._Material', 'material.Material', 'Material', ('pyne', 'material'), 
     ('pyne', 'material'), 
     ('{pytype}({var})', 
      ('{proxy_name} = {pytype}()\n'
       '{proxy_name}.mat_pointer[0] = {var}'),
      ('if {cache_name} is None:\n'
       '    {proxy_name} = {pytype}(free_mat=False)\n'
       '    {proxy_name}.mat_pointer = &{var}\n'
       '    {cache_name} = {proxy_name}\n')
     ),
     ('{proxy_name} = {pytype}({var}, free_mat=not isinstance({var}, {cytype}))',
      '{proxy_name}.mat_pointer[0]')),
    ]

def newoverwrite(s, filename):
    """Useful for not forcing re-compiles and thus playing nicely with the 
    build system.  This is acomplished by not writing the file if the existsing
    contents are exactly the same as what would be written out.

    Parameters
    ----------
    s : str
        string contents of file to possible
    filename : str
        Path to file.

    """
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            old = f.read()
        if s == old:
            return 
    with open(filename, 'w') as f:
        f.write(s)

def newcopyover(f1, f2):
    """Useful for not forcing re-compiles and thus playing nicely with the 
    build system.  This is acomplished by not writing the file if the existsing
    contents are exactly the same as what would be written out.

    Parameters
    ----------
    f1 : str
        Path to file to copy from
    f2 : str
        Path to file to copy over

    """
    if os.path.isfile(f1):
        with open(f1, 'r') as f:
            s = f.read()
        return newoverwrite(s, f2)

def ensuredirs(f):
    """For a file path, ensure that its directory path exists."""
    d = os.path.split(f)[0]
    if not os.path.isdir(d):
        os.makedirs(d)

def genstlcontainers(ns, rc):
    print "generating C++ standard library wrappers & converters"
    fname = os.path.join(rc.packagedir, rc.stlcontainers_module)
    ensuredirs(fname)
    testname = os.path.join(rc.packagedir, 'tests', 'test_' + rc.stlcontainers_module)
    ensuredirs(testname)
    stlwrap.genfiles(rc.stlcontainers, fname=fname, testname=testname)
    d = os.path.split(__file__)[0]
    xetsrc = [os.path.join(d, 'xdress_extra_types.pxd'), 
              os.path.join(d, 'xdress_extra_types.pyx')]
    xettar = [os.path.join(rc.packagedir, rc.extra_types + '.pxd'), 
              os.path.join(rc.packagedir, rc.extra_types + '.pyx')]
    for src, tar in zip(xetsrc, xettar):
        newcopyover(src, tar)

def genbindings(ns, rc):
    """Generates bidnings using the command line setting specified in ns.
    """
    ns.cyclus = False  # FIXME cyclus bindings don't exist yet!

    # compute all descriptions first 
    env = {}
    for classname, fname, mkcython, mkcyclus in CLASSES:
        print("parsing " + classname)
        desc = env[classname] = describe_class(classname, 
                                               os.path.join('cpp', fname), 
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
        class_cimport = ('bright', cpppxd_base) 
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

    # now preregister types with the type system
    for prc in PREREGISTER_CLASSES:
        ts.register_class(**dict(zip(PREREGISTER_KEYS, prc)))

    # Now register specialization
    ts.register_specialization(('map', 'str', ('Material', '*'), 0), 
        cython_c_type='material._MapStrMaterial', 
        cython_cy_type='material._MapStrMaterial', 
        cython_py_type='material.MapStrMaterial',
        cython_cimport=(('pyne', 'material'),),
        cython_cyimport=(('pyne', 'material'),),
        cython_pyimport=(('pyne', 'material'),),
        )

    # next, make cython bindings
    for classname, fname, mkcython, mkcyclus in CLASSES:
        if not mkcython or not ns.cython:
            continue
        print("making cython bindings for " + classname)
        # generate first, then write out to ensure this is atomic per-class
        desc = env[classname]
        cpppxd = gencpppxd(desc)
        pxd = genpxd(desc)
        pyx = genpyx(desc, env)
        newoverwrite(cpppxd, os.path.join('bright', desc['cpppxd_filename']))
        newoverwrite(pxd, os.path.join('bright', desc['pxd_filename']))
        newoverwrite(pyx, os.path.join('bright', desc['pyx_filename']))

    # next, make cyclus bindings
    for classname, fname, mkcython, mkcyclus in CLASSES:
        if not mkcyclus or not ns.cyclus:
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
    parser.add_argument('--no-cython', action='store_false', dest='cython', 
                        default=True, help="don't make cython bindings")
    parser.add_argument('--no-cyclus', action='store_false', dest='cyclus', 
                        default=True, help="don't make cyclus bindings")
    parser.add_argument('--no-stlcont', action='store_false', dest='stlcont', 
                        default=True, help="don't make STL container wrappers")
    parser.add_argument('--dump-desc', action='store_true', dest='dumpdesc', 
                        default=False, help="print description cache")
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', 
                        default=False, help="print more output")
    ns = parser.parse_args()

    rc = dict(defaultrc)
    execfile(ns.rc, rc, rc)
    rc = argparse.Namespace(**rc)

    if ns.dumpdesc:
        dumpdesc(ns)
        return 

    if ns.stlcont:
        genstlcontainers(ns, rc)

    if ns.cython or ns.cyclus:
        genbindings(ns, rc)


if __name__ == '__main__':
    main()
