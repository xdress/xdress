#!/usr/bin/env python
 
import os
import sys
import glob
import json
from distutils.file_util import copy_file, move_file
from distutils.dir_util import mkpath, remove_tree
from copy import deepcopy

from Cython.Compiler.Version import version as CYTHON_VERSION



INFO = {
    'version': '0.1-dev',
    }


def main():
    "Run functions specified on the command line"
    if len(sys.argv) <= 1:
        raise SystemExit("no command(s) specified")
    cmds = sys.argv[1:]
    if '-h' in cmds or '--help' in cmds:
        raise SystemExit("usage: " + sys.argv[0] + " <func-name> [<func-name>]")
    glbs = globals()
    for cmd in cmds:
        if cmd not in glbs:
            raise SystemExit(cmd + " not found")
    for cmd in cmds:
        if callable(glbs[cmd]):
            glbs[cmd]()
        else:
            raise SystemExit(cmd + " not callable")


def metadata(path="xdtest/metadata.json"):
    """Build a metadata file."""
    md = {}
    md.update(INFO)

    # FIXME: Add the contents of CMakeCache.txt to the metadata dictionary

    # write the metadata file
    with open(path, 'w') as f:
        json.dump(md, f, indent=2)

    return md


def final_message(success=True):
    if success:
        return

    metadata = None
    mdpath = os.path.join('xdtest', 'metadata.json')
    if os.path.exists(mdpath):
        with open(mdpath) as f:
            metadata = json.load(f)
    if metadata is not None:
        msg = "\n\nCURRENT METADATA:\n"
        for k, v in sorted(metadata.items()):
            msg += "  {0} = {1}\n".format(k, repr(v))
        print msg[:-1]

    if os.name != 'nt':
        return

    msg = ("\n\nUSAGE: "
           "python setup.py <distutils-args> [-- <cmake-arg>] [-- <make-args>]\n"
           "CMake and make command line arguments are optional, but must be preceeded "
           "by '--'.\n"
           "\n\nIf compilation is failing with HDF5 issues please try the "
           "following steps:\n\n"
           "    1. Install EPD [1].\n"
           "    2. Download the HDF5 Windows binarys from [2].\n"
           "    3. Unzip them to the C-drive (C:\\hdf5-{h5ver}).\n"
           "    4. Re-run setup with the '--hdf5' option:\n\n"
           "        python setup.py install --user --hdf5=C:\\hdf5-{h5ver}\n\n"
           "Should this still fail, please report your problem to scopatz@gmail.com\n\n"
           "[1] http://www.enthought.com/products/epd.php\n"
           "[2] http://www.hdfgroup.org/ftp/HDF5/releases/hdf5-{h5ver}/bin/windows/\n"
           ).format(h5ver=h5ver)
    print msg


def cython_version():
    pxi = ("# Cython compile-time version information\n"
           "DEF CYTHON_VERSION_MAJOR = {major}\n"
           "DEF CYTHON_VERSION_MINOR = {minor}\n"
           "DEF CYTHON_VERSION_MICRO = {micro}")
    cyver = CYTHON_VERSION.split('-')[0].split('.')
    while len(cyver) < 3:
        cyver = cyver + [0]
    cyver = dict([(k, int(cv)) for k, cv in zip(['major', 'minor', 'micro'], cyver)])
    pxi = pxi.format(**cyver)
    basedir = os.path.split(__file__)[0]
    incldir = os.path.join(basedir, 'bright', 'include')
    if not os.path.exists(incldir):
        os.mkdir(incldir)
    with open(os.path.join(incldir, 'cython_version.pxi'), 'w') as f:
        f.write(pxi)

def setup():
    from distutils import core
    scripts = []
    packages = ['xdtest', 'xdtest.lib']
    pack_dir = {'xdtest': 'xdtest', 'xdtest.lib': 'xdtest/lib'}
    extpttn = ['*.dll', '*.so', '*.dylib', '*.pyd', '*.pyo']
    pack_data = {'xdtest': ['*.pxd', '*.json',] + extpttn, 'xdtest.lib': extpttn}
    setup_kwargs = {
        "name": "xdtest",
        "version": INFO['version'],
        "description": 'xdtest',
        "author": 'Anthony Scopatz',
        "author_email": 'scopatz@gmail.com',
        "url": 'http://github.com/scopatz/xdress',
        "packages": packages,
        "package_dir": pack_dir,
        "package_data": pack_data,
        "scripts": scripts,
        }
    rtn = core.setup(**setup_kwargs)


if __name__ == "__main__":
    main()
