#!/usr/bin/env python
from __future__ import print_function 
import os
import sys
import glob
import json
from distutils.file_util import copy_file, move_file
from distutils.dir_util import mkpath, remove_tree
from copy import deepcopy


INFO = {
    'version': '0.2',
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


def metadata(path="xdress/metadata.json"):
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
    mdpath = os.path.join('xdress', 'metadata.json')
    if os.path.exists(mdpath):
        with open(mdpath) as f:
            metadata = json.load(f)
    if metadata is not None:
        msg = "\n\nCURRENT METADATA:\n"
        for k, v in sorted(metadata.items()):
            msg += "  {0} = {1}\n".format(k, repr(v))
        print(msg[:-1])

    if os.name != 'nt':
        return
    print(msg)

long_desc = """XDress
======
XDress is an automatic wrapper generator for C/C++ written in pure Python. Currently,
xdress may generate Python bindings (via Cython) for C++ classes & functions
and in-memory wrappers for C++ standard library containers (sets, vectors, maps).
In the future, other tools and bindings will be supported.

The main enabling feature of xdress is a dynamic type system that was designed with
the purpose of API generation in mind.

XDress currently has the following external dependencies,

*Run Time:*

    #. `pycparser <https://bitbucket.org/eliben/pycparser>`_, optional for C
    #. `GCC-XML <http://www.gccxml.org/HTML/Index.html>`_, optional for C++
    #. `lxml <http://lxml.de/>`_, optional (but nice!)

*Compile Time:*

    #. `Cython <http://cython.org/>`_
    #. `NumPy <http://numpy.scipy.org/>`_

The source code for xdress may be found at the
`GitHub project site <http://github.com/scopatz/xdress>`_.
Or you may simply clone the development branch using git::

    git clone git://github.com/scopatz/xdress.git

`Go here for the latest version of the docs! <http://xdress.org/latest>`_

"""


def setup():
    from distutils import core
    if os.name == 'nt':
        scripts = [os.path.join('scripts', f) for f in os.listdir('scripts')]
    else:
        scripts = [os.path.join('scripts', f) for f in os.listdir('scripts')
                                                    if not f.endswith('.bat')]
    packages = ['xdress', ]
    pack_dir = {'xdress': 'xdress',}
    pack_data = {'xdress': ['*.pxd', '*.pyx', '*.h', '*.cpp']}
    setup_kwargs = {
        "name": "xdress",
        "version": INFO['version'],
        "description": 'xdress',
        "author": 'Anthony Scopatz',
        "author_email": 'xdress@googlegroups.com',
        "url": 'http://xdress.org/',
        "packages": packages,
        "package_dir": pack_dir,
        "package_data": pack_data,
        "scripts": scripts,
        "description": "Goes all J. Edgar Hoover on your code.",
        "long_description": long_desc,
        "download_url": "https://github.com/scopatz/xdress/zipball/0.2",
        "classifiers": ["License :: OSI Approved :: BSD License",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "Programming Language :: C",
            "Programming Language :: C++",
            "Programming Language :: Cython",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: Implementation :: CPython",
            "Topic :: Scientific/Engineering",
            "Topic :: Software Development :: Code Generators",
            "Topic :: Software Development :: Compilers",
            "Topic :: Utilities",
            ],
        "data_files": [("", ['license'])],
        }
    rtn = core.setup(**setup_kwargs)


if __name__ == "__main__":
    main()
