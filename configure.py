#!/usr/bin/env python
from __future__ import print_function
import os
import io
import sys
import json

INFO = {
    'version': '0.3',
}


def main():
    "Run functions specified on the command line"
    if len(sys.argv) <= 1:
        raise SystemExit("no command(s) specified")
    cmds = sys.argv[1:]
    if '-h' in cmds or '--help' in cmds:
        msg = "usage: " + sys.argv[0] + " <func-name> [<func-name>]"
        raise SystemExit(msg)
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

dir_name = os.path.dirname(os.path.abspath(__file__))
fname = os.path.join(dir_name, 'docs', 'index.rst')
with io.open(fname, 'r') as f:
    long_desc = f.read()

long_desc = "\n".join([l for l in long_desc.splitlines() if ":ref:" not in l])
long_desc = "\n".join([l for l in long_desc.splitlines()
                       if ".. toctree::" not in l])
long_desc = "\n".join([l for l in long_desc.splitlines()
                       if ":maxdepth:" not in l])


def setup():
    try:
        from setuptools import setup as setup_
    except ImportError:
        from distutils.core import setup as setup_

    scripts_dir = os.path.join(dir_name, 'scripts')
    if os.name == 'nt':
        scripts = [os.path.join(scripts_dir, f)
                   for f in os.listdir(scripts_dir)]
    else:
        scripts = [os.path.join(scripts_dir, f)
                   for f in os.listdir(scripts_dir)
                   if not f.endswith('.bat')]
    packages = ['xdress',]
    pack_dir = {'xdress': 'xdress', }
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
        "classifiers": [
            "License :: OSI Approved :: BSD License",
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
        "data_files": [("", ['license', 'configure.py']),],
    }
    # changing dirs for virtualenv
    cwd = os.getcwd()
    os.chdir(dir_name)
    setup_(**setup_kwargs)
    os.chdir(cwd)


if __name__ == "__main__":
    main()
