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
    'version': '0.1',
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


def setup():
    from distutils import core
    scripts = [os.path.join('scripts', f) for f in os.listdir('scripts')]
    #scripts = [s for s in scripts if (os.name == 'nt' and s.endswith('.bat')) or 
    #                                 (os.name != 'nt' and not s.endswith('.bat'))]
    packages = ['xdress', ]
    pack_dir = {'xdress': 'xdress',}
    pack_data = {'xdress': ['*.pxd', '*.pyx', '*.h', '*.cpp']}
    setup_kwargs = {
        "name": "xdress",
        "version": INFO['version'],
        "description": 'xdress',
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
