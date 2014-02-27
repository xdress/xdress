#!/usr/bin/env python
from __future__ import print_function
import os
import io
import sys
import json
import glob
import subprocess

sys.path.insert(0, '')
import xdress.version
sys.path.pop(0)

# Fix bug in distutils for python 3
if sys.version_info[0] >= 3:
    def decode(s):
        return s if isinstance(s,str) else bytes.decode(s)
    import distutils.spawn
    old_spawn_posix = distutils.spawn._spawn_posix
    def hack_spawn_posix(cmd, search_path=1, verbose=0, dry_run=0):
        return old_spawn_posix(list(map(decode, cmd)), search_path, verbose, dry_run)
    distutils.spawn._spawn_posix = hack_spawn_posix

INFO = {
    'version': xdress.version.xdress_version,
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
        from setuptools import setup as setup_, Extension
    except ImportError:
        from distutils.core import setup as setup_, Extension
    from distutils.spawn import find_executable
    from distutils.sysconfig import get_config_vars

    scripts_dir = os.path.join(dir_name, 'scripts')
    if os.name == 'nt':
        scripts = [os.path.join(scripts_dir, f)
                   for f in os.listdir(scripts_dir)]
    else:
        scripts = [os.path.join(scripts_dir, f)
                   for f in os.listdir(scripts_dir)
                   if not f.endswith('.bat')]
    packages = ['xdress', 'xdress.clang', 'xdress._enum']
    pack_dir = {'xdress': 'xdress', 'xdress.clang': 'xdress/clang', 
                'xdress._enum': 'xdress/_enum'}
    pack_data = {'xdress': ['*.pxd', '*.pyx', '*.h', '*.cpp'], 
                 'xdress._enum': ['LICENSE', 'README']}

    # llvm+clang configuration can be controlled by the environment variables
    # LLVM_CONFIG, LLVM_CPPFLAGS, LLVM_LDFLAGS, and CLANG_LIBS.  LLVM_CONFIG is
    # not used if both LLVM_CPPFLAGS and LLVM_LDFLAGS are set.

    if 'LLVM_CPPFLAGS' in os.environ and 'LLVM_LDFLAGS' in os.environ:
        llvm_config = True # Will be unused below
    else:
        if 'LLVM_CONFIG' in os.environ:
            llvm_config = os.environ['LLVM_CONFIG']
        else:
            options = 'llvm-config llvm-config-3.5 llvm-config-3.4 llvm-config-3.3 llvm-config-3.2'.split()
            for p in options:
                p = find_executable(p)
                if p is not None:
                    print('using llvm-config from %s'%p)
                    llvm_config = p
                    break
            else:
                print('Disabling clang since llvm-config not found: tried %s'%', '.join(options))
                print('To override, set the LLVM_CONFIG environment variable.')
                llvm_config = None
    if llvm_config is not None:
        try:
            llvm_cppflags = (   os.environ.get('LLVM_CPPFLAGS')
                             or subprocess.check_output([llvm_config,'--cppflags'])).split()
            llvm_ldflags  = (   os.environ.get('LLVM_LDFLAGS')
                             or subprocess.check_output([llvm_config,'--ldflags','--libs'])).split()
        except OSError as e:
            raise OSError("Failed to run llvm-config program '%s': %s" % (llvm_config, e))
        clang_dir = os.path.join(dir_name, 'xdress', 'clang')
        clang_src_dir = os.path.join(clang_dir, 'src')
        clang_libs = (   os.environ.get('CLANG_LIBS')
                      or '''clangTooling clangFrontend clangDriver clangSerialization clangCodeGen
                            clangParse clangSema clangStaticAnalyzerFrontend clangStaticAnalyzerCheckers
                            clangStaticAnalyzerCore clangAnalysis clangARCMigrate clangEdit
                            clangRewriteCore clangAST clangLex clangBasic''').split()
        # If the user sets CFLAGS, make sure we still have our own include path first
        if 'CFLAGS' in os.environ:
            os.environ['CFLAGS'] = '-I%s '%clang_dir + os.environ['CFLAGS']
        # Remove -Wstrict-prototypes to prevent warnings in libclang C++ code,
        # following http://stackoverflow.com/questions/8106258.
        opt, = get_config_vars('OPT')
        os.environ['OPT'] = ' '.join(f for f in opt.split() if f != '-Wstrict-prototypes')
        modules = [Extension('xdress.clang.libclang',
                             sources=glob.glob(os.path.join(clang_src_dir, '*.cpp')),
                             define_macros=[('XDRESS', 1)],
                             include_dirs=[clang_dir],
                             extra_compile_args=llvm_cppflags+['-fno-rtti'],
                             extra_link_args=llvm_ldflags,
                             libraries=clang_libs,
                             language='c++')]
    else:
        modules = ()

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
        "ext_modules": modules,
        "scripts": scripts,
        "description": ("Cython-based, NumPy-aware automatic wrapper generation for "
                        "C / C++."),
        "long_description": long_desc,
        "download_url": ("https://github.com/scopatz/xdress/"
                         "zipball/{0}.{1}").format(*xdress.version.xdress_version[:2]),
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
