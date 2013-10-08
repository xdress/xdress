#!/usr/bin/env python
from __future__ import print_function 
import os
import sys
import subprocess

PKG = "cppproj"
version = "test"

def setup():
    try:
        from setuptools import setup as setup_
    except ImportError:
        from distutils.core import setup as setup_
    scripts = []
    packages = [PKG, PKG + '.lib']
    pack_dir = {PKG: PKG, PKG + '.lib': os.path.join(PKG, 'lib')}
    extpttn = ['*.dll', '*.so', '*.dylib', '*.pyd', '*.pyo']
    pack_data = {PKG: ['*.pxd', '*.json',] + extpttn, PKG + '.lib': extpttn}
    setup_kwargs = {
        "name": PKG,
        "version": version,
        "description": "The {0} package".format(PKG),
        "author": 'Anthony Scopatz',
        "author_email": 'scopatz@gmail.com',
        "url": 'http://github.com/scopatz/xdress',
        "packages": packages,
        "package_dir": pack_dir,
        "package_data": pack_data,
        "scripts": scripts,
        }
    rtn = setup_(**setup_kwargs)

def parse_args():
    distutils = []
    cmake = []
    make = []
    argsets = [distutils, cmake, make]
    i = 0
    for arg in sys.argv:
        if arg == '--':
            i += 1
        else:
            argsets[i].append(arg)
    hdf5opt = [o.split('=')[1] for o in distutils if o.startswith('--hdf5=')]
    if 0 < len(hdf5opt):
        os.environ['HDF5_ROOT'] = hdf5opt[0]  # Expose to CMake
        distutils = [o for o in distutils if not o.startswith('--hdf5=')]
    return distutils, cmake, make

def main_body():
    if not os.path.exists('build'):
        os.mkdir('build')
    sys.argv, cmake_args, make_args = parse_args()
    makefile = os.path.join('build', 'Makefile')
    if not os.path.exists(makefile):
        cmake_cmd = ['cmake', '..'] + cmake_args
        cmake_cmd += ['-DPYTHON_EXECUTABLE=' + sys.executable, ]
        if os.name == 'nt':
            files_on_path = set()
            for p in os.environ['PATH'].split(';')[::-1]:
                if os.path.exists(p):
                    files_on_path.update(os.listdir(p))
            if 'cl.exe' in files_on_path:
                pass
            elif 'sh.exe' in files_on_path:
                cmake_cmd += ['-G "MSYS Makefiles"']
            elif 'gcc.exe' in files_on_path:
                cmake_cmd += ['-G "MinGW Makefiles"']
            cmake_cmd = ' '.join(cmake_cmd)
        rtn = subprocess.check_call(cmake_cmd, cwd='build', shell=(os.name=='nt'))
    rtn = subprocess.check_call(['make'] + make_args, cwd='build')
    cwd = os.getcwd()
    os.chdir('build')
    setup()
    os.chdir(cwd)

def main():
    success = False
    try:
        main_body()
        success = True
    finally:
        print("success: {0}".format(success))

if __name__ == "__main__":
    main()
