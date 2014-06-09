import os
import sys
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as np

if not os.path.exists('mypack/mypack_extra_types.h'):
    sys.exit("please run xdress first!")

incdirs = [os.path.join(os.getcwd(), 'src'), np.get_include()]

ext_modules = [
    Extension("mypack.mypack_extra_types", ["mypack/mypack_extra_types.pyx"], 
              include_dirs=incdirs, language="c++"),
    Extension("mypack.stlcontainers", ["mypack/stlcontainers.pyx"], 
              include_dirs=incdirs, language="c++"),
    Extension("mypack.hoover", ['src/hoover.cpp', "mypack/hoover.pyx", ],
    	      include_dirs=incdirs, language="c++"),
    Extension("mypack.hoover_b", ['src/hoover.cpp', "mypack/hoover_b.pyx", ],
    	      include_dirs=incdirs, language="c++"),
    ]

setup(  
  name = 'mypack',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules,
  packages = ['mypack']
)
