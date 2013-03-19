#!/bin/bash
export INST=$(pwd)/inst
export SITEPATH=${INST}/lib/python2.7/site-packages
python ../xdress/main.py --no-cython --no-cyclus && \
python setup.py install --prefix=$INST -- -- && \
PYTHONPATH=$SITEPATH nosetests xdtest/tests/test_stlcontainers.py
