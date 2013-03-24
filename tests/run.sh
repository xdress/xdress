#!/bin/bash
export INST=$(pwd)/inst
export SITEPATH=${INST}/lib/python2.7/site-packages
python ../xdress/main.py --no-cyclus && \
python setup.py install --prefix=$INST -- -- && \
cd xdtest/tests && \
PYTHONPATH=$SITEPATH nosetests test_xdstlc.py && \
cd ../..
