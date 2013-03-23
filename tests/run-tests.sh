#!/bin/bash
export INST=$(pwd)/inst
export SITEPATH=${INST}/lib/python2.7/site-packages
cd xdtest/tests && \
PYTHONPATH=$SITEPATH nosetests test_xdstlc.py && \
cd ../..
