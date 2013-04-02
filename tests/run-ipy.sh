#!/bin/bash
export INST=$(pwd)/inst
export SITEPATH=${INST}/lib/python2.7/site-packages
cd gdb && \
PYTHONPATH=$SITEPATH ipython && \
cd ..
