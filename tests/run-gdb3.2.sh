#!/bin/bash
export INST=$(pwd)/inst
export SITEPATH=${INST}/lib/python3.2/site-packages
cd gdb
PYTHONPATH="$SITEPATH" gdb python3.2
cd .. 
