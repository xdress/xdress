#!/bin/bash
export INST=$(pwd)/inst
export SITEPATH=${INST}/lib/python2.7/site-packages
#python ../xdress/main.py --no-cyclus && \
python ../xdress/main.py --no-cyclus && \
python setup.py install --prefix=$INST -- -- && \
cd xdtest/tests && \
PYTHONPATH=$SITEPATH nosetests test_xdstlc.py && \
PYTHONPATH=$SITEPATH python -c "import xdtest.fccomp as x; print x.__doc__" && \
PYTHONPATH=$SITEPATH python -c "import xdtest.enrichment_parameters as x; print x.__doc__" && \
PYTHONPATH=$SITEPATH python -c "import xdtest.enrichment as x; print x.__doc__" && \
PYTHONPATH=$SITEPATH python -c "import xdtest.reprocess as x; print x.__doc__" && \
echo "Ran imports OK" && \
cd ../..
