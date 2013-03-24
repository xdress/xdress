#!/bin/bash
export INST=$(pwd)/inst
export SITEPATH=${INST}/lib/python2.7/site-packages
python ../xdress/main.py --no-cyclus && \
python setup.py install --prefix=$INST -- -- && \
cd xdtest/tests && \
PYTHONPATH=$SITEPATH nosetests test_xdstlc.py && \
PYTHONPATH=$SITEPATH python -c "import xdtest.fccomp" && \
PYTHONPATH=$SITEPATH python -c "import xdtest.enrichment_parameters" && \
PYTHONPATH=$SITEPATH python -c "import xdtest.enrichment" && \
PYTHONPATH=$SITEPATH python -c "import xdtest.reprocess" && \
echo "Ran imports OK" && \
cd ../..
