#!/bin/bash
export INST=$(pwd)/inst
export LOCPATH=$(pwd)/..
export SITEPATH=${INST}/lib/python2.7/site-packages
PYTHONPATH=$LOCPATH python2.7 ../scripts/xdress --debug && \
PYTHONPATH=$LOCPATH python2.7 ../scripts/xdress && \
python2.7 setup.py install --prefix=$INST -- -- && \
cd xdtest/tests && \
PYTHONPATH=$SITEPATH nosetests-2.7 test_xdstlc.py && \
PYTHONPATH=$SITEPATH python2.7 -c "import xdtest.bright as x; print 'imported bright\n'" && \
PYTHONPATH=$SITEPATH python2.7 -c "import xdtest.bright as x; print 'imported pybright\n'" && \
PYTHONPATH=$SITEPATH python2.7 -c "import xdtest.fccomp as x; print x.__doc__" && \
PYTHONPATH=$SITEPATH python2.7 -c "import xdtest.enrichment_parameters as x; print x.__doc__" && \
PYTHONPATH=$SITEPATH python2.7 -c "import xdtest.enrichment as x; print x.__doc__" && \
PYTHONPATH=$SITEPATH python2.7 -c "import xdtest.reprocess as x; print x.__doc__" && \
PYTHONPATH=$SITEPATH python2.7 -c "import xdtest.pydevice as x; print x.__doc__" && \
echo "Ran imports OK" && \
cd ../..
