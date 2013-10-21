"""Version information about xdress and its dependencies.
"""

import re

try:
    import Cython
except ImportError:
    Cython = None

_cyver_r = re.compile('(\d+)\.(\d+)\.?(\d+)?(.*)')

if Cython is None:
    CYTHON_VERSION = CYTHON_MAJOR = CYTHON_MINOR = CYTHON_MICRO = CYTHON_EXTRA =None
else:
    CYTHON_VERSION = Cython.__version__
    _cygs = _cyver_r.match(CYTHON_VERSION).groups()
    CYTHON_MAJOR = int(_cygs[0])
    CYTHON_MINOR = int(_cygs[1])
    CYTHON_MICRO = int(_cygs[2] or 0)
    CYTHON_EXTRA = _cygs[3]
    del _cygs
