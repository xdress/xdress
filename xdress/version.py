"""Version information about xdress and its dependencies.
"""

import re
from collections import namedtuple

class version_info(namedtuple('version_info', ['major', 'minor', 'micro', 'extra'])):
    """A representation of version information.
    """
    def __new__(cls, major=-1, minor=-1, micro=-1, extra=''):
        return super(version_info, cls).__new__(cls, major, minor, micro, extra)

_ver_r = re.compile('(\d+)\.(\d+)\.?(\d+)?-?(.*)')

def version_parser(ver):
    """Parses a nominal version string into a version_info object.
    e.g. '0.20dev' -> version_info(0, 20, 0, 'dev').
    """
    m = _ver_r.match(cython_version)
    g = m.groups()
    vi = version_info(int(g[0]), int(g[1] or 0), int(g[2] or 0), g[3])
    return vi

#
# Cython
#

try:
    import Cython
except ImportError:
    Cython = None

if Cython is None:
    cython_version = None
    cython_version_info = version_info()
else:
    cython_version = Cython.__version__
    cython_version_info = version_parser(cython_version)
