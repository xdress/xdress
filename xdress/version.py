"""Version information about xdress and its dependencies.
"""

import re
from collections import namedtuple

class version_info(namedtuple('version_info', ['major', 'minor', 'micro', 'extra'])):
    """A representation of version information.
    """
    def __new__(cls, major=-1, minor=-1, micro=-1, extra=''):
        return super(version_info, cls).__new__(cls, major, minor, micro, extra)

_ver_r = re.compile('(\d+)\.(\d+)\.?(\d+)?[-_ \.]*?(.*)')

def version_parser(ver):
    """Parses a nominal version string into a version_info object.
    e.g. '0.20dev' -> version_info(0, 20, 0, 'dev').
    """
    m = _ver_r.match(ver)
    g = m.groups()
    vi = version_info(int(g[0]), int(g[1] or 0), int(g[2] or 0), g[3])
    return vi

def report_versions():
    """Creates a string that reports the version of xdress and all its
    dependencies.
    """
    vstr = ("XDress: {xdress_version}\n"
            "pycparser (optional): {pycparser_version}\n"
            "lxml (optional): {lxml_version}\n"
            "NumPy (optional): {numpy_version}\n"
            "Cython (optional): {cython_version}"
            )
    return vstr.format(**globals())

#
# XDress
#

xdress_version = '0.5-dev'
xdress_version_info = version_info(0, 5, 0, 'dev')

#
# pycparser
#

try:
    import pycparser
except ImportError:
    pycparser = None

if pycparser is None:
    pycparser_version = None
    pycparser_version_info = version_info()
else:
    pycparser_version = pycparser.__version__
    pycparser_version_info = version_parser(pycparser_version)

#
# numpy
#

try:
    import numpy
except ImportError:
    numpy = None

if numpy is None:
    numpy_version = None
    numpy_version_info = version_info()
else:
    numpy_version = numpy.__version__
    numpy_version_info = version_parser(numpy_version)

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

#
# lxml
#

try:
    import lxml.etree
except ImportError:
    lxml = None

if lxml is None:
    lxml_version = None
    lxml_version_info = version_info()
else:
    lxml_version = lxml.etree.__version__
    lxml_version_info = version_parser(lxml_version)
