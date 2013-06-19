"""The base plugin for XDress.

This module is available as an xdress plugin by the name ``xdress.base``.

:author: Anthony Scopatz <scopatz@gmail.com>

Base Plugin API
===============
"""
from __future__ import print_function
import os
import sys

from .utils import RunControl, NotSpecified, writenewonly, DescriptionCache, \
    DEFAULT_RC_FILE, DEFAULT_PLUGINS
from .plugins import Plugin

if sys.version_info[0] >= 3:
    basestring = str

class XDressPlugin(Plugin):
    """This class provides base functionality for xdress itself."""

    defaultrc = RunControl(
        rc=DEFAULT_RC_FILE,
        plugins=DEFAULT_PLUGINS,
        debug=False,
        verbose=False,
        dumpdesc=False,
        package=NotSpecified,
        packagedir=NotSpecified,
        sourcedir='src',
        builddir='build',
        )

    def update_argparser(self, parser):
        parser.add_argument('--rc', help="path to run control file")
        parser.add_argument('--plugins', nargs="+", help="plugins to include")
        parser.add_argument('--debug', action='store_true', 
                            help='build in debugging mode')
        parser.add_argument('-v', '--verbose', action='store_true', dest='verbose',
                            help="print more output")
        parser.add_argument('--dumpdesc', action='store_true', dest='dumpdesc',
                            help="print description cache")
        parser.add_argument('--package', action='store', dest='package',
                            help="package name")
        parser.add_argument('--packagedir', action='store', dest='packagedir',
                            help="path to package directory")
        parser.add_argument('--sourcedir', action='store', dest='sourcedir',
                            help="path to source directory")
        parser.add_argument('--builddir', action='store', dest='builddir',
                            help="path to build directory")

    def setup(self, rc):
        if rc.package is NotSpecified:
            msg = "no package name given; please add 'package' to {0}"
            sys.exit(msg.format(rc.rc))
        if rc.packagedir is NotSpecified:
            rc.packagedir = rc.package.replace('.', os.path.sep)
        if not os.path.isdir(rc.packagedir):
            os.makedirs(rc.packagedir)
        if not os.path.isdir(rc.sourcedir):
            os.makedirs(rc.sourcedir)
        if not os.path.isdir(rc.builddir):
            os.makedirs(rc.builddir)
        writenewonly("", os.path.join(rc.packagedir, '__init__.py'), rc.verbose)
        writenewonly("", os.path.join(rc.packagedir, '__init__.pxd'), rc.verbose)

    def execute(self, rc):
        rc._cache = DescriptionCache(cachefile=os.path.join(rc.builddir, 'desc.cache'))
        if rc.dumpdesc:
            print(str(rc._cache))
            sys.exit()

    def report_debug(self, rc):
        msg = 'Current descripton cache contents:\n\n{0}\n\n'
        msg = msg.format(str(rc._cache))
        return msg

