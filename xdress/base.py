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
    DEFAULT_RC_FILE, DEFAULT_PLUGINS, nyansep
from .plugins import Plugin
from .typesystem import TypeSystem

if sys.version_info[0] >= 3:
    basestring = str

class XDressPlugin(Plugin):
    """This class provides base functionality for xdress itself."""

    defaultrc = RunControl(
        rc=DEFAULT_RC_FILE,
        plugins=DEFAULT_PLUGINS,
        debug=False,
        ts=TypeSystem(),
        verbose=False,
        dumpdesc=False,
        package=NotSpecified,
        packagedir=NotSpecified,
        sourcedir='src',
        builddir='build',
        bash_completion=True,
        )

    # Sweet hack because ts.update() returns None
    rcupdaters = {'ts': (lambda old, new: old.update(new) or old)}

    rcdocs = {
        'rc': "Path to run control file",
        'plugins': "Plugins to include",
        'debug': 'Build in debugging mode', 
        'ts': "The xdress type system.",
        'verbose': "Print more output.",
        'dumpdesc': "Print the description cache",
        'package': "The Python package name for the generated wrappers", 
        'packagedir': "Path to package directory, same as 'package' if not specified",
        'sourcedir': "Path to source directory",
        'builddir': "Path to build directory",
        'bash_completion': ("Flag for enabling / disabling BASH completion. "
                            "This is only relevant when using argcomplete."),
        }

    def update_argparser(self, parser):
        parser.add_argument('--rc', help=self.rcdocs['rc'])
        parser.add_argument('--plugins', nargs="+", help=self.rcdocs["plugins"])
        parser.add_argument('--debug', action='store_true', 
                            help=self.rcdocs["debug"])
        parser.add_argument('-v', '--verbose', action='store_true', dest='verbose',
                            help=self.rcdocs["verbose"])
        parser.add_argument('--dumpdesc', action='store_true', dest='dumpdesc',
                            help=self.rcdocs["dumpdesc"])
        parser.add_argument('--package', action='store', dest='package',
                            help=self.rcdocs["package"])
        parser.add_argument('--packagedir', action='store', dest='packagedir',
                            help=self.rcdocs["packagedir"])
        parser.add_argument('--sourcedir', action='store', dest='sourcedir',
                            help=self.rcdocs["sourcedir"])
        parser.add_argument('--builddir', action='store', dest='builddir',
                            help=self.rcdocs["builddir"])
        parser.add_argument('--bash-completion', action='store_true',
                            help="enable bash completion", dest="bash_completion")
        parser.add_argument('--no-bash-completion', action='store_false',
                            help="disable bash completion", dest="bash_completion")

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
        rc._cache = DescriptionCache(cachefile=os.path.join(rc.builddir, 'desc.cache'))

        if rc.dumpdesc:
            print(str(rc._cache))
            sys.exit()

    def report_debug(self, rc):
        msg = 'Current descripton cache contents:\n\n{0}\n\n'
        msg = msg.format(str(rc._cache))
        msg += nyansep + "\n\n"
        msg += "Current type system contents:\n\n" + str(rc.ts) + "\n\n"
        return msg

