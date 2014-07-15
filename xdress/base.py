"""The base plugin for XDress.

This module is available as an xdress plugin by the name ``xdress.base``.

:author: Anthony Scopatz <scopatz@gmail.com>

Base Plugin API
===============
"""
from __future__ import print_function
import os
import sys
from warnings import warn

from .utils import RunControl, NotSpecified, writenewonly, DescriptionCache, \
    DEFAULT_RC_FILE, DEFAULT_PLUGINS, nyansep, indent
from .plugins import Plugin
from .types.system import TypeSystem
from .version import report_versions

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
        version=False,
        dumpdesc=False,
        package=NotSpecified,
        packagedir=NotSpecified,
        testdir=NotSpecified,
        sourcedir=NotSpecified,
        builddir='build',
        bash_completion=True,
        dtypes_module='dtypes',
        stlcontainers_module='stlcontainers',
        )

    # Sweet hack because ts.update() returns None
    rcupdaters = {'ts': (lambda old, new: old.update(new) or old)}

    rcdocs = {
        'rc': "Path to run control file",
        'plugins': "Plugins to include",
        'debug': 'Build in debugging mode', 
        'ts': "The xdress type system.",
        'verbose': "Print more output.",
        'version': "Print version information.",
        'dumpdesc': "Print the description cache",
        'package': "The Python package name for the generated wrappers", 
        'packagedir': "Path to package directory, same as 'package' if not specified",
        'testdir': "Path to root directory for tests (tests are placed in root/tests), same as 'package' if not specified",
        'sourcedir': "Path to source directory (deprecated)",
        'builddir': "Path to build directory",
        'bash_completion': ("Flag for enabling / disabling BASH completion. "
                            "This is only relevant when using argcomplete."),
        'dtypes_module': "Module name for numpy dtype wrappers.",
        'stlcontainers_module': ("Module name for C++ standard library "
                                 "container wrappers."),
        }

    def update_argparser(self, parser):
        parser.add_argument('--rc', help=self.rcdocs['rc'])
        parser.add_argument('--plugins', nargs="+", help=self.rcdocs["plugins"])
        parser.add_argument('--debug', action='store_true', 
                            help=self.rcdocs["debug"])
        parser.add_argument('-v', '--verbose', action='store_true', dest='verbose',
                            help=self.rcdocs["verbose"])
        parser.add_argument('--version', action='store_true', dest='version',
                            help=self.rcdocs["version"])
        parser.add_argument('--dumpdesc', action='store_true', dest='dumpdesc',
                            help=self.rcdocs["dumpdesc"])
        parser.add_argument('--package', action='store', dest='package',
                            help=self.rcdocs["package"])
        parser.add_argument('--packagedir', action='store', dest='packagedir',
                            help=self.rcdocs["packagedir"])
        parser.add_argument('--testdir', action='store', dest='testdir',
                            help=self.rcdocs["testdir"])
        parser.add_argument('--sourcedir', action='store', dest='sourcedir',
                            help=self.rcdocs["sourcedir"])
        parser.add_argument('--builddir', action='store', dest='builddir',
                            help=self.rcdocs["builddir"])
        parser.add_argument('--bash-completion', action='store_true',
                            help="enable bash completion", dest="bash_completion")
        parser.add_argument('--no-bash-completion', action='store_false',
                            help="disable bash completion", dest="bash_completion")
        parser.add_argument('--dtypes-module', action='store', dest='dtypes_module', 
                            help=self.rcdocs["dtypes_module"])
        parser.add_argument('--stlcontainers-module', action='store',
                            dest='stlcontainers_module', 
                            help=self.rcdocs["stlcontainers_module"])

    def setup(self, rc):
        if rc.version:
            print(report_versions())
            sys.exit()

        # This should be done ASAP after the ts is set
        rc.ts.dtypes = rc.dtypes_module
        rc.ts.stlcontainers = rc.stlcontainers_module

        if rc.package is NotSpecified:
            msg = "no package name given; please add 'package' to {0}"
            sys.exit(msg.format(rc.rc))
        if rc.packagedir is NotSpecified:
            rc.packagedir = rc.package.replace('.', os.path.sep)
        if rc.testdir is NotSpecified:
            rc.testdir = rc.package.replace('.', os.path.sep)
        if not os.path.isdir(rc.packagedir):
            os.makedirs(rc.packagedir)
        if not os.path.isdir(rc.testdir):
            os.makedirs(rc.testdir)
        if rc.sourcedir is not NotSpecified:
            warn("run control parameter 'sourcedir' has been removed in favor "
                 "of new apiname semantics", DeprecationWarning)
        if not os.path.isdir(rc.builddir):
            os.makedirs(rc.builddir)
        writenewonly("", os.path.join(rc.packagedir, '__init__.py'), rc.verbose)
        writenewonly("", os.path.join(rc.packagedir, '__init__.pxd'), rc.verbose)
        rc._cache = DescriptionCache(cachefile=os.path.join(rc.builddir, 'desc.cache'))

        if rc.dumpdesc:
            print(str(rc._cache))
            sys.exit()

    def report_debug(self, rc):
        msg = 'Version Information:\n\n{0}\n\n'
        msg += nyansep + "\n\n"
        msg += 'Current descripton cache contents:\n\n{1}\n\n'
        msg = msg.format(indent(report_versions()), str(rc._cache))
        msg += nyansep + "\n\n"
        msg += "Current type system contents:\n\n" + str(rc.ts) + "\n\n"
        return msg

