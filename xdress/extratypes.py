"""The extra types plugin for XDress.

This module is available as an xdress plugin by the name ``xdress.extratypes``.

:author: Anthony Scopatz <scopatz@gmail.com>

Extra Types Plugin API
======================
"""
from __future__ import print_function
import os
import io
import sys

from .utils import RunControl, NotSpecified, newoverwrite
from .plugins import Plugin

if sys.version_info[0] >= 3:
    basestring = str

class XDressPlugin(Plugin):
    """This class provides extra type functionality for xdress."""

    requires = ('xdress.base',)

    defaultrc = RunControl(
        extra_types='xdress_extra_types',
        make_extra_types=True,
        )

    rcdocs = {
        "extra_types": "Module and header file name for xdress extra types.",
        "make_extra_types": "Flag to enable / disable making the extra types module",
        }

    def update_argparser(self, parser):
        parser.add_argument('--extra-types', action='store', dest='extra_types', 
                            help=self.rcdocs["extra_types"])
        parser.add_argument('--make-extra-types', action='store_true',
                            dest='make_extra_types', help="make extra types wrapper")
        parser.add_argument('--no-make-extra-types', action='store_false',
                      dest='make_extra_types', help="don't make extra types wrapper")

    def setup(self, rc):
        rc.ts.extra_types = rc.extra_types

    def execute(self, rc):
        if not rc.make_extra_types:
            return
        print("extratypes: generating extra type header & source files for xdress")
        d = os.path.split(__file__)[0]
        srcs = [os.path.join(d, 'xdress_extra_types.h'),
                os.path.join(d, 'xdress_extra_types.pxd'),
                os.path.join(d, 'xdress_extra_types.pyx')]
        tars = [os.path.join(rc.packagedir, rc.extra_types + '.h'),
                os.path.join(rc.packagedir, rc.extra_types + '.pxd'),
                os.path.join(rc.packagedir, rc.extra_types + '.pyx')]
        for src, tar in zip(srcs, tars):
            with io.open(src, 'r') as f:
                s = f.read()
                s = s.format(extra_types=rc.extra_types)
                newoverwrite(s, tar, rc.verbose)
