from __future__ import print_function

import os
import sys
import subprocess

from nose.tools import assert_true, assert_equal

from xdress.astparsers import PARSERS_AVAILABLE

PROJDIR = os.path.abspath("cppproj")

def check_cmd(args):
    print("TESTING: Running Command:\n\n{0}\n\n".format(" ".join(args)))
    return subprocess.call(args, cwd=PROJDIR, stderr=subprocess.STDOUT)

# Because we want to guarentee build and test order, we can only have one 
# master test function which generates the individual tests.

def test_all():
    parsers = ['gccxml', 'clang']
    cases = [{'parsers': p} for p in parsers]

    cwd = os.getcwd()
    pyexec = sys.executable
    xdexec = os.path.join(cwd, 'scripts', 'xdress')
    instdir = os.path.join(PROJDIR, 'install')

    cmds = [
        ['PYTHONPATH="{0}"'.format(cwd), pyexec, xdexec, '--debug'],
        [pyexec, 'setup.py', 'install', '--prefix="{0}"'.format(instdir), '--', '--'],
        ]

    for case in cases:
        parser = case['parsers']
        if not PARSERS_AVAILABL[parser]:
            continue
        for cmd in cmds:
            rtn = yield check_cmd, cmd
            assert_equal(rtn, 0)
