from __future__ import print_function

import os
import sys
import subprocess

from nose.tools import assert_true, assert_equal

from xdress.astparsers import PARSERS_AVAILABLE

if sys.version_info[0] >= 3:
    basestring = str

PROJDIR = os.path.abspath("cppproj")

def check_cmd(args):
    if not isinstance(args, basestring):
        args = " ".join(args)
    print("TESTING: running command in {0}:\n\n{1}\n\n".format(PROJDIR, args))
    rtn = subprocess.call(args, shell=True, cwd=PROJDIR, stderr=subprocess.STDOUT)
    assert_equal(rtn, 0)
    return rtn

# Because we want to guarentee build and test order, we can only have one 
# master test function which generates the individual tests.

def test_all():
    parsers = ['gccxml', 'clang']
    cases = [{'parsers': p} for p in parsers]

    cwd = os.getcwd()
    base = os.path.dirname(cwd)
    pyexec = sys.executable
    xdexec = os.path.join(base, 'scripts', 'xdress')
    instdir = os.path.join(PROJDIR, 'install')

    cmds = [
        ['PYTHONPATH="{0}"'.format(base), pyexec, xdexec, '--debug'],
        [pyexec, 'setup.py', 'install', '--prefix="{0}"'.format(instdir), '--', '--'],
        ]

    for case in cases:
        parser = case['parsers']
        if not PARSERS_AVAILABLE[parser]:
            continue
        rtn = 1
        for cmd in cmds:
            rtn = yield check_cmd, cmd
            if rtn != 0:
                break  # don't execute further commands
        if rtn != 0:
            break  # don't try further cases
