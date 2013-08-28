from __future__ import print_function

import os
import sys
import shutil
import subprocess
import tempfile

from nose.tools import assert_true, assert_equal

from xdress.astparsers import PARSERS_AVAILABLE

if sys.version_info[0] >= 3:
    basestring = str

PROJNAME = "cppproj"
PROJDIR = os.path.abspath(PROJNAME)
INSTDIR = os.path.join(PROJDIR, 'install')

GENERATED_PATHS = [
    [PROJDIR, 'build'],
    [PROJDIR, PROJNAME, 'basics.pxd'],
    [PROJDIR, PROJNAME, 'basics.pyx'],
    [PROJDIR, PROJNAME, 'cpp_basics.pxd'],
    [PROJDIR, PROJNAME, 'cpp_discovery.pxd'],
    [PROJDIR, PROJNAME, 'cpp_pybasics.pxd'],
    [PROJDIR, PROJNAME, 'cppproj_extra_types.pxd'],
    [PROJDIR, PROJNAME, 'cppproj_extra_types.pyx'],
    [PROJDIR, PROJNAME, 'discovery.pxd'],
    [PROJDIR, PROJNAME, 'discovery.pyx'],
    [PROJDIR, PROJNAME, 'pybasics.pxd'],
    [PROJDIR, PROJNAME, 'pybasics.pyx'],
    [PROJDIR, PROJNAME, 'stlc.pxd'],
    [PROJDIR, PROJNAME, 'stlc.pyx'],
    [PROJDIR, PROJNAME, 'tests'],
    [PROJDIR, 'src', 'cppproj_extra_types.h'],
    [INSTDIR],
    ]

def cleanfs():
    for p in GENERATED_PATHS:
        p = os.path.join(*p)
        if os.path.isfile(p):
            os.remove(p)
        elif os.path.isdir(p):
            shutil.rmtree(p)

def check_cmd(args, holdsrtn):
    if not isinstance(args, basestring):
        args = " ".join(args)
    print("TESTING: running command in {0}:\n\n{1}\n".format(PROJDIR, args))
    f = tempfile.NamedTemporaryFile()
    rtn = subprocess.call(args, shell=True, cwd=PROJDIR, stdout=f, stderr=f)
    if rtn != 0:
        f.seek(0)
        print("STDOUT + STDERR:\n\n" + f.read())
    f.close()
    holdsrtn[0] = rtn
    assert_equal(rtn, 0)

# Because we want to guarentee build and test order, we can only have one 
# master test function which generates the individual tests.

def test_all():
    parsers = ['gccxml', 'clang']
    cases = [{'parser': p} for p in parsers]

    cwd = os.getcwd()
    base = os.path.dirname(cwd)
    pyexec = sys.executable
    xdexec = os.path.join(base, 'scripts', 'xdress')
    defaults = {'cwd': cwd, 'base': base, 'pyexec': pyexec, 'xdexec': xdexec, 
                'instdir': INSTDIR}

    commands = (
        'PYTHONPATH="{base}" {pyexec} {xdexec} --debug -p={parser}\n'
        '{pyexec} setup.py install --prefix="{instdir}" -- --\n'
        )

    for case in cases:
        parser = case['parser']
        if not PARSERS_AVAILABLE[parser]:
            continue
        cleanfs()
        rtn = 1
        holdsrtn = [rtn]  # needed because nose does not send() to test generator
        fill = dict(defaults)
        fill.update(case)
        cmds = commands.format(**fill).strip().splitlines()
        for cmd in cmds:
            yield check_cmd, cmd, holdsrtn
            rtn = holdsrtn[0]
            if rtn != 0:
                break  # don't execute further commands
        if rtn != 0:
            break  # don't try further cases
    else:
        cleanfs()
