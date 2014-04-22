from __future__ import print_function

import os
import re
import sys
import imp
import shutil
import unittest
import subprocess
import tempfile
from glob import glob
from contextlib import contextmanager
from pprint import pformat

from nose.tools import assert_true, assert_equal
from nose.plugins.attrib import attr
from nose.plugins.skip import SkipTest

if sys.version_info[0] >= 3:
    basestring = str

unit = attr('unit')
integration = attr('integration')

def cleanfs(paths):
    """Removes the paths from the file system."""
    if isinstance(paths, basestring):
        paths = [paths]
    for path in paths:
        path = os.path.join(*path)
        for p in glob(path):
            if os.path.isfile(p):
                os.remove(p)
                print(' -- removed {0!r}'.format(p))
            elif os.path.isdir(p):
                shutil.rmtree(p)
                print(' -- removed {0!r}'.format(p))

def check_cmd(args, cwd, holdsrtn, suppress=True):
    """Runs a command in a subprocess and verifies that it executed properly.
    """
    if not isinstance(args, basestring):
        args = " ".join(args)
    print("TESTING: running command in {0}:\n\n{1}\n".format(cwd, args))
    f = tempfile.NamedTemporaryFile() if suppress else None
    rtn = subprocess.call(args, shell=True, cwd=cwd, stdout=f, stderr=f)
    if rtn != 0 and suppress:
        f.seek(0)
        print("STDOUT + STDERR:\n")
        print(f.read())
    if suppress:
        f.close()
    holdsrtn[0] = rtn
    assert_equal(rtn, 0)

@contextmanager 
def clean_import(name, paths=None):
    """Imports and returns a module context manager and then removes
    all modules which didn't originally exist when exiting the block.
    Be sure to delete any references to the returned module prior to 
    exiting the context.
    """
    sys.path = paths + sys.path
    origmods = set(sys.modules.keys())
    mod = imp.load_module(name, *imp.find_module(name, paths))
    yield mod
    sys.path = sys.path[len(paths):]
    del mod
    newmods = set(sys.modules.keys()) - origmods
    for newmod in newmods:
        del sys.modules[newmod]

TESTNAME_RE = re.compile('(?:^|[\\b_\\.-])[Tt]est')

def modtests(mod):
    """Finds all of the tests in a module."""
    tests = []
    for name in dir(mod):
        if TESTNAME_RE.match(name) is None:
            continue
        test = getattr(mod, name) 
        if test is unittest.TestCase:
            continue
        tests.append(test)
    return tests
            
def dirtests(d):
    """Finds all of the test files in a directory."""
    files = os.listdir(d)
    filenames = []
    for file in files:
        if not file.endswith('.py'):
            continue
        if TESTNAME_RE.match(file) is None:
            continue
        filenames.append(file[:-3])
    return filenames

def skip_then_continue(msg=""):
    """A simple function to yield such that a test is marked as skipped
    and we may continue on our merry way. A message may be optionally passed
    to this function.
    """
    raise SkipTest(msg)

def show_diff(a,b,key=None):
    """Generated a colored diff between two strings.
    If key is passed, {0} and {1} are substituted with the colors of a and b, respectively."""
    red   = chr(27)+'[1;31m'
    green = chr(27)+'[1;32m'
    blue  = chr(27)+'[1;34m'
    clear = chr(27)+'[00m'
    import difflib
    m = difflib.SequenceMatcher(a=a,b=b,autojunk=0)
    r = []
    if key is not None:
        r.extend((green,key.format(blue+'blue'+green,red+'red'+green)))
    ia,ib = 0,0
    for ja,jb,n in m.get_matching_blocks():
        r.extend((blue, a[ia:ja],
                  red,  b[ib:jb],
                  clear,a[ja:ja+n]))
        ia = ja+n
        ib = jb+n
    return ''.join(r)

def assert_equal_or_diff(obs, exp):
    try:
        assert_equal(obs, exp)
    except:
        key = '\n\n# only expected = {0}, only computed = {1}\n'
        print(show_diff(pformat(exp),pformat(obs),key=key))
        raise

#
# Here there be Hackons!
#

# hack to make sure that we are actually in the tests dir when we start running 
# tests.  This works because this file is imported by many of the other test 
# files.
_fdir = os.path.dirname(__file__)
if os.getcwd() != _fdir:
    os.chdir(_fdir)
del _fdir
