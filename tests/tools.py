from __future__ import print_function

import os
import sys
import shutil
import subprocess
import tempfile

from nose.tools import assert_true, assert_equal
from nose.plugins.attrib import attr

if sys.version_info[0] >= 3:
    basestring = str

unit = attr('unit')
integration = attr('integration')

def cleanfs(paths):
    for p in paths:
        p = os.path.join(*p)
        if os.path.isfile(p):
            os.remove(p)
        elif os.path.isdir(p):
            shutil.rmtree(p)

def check_cmd(args, cwd, holdsrtn):
    if not isinstance(args, basestring):
        args = " ".join(args)
    print("TESTING: running command in {0}:\n\n{1}\n".format(cwd, args))
    f = tempfile.NamedTemporaryFile()
    rtn = subprocess.call(args, shell=True, cwd=cwd, stdout=f, stderr=f)
    if rtn != 0:
        f.seek(0)
        print("STDOUT + STDERR:\n\n" + f.read())
    f.close()
    holdsrtn[0] = rtn
    assert_equal(rtn, 0)

