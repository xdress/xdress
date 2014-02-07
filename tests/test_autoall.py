from __future__ import print_function
import os
from xdress import autoall
from xdress.astparsers import PARSERS_AVAILABLE
from xdress.utils import parse_global_rc
from tools import unit, assert_equal_or_diff, skip_then_continue, cleanfs

@unit
def test_autoall():
    rc = parse_global_rc()
    clang_includes = rc.clang_includes if 'clang_includes' in rc else ()
    exp_var = ['Choice']
    exp_fun = ['foo']
    exp_cls = ['Blah']
    testdir = os.path.dirname(__file__)
    filename = os.path.join(testdir, 'all.h')
    buildbase = os.path.join(testdir, 'build')
    def check_all(parser):
        obs_var, obs_fun, obs_cls = autoall.findall(filename, parsers=parser, 
                                                    builddir=buildbase + '-' + parser,
                                                    clang_includes=clang_includes)
        assert_equal_or_diff(obs_var, exp_var)
        assert_equal_or_diff(obs_fun, exp_fun)
        assert_equal_or_diff(obs_cls, exp_cls)
    for parser in 'gccxml', 'clang':
        cleanfs(buildbase + '-' + parser)
        if PARSERS_AVAILABLE[parser]:
            yield check_all, parser
        else:
            yield skip_then_continue, parser + ' unavailable'

if __name__ == '__main__':
    import nose
    nose.runmodule()
