from __future__ import print_function
import os
from xdress import autoall
from xdress.astparsers import PARSERS_AVAILABLE
from xdress.utils import parse_global_rc
from tools import unit, assert_equal_or_diff

@unit
def test_autoall():
    rc = parse_global_rc()
    clang_includes = rc.clang_includes if 'clang_includes' in rc else None
    exp_var = ['Choice']
    exp_fun = ['foo']
    exp_cls = ['Blah']
    filename = os.path.join(os.path.dirname(__file__), 'all.h')
    def check_all(parser):
        obs_var,obs_fun,obs_cls = autoall.findall(filename, parsers=parser, clang_includes=clang_includes)
        assert_equal_or_diff(obs_var, exp_var)
        assert_equal_or_diff(obs_fun, exp_fun)
        assert_equal_or_diff(obs_cls, exp_cls)
    for parser in 'gccxml','clang':
        if parser in PARSERS_AVAILABLE:
            yield check_all, parser

if __name__ == '__main__':
    import nose
    nose.runmodule()
