from __future__ import print_function
import pprint
import os
import sys

from xdress.types.system import TypeSystem
from xdress.utils import Arg

from nose.tools import assert_equal, with_setup
from tools import unit

if sys.version_info[0] > 2:
    basestring = str

# default typesystem
ts = TypeSystem()

# setup and teardown new refinement cases
new_refined = {
    'comp_map': ('map', 'nucid', 'float64'),
    ('intrange', ('low', 'int32'), ('high', 'int32')): 'int32',
    ('nucrange', ('low', 'nucid'), ('high', 'nucid')): 'nucid',
    ('range', 'vtype', ('low', 'vtype'), ('high', 'vtype')): 'vtype',
    }
new_template = {
    'range': ('vtype', ('low', 'vtype'), ('high', 'vtype')),
    }

def add_new_refined():
    ts.refined_types.update(new_refined)
    ts.template_types.update(new_template)

def del_new_refined():
    [ts.refined_types.pop(key) for key in new_refined]
    for key in new_template:
        del ts.template_types[key]

new_py2c = {
    'comp_map': ('stlcontainers.dict_to_map_int_dbl({var})', False),
    'intrange': ('intrange({var}, {low}, {high})', False),
    'nucrange': ('nucrange({var}, {low}, {high})', False),
    'range': ('<{vtype}> range({var}, {low}, {high})', False),
    }
add_new_py2c = lambda: ts.cython_py2c_conv.update(new_py2c)
del_new_py2c = lambda: [ts.cython_py2c_conv.pop(key) for key in new_py2c]

#
# Actual tests follow
#

def check_canon(t, exp):
    obs = ts.canon(t)
    pprint.pprint(exp)
    pprint.pprint(obs)
    assert_equal(exp, obs)

@unit
@with_setup(add_new_refined, del_new_refined)
def test_canon():
    cases = (
        ('str', 'str'),
        (('str',), ('str', 0)),
        ('f4', 'float32'),
        ('nucid', ('int32', 'nucid')),
        (('nucid',), (('int32', 'nucid'), 0)),
        (('set', 'complex'), ('set', 'complex128', 0)),
        (('map', 'nucid', 'float'), ('map', ('int32', 'nucid'), 'float64', 0)),
        (('pair', 'nucid', 'float'), ('pair', ('int32', 'nucid'), 'float64', 0)),
        ('comp_map', (('map', ('int32', 'nucid'), 'float64', 0), 'comp_map')),
        (('char', '*'), ('char', '*')),
        (('char', 42), ('char', 42)),
        (('map', 'nucid', ('set', 'nucname')), 
            ('map', ('int32', 'nucid'), ('set', ('str', 'nucname'), 0), 0)),
        (((('vector', 'int32'), 'const'), '&'), 
         ((('vector', 'int32', 0), 'const'), '&')),
        (('intrange', 1, 2), ('int32', ('intrange', 
                                            ('low', 'int32', 1), 
                                            ('high', 'int32', 2)))), 
        (('nucrange', 92000, 93000), (('int32', 'nucid'), 
                                        ('nucrange', 
                                            ('low', ('int32', 'nucid'), 92000), 
                                            ('high', ('int32', 'nucid'), 93000)))), 
        (('range', 'int32', 1, 2), ('int32', 
                                        ('range', 'int32',
                                            ('low', 'int32', 1), 
                                            ('high', 'int32', 2)))), 
        (('range', 'nucid', 92000, 93000), (('int32', 'nucid'), 
                                        ('range', ('int32', 'nucid'),
                                            ('low', ('int32', 'nucid'), 92000), 
                                            ('high', ('int32', 'nucid'), 93000)))), 
        (('function_pointer', (('_0', ('uint32', '*')),), 'int'), 
            (('void', '*'), ('function_pointer', ('arguments', ('list', 
                ('pair', 'str', 'type', 0), 0), (('_0', ('uint32', '*')),)), 
                ('returns', 'type', 'int')))),
    )
    for t, exp in cases:
        yield check_canon, t, exp            # Check that the case works,
        yield check_canon, ts.canon(t), exp  # And that it is actually canonical.


def check_basename(t, exp):
    obs = ts.basename(t)
    pprint.pprint(exp)
    pprint.pprint(obs)
    assert_equal(exp, obs)

@unit
@with_setup(add_new_refined, del_new_refined)
def test_basename():
    cases = (
        ('str', 'str'),
        (('str',), 'str'),
        ('f4', 'float32'),
        ('nucid', 'int32'),
        (('nucid',), 'int32'),
        (('set', 'complex'), 'set',),
        (('map', 'nucid', 'float'), 'map'),
        (('pair', 'nucid', 'float'), 'pair'),
        ('comp_map', 'map'), 
        (('char', '*'), 'char'),
        (('char', 42), 'char'),
        (('map', 'nucid', ('set', 'nucname')), 'map'),
        (('intrange', 1, 2), 'int32'),
        (('nucrange', 92000, 93000), 'int32'),
        (('range', 'int32', 1, 2), 'int32'), 
        (('range', 'nucid', 92000, 93000), 'int32'),
        (('function_pointer', (('_0', ('uint32', '*')),), 'int'), 'void'), 
    )
    for t, exp in cases:
        yield check_basename, t, exp  # Check that the case works,



def check_cython_ctype(t, exp):
    obs = ts.cython_ctype(t)
    assert_equal(exp, obs)

@unit
@with_setup(add_new_refined, del_new_refined)
def test_cython_ctype():
    cases = (
        ('str', 'std_string'),
        (('str',), 'std_string'),
        ('f4', 'float'),
        ('nucid', 'int'),
        (('nucid',), 'int'), 
        (('set', 'complex'), 'cpp_set[xdress_extra_types.complex_t]'),
        (('map', 'nucid', 'float'), 'cpp_map[int, double]'),
        (('pair', 'nucid', 'float'), 'cpp_pair[int, double]'),
        ('comp_map', 'cpp_map[int, double]'),
        (('char', '*'), 'char *'),
        (('char', 42), 'char [42]'),
        (('map', 'nucid', ('set', 'nucname')), 'cpp_map[int, cpp_set[std_string]]'),
        (('pair', 'nucid', ('set', 'nucname')), 'cpp_pair[int, cpp_set[std_string]]'),
        (('intrange', 1, 2), 'int'), 
        (('nucrange', 92000, 93000), 'int'),
        (('range', 'int32', 1, 2), 'int'), 
        (('range', 'nucid', 92000, 93000), 'int'), 
    )
    for t, exp in cases:
        yield check_cython_ctype, t, exp  # Check that the case works,


def check_cython_funcname(name, exp):
    obs = ts.cython_funcname(name)
    assert_equal(exp, obs)

@unit
@with_setup(add_new_refined, del_new_refined)
def test_cython_funcname():
    cases = (
        ('joan', 'joan'),
        (('hoover',), 'hoover'),
        (('brienne', 'complex'), 'brienne_complex'),
        (('mulan', 'int', 'float'), 'mulan_int_double'),
        (('leslie', 3, True), 'leslie_3_True'),
    )
    for t, exp in cases:
        yield check_cython_funcname, t, exp  # Check that the case works,

def check_cython_template_funcname(name, argkinds, exp):
    obs = ts.cython_funcname(name, argkinds=argkinds)
    assert_equal(exp, obs)

@unit
@with_setup(add_new_refined, del_new_refined)
def test_cython_template_funcname():
    cases = (
        (('brienne', 'complex'), [(Arg.TYPE, 'complex')], 'brienne_complex'),
        (('mulan', 'int', 'float'), [(Arg.TYPE, 'int'), (Arg.TYPE, 'float')], 
            'mulan_int_double'),
        (('leslie', 3, True), [(Arg.LIT, 3), (Arg.LIT, True)], 'leslie_3_True'),
    )
    for t, ak, exp in cases:
        yield check_cython_template_funcname, t, ak, exp  # Check that the case works,

def check_cython_cimport_tuples_no_cy(t, exp):
    obs = ts.cython_cimport_tuples(t, inc=set(['c']))
    assert_equal(obs, exp)

@unit
@with_setup(add_new_refined, del_new_refined)
def test_cython_cimport_tuples_no_cy():
    cases = (
        ('str', set([('libcpp.string', 'string', 'std_string')])),
        (('str',), set([('libcpp.string', 'string', 'std_string')])),
        ('f4', set()),
        ('nucid', set([('pyne', 'cpp_nucname')])),
        (('nucid'), set([('pyne', 'cpp_nucname')])), 
        (('set', 'complex'), set([('libcpp.set', 'set', 'cpp_set'), 
                                  ('xdress_extra_types',)])),
        (('map', 'nucid', 'float'), set([('pyne', 'cpp_nucname'), 
                                         ('libcpp.map', 'map', 'cpp_map')])),
        (('pair', 'nucid', 'float'), set([('pyne', 'cpp_nucname'), 
                                          ('libcpp.utility', 'pair', 'cpp_pair')])),
        ('comp_map', set([('pyne', 'cpp_nucname'), ('libcpp.map', 'map', 'cpp_map')])),
        (('char', '*'), set()),
        (('char', 42), set()),
        (('map', 'nucid', ('set', 'nucname')), set([('libcpp.set', 'set', 'cpp_set'),
                                                    ('libcpp.map', 'map', 'cpp_map'),
                                                    ('pyne', 'cpp_nucname'),
                                        ('libcpp.string', 'string', 'std_string')])),
        (('intrange', 1, 2), set()), 
        (('nucrange', 92000, 93000), set([('pyne', 'cpp_nucname')])),
        (('range', 'int32', 1, 2), set()), 
        (('range', 'nucid', 92000, 93000), set([('pyne', 'cpp_nucname')])), 
    )
    for t, exp in cases:
        yield check_cython_cimport_tuples_no_cy, t, exp  # Check that the case works


def check_cython_cimport_tuples_with_cy(t, exp):
    obs = ts.cython_cimport_tuples(t)
    assert_equal(exp, obs)

@unit
@with_setup(add_new_refined, del_new_refined)
def test_cython_cimport_tuples_with_cy():
    cases = (
        ('str', set([('libcpp.string', 'string', 'std_string')])),
        (('str',), set([('libcpp.string', 'string', 'std_string')])),
        ('f4', set()),
        ('nucid', set([('pyne', 'nucname'), ('pyne', 'cpp_nucname')])),
        (('nucid',), set([('pyne', 'nucname'), ('pyne', 'cpp_nucname')])), 
        (('set', 'complex'), set([('libcpp.set', 'set', 'cpp_set'), 
                                  ('stlcontainers',),
                                  ('xdress_extra_types',)])),
        (('map', 'nucid', 'float'), set([('libcpp.map', 'map', 'cpp_map'), 
                                         ('pyne', 'nucname'), 
                                         ('pyne', 'cpp_nucname'),
                                         ('stlcontainers',)])),
        (('pair', 'nucid', 'float'), set([('libcpp.utility', 'pair', 'cpp_pair'), 
                                         ('pyne', 'nucname'), 
                                         ('pyne', 'cpp_nucname'),
                                         ('stlcontainers',)])),
        ('comp_map', set([('libcpp.map', 'map', 'cpp_map'), 
                          ('pyne', 'nucname'), 
                          ('pyne', 'cpp_nucname'),
                          ('stlcontainers',)])),
        (('char', '*'), set()),
        (('char', 42), set()),
        (('map', 'nucid', ('set', 'nucname')), set([('libcpp.set', 'set', 'cpp_set'),
                                                    ('libcpp.map', 'map', 'cpp_map'),
                                                    ('stlcontainers',),
                                                    ('pyne', 'nucname'), 
                                                    ('pyne', 'cpp_nucname'),
                                        ('libcpp.string', 'string', 'std_string')])),
        (('intrange', 1, 2), set()), 
        (('nucrange', 92000, 93000), set([('pyne', 'nucname'), 
                                          ('pyne', 'cpp_nucname')])),
        (('range', 'int32', 1, 2), set()), 
        (('range', 'nucid', 92000, 93000), set([('pyne', 'nucname'), 
                                                ('pyne', 'cpp_nucname')])), 
    )
    for t, exp in cases:
        yield check_cython_cimport_tuples_with_cy, t, exp  # Check that the case works


def check_cython_cimport_lines(t, exp):
    obs = ts.cython_cimport_lines(t)
    assert_equal(exp, obs)

@unit
def test_cython_cimport_lines():
    cases = (
        # type checks
        ('str', set(['from libcpp.string cimport string as std_string',])),
        ('f4', set()),
        # seen checks
        (set([('orly',)]), set(['cimport orly',])),
        (set([('orly','yarly')]), set(['from orly cimport yarly',])),
        (set([('orly','as', 'yarly')]), set(['cimport orly as yarly',])),
        (set([('orly','yarly','nowai')]), set(['from orly cimport yarly as nowai',])),
        (set([('orly',), ('orly','yarly')]), 
            set(['cimport orly', 'from orly cimport yarly'])),
    )
    for t, exp in cases:
        yield check_cython_cimport_lines, t, exp  # Check that the case works,

def check_cython_import_tuples(t, exp):
    obs = ts.cython_import_tuples(t)
    assert_equal(exp, obs)

@unit
@with_setup(add_new_refined, del_new_refined)
def test_cython_import_tuples():
    cases = (
        ('str', set()),
        (('str',), set()),
        ('f4', set()),
        ('nucid', set([('pyne', 'nucname')])),
        (('nucid',), set([('pyne', 'nucname')])), 
        (('set', 'complex'), set([('stlcontainers',), ('collections',)])),
        (('map', 'nucid', 'float'), set([('stlcontainers',), ('collections',),
                                         ('pyne', 'nucname')])),
        (('pair', 'nucid', 'float'), set([('stlcontainers',), ('pyne', 'nucname')])),
        ('comp_map', set([('stlcontainers',), ('pyne', 'nucname'), ('collections',)])),
        (('char', '*'), set()),
        (('char', 42), set()),
        (('map', 'nucid', ('set', 'nucname')), 
            set([('stlcontainers',), ('pyne', 'nucname'), ('collections',)])),
        (('intrange', 1, 2), set()), 
        (('nucrange', 92000, 93000), set([('pyne', 'nucname')])),
        (('range', 'int32', 1, 2), set()), 
        (('range', 'nucid', 92000, 93000), set([('pyne', 'nucname')])), 
    )
    for t, exp in cases:
        yield check_cython_import_tuples, t, exp  # Check that the case works


def check_cython_import_lines(t, exp):
    obs = ts.cython_import_lines(t)
    assert_equal(exp, obs)

@unit
def test_cython_import_lines():
    cases = (
        # type checks
        ('str', set()),
        ('f4', set()),
        # seen checks
        (set([('orly',)]), set(['import orly',])),
        (set([('orly','yarly')]), set(['from orly import yarly',])),
        (set([('orly','as', 'yarly')]), set(['import orly as yarly',])),
        (set([('orly','yarly','nowai')]), set(['from orly import yarly as nowai',])),
        (set([('orly',), ('orly','yarly')]), 
            set(['import orly', 'from orly import yarly'])),
    )
    for t, exp in cases:
        yield check_cython_import_lines, t, exp  # Check that the case works,

def check_cython_cytype(t, exp):
    obs = ts.cython_cytype(t)
    assert_equal(exp, obs)

@unit
@with_setup(add_new_refined, del_new_refined)
def test_cython_cytype():
    cases = (
        ('str', 'char *'),
        (('str',), 'char *'),
        ('f4', 'float'),
        ('nucid', 'int'),
        (('nucid',), 'int'), 
        (('set', 'complex'), 'stlcontainers._SetComplex'),
        (('map', 'nucid', 'float'), 'stlcontainers._MapIntDouble'),
        (('pair', 'nucid', 'float'), 'stlcontainers._PairIntDouble'),
        ('comp_map', 'stlcontainers._MapIntDouble'),
        (('char', '*'), 'char *'),
        (('char', 42), 'char [42]'),
        (('map', 'nucid', ('set', 'nucname')), 'stlcontainers._MapIntSetStr'),
        (('intrange', 1, 2), 'int'), 
        (('nucrange', 92000, 93000), 'int'),
        (('range', 'int32', 1, 2), 'int'), 
        (('range', 'nucid', 92000, 93000), 'int'), 
    )
    for t, exp in cases:
        yield check_cython_cytype, t, exp  # Check that the case works,


def check_cython_pytype(t, exp):
    obs = ts.cython_pytype(t)
    assert_equal(obs, exp)

@unit
@with_setup(add_new_refined, del_new_refined)
def test_cython_pytype():
    cases = (
        ('str', 'str'),
        (('str',), 'str'),
        ('f4', 'float'),
        ('nucid', 'int'),
        (('nucid',), 'int'), 
        (('set', 'complex'), 'stlcontainers.SetComplex'),
        (('map', 'nucid', 'float'), 'stlcontainers.MapIntDouble'),
        (('pair', 'nucid', 'float'), 'stlcontainers.PairIntDouble'),
        ('comp_map', 'stlcontainers.MapIntDouble'),
        (('char', '*'), 'str'),
        (('char', 42), 'str'),
        (('map', 'nucid', ('set', 'nucname')), 'stlcontainers.MapIntSetStr'),
        (('intrange', 1, 2), 'int'), 
        (('nucrange', 92000, 93000), 'int'),
        (('range', 'int32', 1, 2), 'int'), 
        (('range', 'nucid', 92000, 93000), 'int'), 
    )
    for t, exp in cases:
        yield check_cython_pytype, t, exp  # Check that the case works,


def check_cython_c2py(name, t, inst_name, exp):
    #import pprint; pprint.pprint(ts.refined_types)
    obs = ts.cython_c2py(name, t, inst_name=inst_name)
    assert_equal(exp, obs)

@unit
@with_setup(add_new_refined, del_new_refined)
def test_cython_c2py():
    cases = (
        (('llama', 'str', None), (None, None, 'bytes(<char *> llama.c_str()).decode()', False)),
        (('llama', ('str',), None), 
            (None, None, 'bytes(<char *> llama.c_str()).decode()', False)),
        (('llama', 'f4', None), (None, None, 'float(llama)', False)),
        (('llama', 'nucid', None), (None, None, 'nucname.zzaaam(llama)', False)),
        (('llama', ('nucid',), None), (None, None, 'nucname.zzaaam(llama)', False)), 
        (('llama', ('set', 'complex'), 'self._inst'), 
            ('cdef stlcontainers._SetComplex llama_proxy\n', 
            ('if self._llama is None:\n'
             '    llama_proxy = stlcontainers.SetComplex(False, False)\n'
             '    llama_proxy.set_ptr = &self._inst.llama\n'
             '    self._llama = llama_proxy'), 'self._llama', True)),
        (('llama', ('map', 'nucid', 'float'), 'self._inst'), 
            ('cdef stlcontainers._MapIntDouble llama_proxy\n', 
            ('if self._llama is None:\n'
             '    llama_proxy = stlcontainers.MapIntDouble(False, False)\n'
             '    llama_proxy.map_ptr = &self._inst.llama\n'
             '    self._llama = llama_proxy'), 'self._llama', True)),
        (('llama', ('pair', 'nucid', 'float'), 'self._inst'), 
            ('cdef stlcontainers._PairIntDouble llama_proxy\n', 
            ('if self._llama is None:\n'
             '    llama_proxy = stlcontainers.PairIntDouble(False, False)\n'
             '    llama_proxy.pair_ptr = &self._inst.llama\n'
             '    self._llama = llama_proxy'), 'self._llama', True)),
        (('llama', 'comp_map', 'self._inst'), 
            ('cdef stlcontainers._MapIntDouble llama_proxy\n', 
            ('if self._llama is None:\n'
             '    llama_proxy = stlcontainers.MapIntDouble(False, False)\n'
             '    llama_proxy.map_ptr = &self._inst.llama\n'
             '    self._llama = llama_proxy'), 'self._llama', True)),
        (('llama', ('char', '*'), None), (None, None, 'bytes(llama).decode()', False)),
        (('llama', ('char', 42), None), (None, None, 'chr(<int> llama)', False)),
        (('llama', ('map', 'nucid', ('set', 'nucname')), 'self._inst'), 
            ('cdef stlcontainers._MapIntSetStr llama_proxy\n', 
            ('if self._llama is None:\n'
             '    llama_proxy = stlcontainers.MapIntSetStr(False, False)\n'
             '    llama_proxy.map_ptr = &self._inst.llama\n'
             '    self._llama = llama_proxy'), 'self._llama', True)),
        (('llama', ('intrange', 1, 2), None), (None, None, 'int(llama)', False)), 
        (('llama', ('nucrange', 92000, 93000), None), 
            (None, None, 'int(llama)', False)),
        (('llama', ('range', 'int32', 1, 2), None), 
            (None, None, 'int(llama)', False)), 
        (('llama', ('range', 'nucid', 92000, 93000), None), 
            (None, None, 'int(llama)', False)),
    )
    for (name, t, inst_name), exp in cases:
        yield check_cython_c2py, name, t, inst_name, exp  # Check that the case works,


def check_cython_py2c(name, t, inst_name, exp):
    obs = ts.cython_py2c(name, t, inst_name=inst_name)
    assert_equal(exp, obs)

@unit
@with_setup(add_new_refined, del_new_refined)
@with_setup(add_new_py2c, del_new_py2c)
def test_cython_py2c():
    cases = (
        (('frog', 'str', None), ('cdef char * frog_proxy\n', 
            'frog_bytes = frog.encode()', 'std_string(<char *> frog_bytes)')),
        (('frog', ('str',), None), ('cdef char * frog_proxy\n', 
            'frog_bytes = frog.encode()', 'std_string(<char *> frog_bytes)')),
        (('frog', 'f4', None), (None, None, '<float> frog')),
        (('frog', 'nucid', None), (None, None, 'nucname.zzaaam(frog)')),
        (('frog', ('nucid',), None), (None, None, 'nucname.zzaaam(frog)')), 
        (('frog', ('set', 'complex'), 'self._inst'), 
            ('cdef stlcontainers._SetComplex frog_proxy\n', 
            ('frog_proxy = stlcontainers.SetComplex(self._inst.frog, '
             'not isinstance(self._inst.frog, stlcontainers._SetComplex))'), 
             'frog_proxy.set_ptr[0]')),
        (('frog', ('map', 'nucid', 'float'), 'self._inst'), 
            ('cdef stlcontainers._MapIntDouble frog_proxy\n', 
            ('frog_proxy = stlcontainers.MapIntDouble(self._inst.frog, '
             'not isinstance(self._inst.frog, stlcontainers._MapIntDouble))'), 
             'frog_proxy.map_ptr[0]')),
        (('frog', ('pair', 'nucid', 'float'), 'self._inst'), 
            ('cdef stlcontainers._PairIntDouble frog_proxy\n', 
            ('frog_proxy = stlcontainers.PairIntDouble(self._inst.frog, '
             'not isinstance(self._inst.frog, stlcontainers._PairIntDouble))'), 
             'frog_proxy.pair_ptr[0]')),
        (('frog', 'comp_map', 'self._inst'), (None, None, 
            'stlcontainers.dict_to_map_int_dbl(self._inst.frog)')),
        (('frog', ('char', '*'), None), ('cdef char * frog_proxy\n',
            'frog_bytes = frog.encode()', '<char *> frog_bytes')),
        (('frog', ('char', 42), None), ('cdef char [42] frog_proxy\n',
            'frog_bytes = frog.encode()', '(<char *> frog_bytes)[0]')),
        (('frog', ('map', 'nucid', ('set', 'nucname')), 'self._inst'), 
            ('cdef stlcontainers._MapIntSetStr frog_proxy\n', 
            ('frog_proxy = stlcontainers.MapIntSetStr(self._inst.frog, '
             'not isinstance(self._inst.frog, stlcontainers._MapIntSetStr))'), 
             'frog_proxy.map_ptr[0]')),
        (('frog', ('intrange', 1, 2), None), (None, None, 'intrange(frog, 1, 2)')), 
        (('frog', ('nucrange', 92000, 93000), None), 
            (None, None, 'nucrange(nucname.zzaaam(frog), 92000, 93000)')),
        (('frog', ('range', 'int32', 1, 2), None), 
            (None, None, '<int> range(frog, 1, 2)')), 
        (('frog', ('range', 'nucid', 92000, 93000), None), 
            (None, None, '<int> range(nucname.zzaaam(frog), 92000, 93000)')),
    )
    for (name, t, inst_name), exp in cases:
        yield check_cython_py2c, name, t, inst_name, exp  # Check that the case works,

def check_strip_predicates(t, exp):
    obs = ts.strip_predicates(t)
    assert_equal(exp, obs)

@unit
def test_strip_predicates():
    cases = [
        [('vector', 'f8'), ('vector', 'float64', 0)],
        [('vector', 'f8', 'const'), ('vector', 'float64', 0)],
        [('vector', 'f8', '&'), ('vector', 'float64', 0)],
        [(('vector', 'f8', '&'), 'const'), 
         ('vector', 'float64', 0)],
        [(('vector', 'f8', '&'), 0), 
         (('vector', 'float64', 0), 0)],
        [('vector', ('vector', 'f8', '&'), 'const'), 
         ('vector', ('vector', 'float64', '&'), 0)],
        ]
    for t, exp in cases:
        yield check_strip_predicates, t, exp

def check_cpp_type(t, exp):
    obs = ts.cpp_type(t)
    assert_equal(exp, obs)

@unit
@with_setup(add_new_refined, del_new_refined)
def test_cpp_type():
    cases = (
        ('str', 'std::string'),
        (('str',), 'std::string'),
        ('f4', 'float'),
        ('nucid', 'int'),
        (('nucid',), 'int'), 
        (('set', 'complex'), 'std::set< xdress_extra_types.complex_t >'),
        (('map', 'nucid', 'float'), 'std::map< int, double >'),
        (('pair', 'nucid', 'float'), 'std::pair< int, double >'),
        ('comp_map', 'std::map< int, double >'),
        (('char', '*'), 'char *'),
        (('char', 42), 'char [42]'),
        (('map', 'nucid', ('set', 'nucname')), 
            'std::map< int, std::set< std::string > >'),
        (('intrange', 1, 2), 'int'), 
        (('nucrange', 92000, 93000), 'int'),
        (('range', 'int32', 1, 2), 'int'), 
        (('range', 'nucid', 92000, 93000), 'int'), 
    )
    for t, exp in cases:
        yield check_cpp_type, t, exp  # Check that the case works,

def check_cpp_funcname(name, exp):
    obs = ts.cpp_funcname(name)
    assert_equal(exp, obs)

@unit
@with_setup(add_new_refined, del_new_refined)
def test_cpp_funcname():
    cases = (
        ('joan', 'joan'),
        (('hoover',), 'hoover'),
        (('brienne', 'complex'), 'brienne< xdress_extra_types.complex_t >'),
        (('mulan', 'int', 'float'), 'mulan< int, double >'),
        (('leslie', 3, True), 'leslie< 3, true >'),
    )
    for t, exp in cases:
        yield check_cpp_funcname, t, exp  # Check that the case works,

def check_cpp_template_funcname(name, argkinds, exp):
    obs = ts.cpp_funcname(name, argkinds=argkinds)
    assert_equal(exp, obs)

@unit
@with_setup(add_new_refined, del_new_refined)
def test_cpp_template_funcname():
    cases = (
        (('brienne', 'complex'), [(Arg.TYPE, 'complex')], 
            'brienne< xdress_extra_types.complex_t >'),
        (('mulan', 'int', 'float'), [(Arg.TYPE, 'int'), (Arg.TYPE, 'float')], 
            'mulan< int, double >'),
        (('leslie', 3, True), [(Arg.LIT, 3), (Arg.LIT, True)], 'leslie< 3, true >'),
    )
    for t, ak, exp in cases:
        yield check_cpp_template_funcname, t, ak, exp  # Check that the case works,

def check_gccxml_type(t, exp):
    obs = ts.gccxml_type(t)
    assert_equal(exp, obs)

@unit
@with_setup(add_new_refined, del_new_refined)
def test_gccxml_type():
    cases = (
        ('str', 'std::string'),
        (('str',), 'std::string'),
        ('f4', 'float'),
        ('nucid', 'int'),
        (('nucid',), 'int'), 
        (('set', 'complex'), 'std::set<xdress_extra_types.complex_t>'),
        (('map', 'nucid', 'float'), 'std::map<int,double>'),
        (('pair', 'nucid', 'float'), 'std::pair<int,double>'),
        ('comp_map', 'std::map<int,double>'),
        (('char', '*'), 'char *'),
        (('char', 42), 'char [42]'),
        (('map', 'nucid', ('set', 'nucname')), 
            'std::map<int,std::set<std::string> >'),
        (('intrange', 1, 2), 'int'), 
        (('nucrange', 92000, 93000), 'int'),
        (('range', 'int32', 1, 2), 'int'), 
        (('range', 'nucid', 92000, 93000), 'int'), 
    )
    for t, exp in cases:
        yield check_gccxml_type, t, exp  # Check that the case works,

@unit
@with_setup(lambda: None, lambda: os.remove('hoover'))
def test_io():
    filename = 'hoover'
    hoover = TypeSystem().empty()
    hoover.extra_types = "excellent"
    hoover.dump(filename, format='pkl.gz')
    hoover = TypeSystem.load(filename, format='pkl.gz')
