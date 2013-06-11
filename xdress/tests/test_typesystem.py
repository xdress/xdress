from __future__ import print_function
from xdress import typesystem as ts
from xdress.typesystem import MatchAny

from nose.tools import assert_equal, with_setup

# setup and teardown new refinement cases
new_refined = {
    'comp_map': ('map', 'nucid', 'float64'),
    ('intrange', ('low', 'int32'), ('high', 'int32')): 'int32',
    ('nucrange', ('low', 'nucid'), ('high', 'nucid')): 'nucid',
    ('range', 'vtype', ('low', 'vtype'), ('high', 'vtype')): 'vtype',
    }
add_new_refined = lambda: ts.refined_types.update(new_refined)
del_new_refined = lambda: [ts.refined_types.pop(key) for key in new_refined]

new_py2c = {
    'comp_map': ('stlcontainers.dict_to_map_int_dbl({var})', False),
    'intrange': ('intrange({var}, {low}, {high})', False),
    'nucrange': ('nucrange({var}, {low}, {high})', False),
    'range': ('<{vtype}> range({var}, {low}, {high})', False),
    }
add_new_py2c = lambda: ts._cython_py2c_conv.update(new_py2c)
del_new_py2c = lambda: [ts._cython_py2c_conv.pop(key) for key in new_py2c]



def check_canon(t, exp):
    obs = ts.canon(t)
    assert_equal(exp, obs)

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
        ('comp_map', (('map', ('int32', 'nucid'), 'float64', 0), 'comp_map')),
        (('char', '*'), ('char', '*')),
        (('char', 42), ('char', 42)),
        (('map', 'nucid', ('set', 'nucname')), 
            ('map', ('int32', 'nucid'), ('set', ('str', 'nucname'), 0), 0)),
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


def check_cython_ctype(t, exp):
    obs = ts.cython_ctype(t)
    assert_equal(exp, obs)

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
        ('comp_map', 'cpp_map[int, double]'),
        (('char', '*'), 'char *'),
        (('char', 42), 'char [42]'),
        (('map', 'nucid', ('set', 'nucname')), 'cpp_map[int, cpp_set[std_string]]'),
        (('intrange', 1, 2), 'int'), 
        (('nucrange', 92000, 93000), 'int'),
        (('range', 'int32', 1, 2), 'int'), 
        (('range', 'nucid', 92000, 93000), 'int'), 
    )
    for t, exp in cases:
        yield check_cython_ctype, t, exp  # Check that the case works,


def check_cython_cimport_tuples_no_cy(t, exp):
    obs = ts.cython_cimport_tuples(t, inc=set(['c']))
    assert_equal(obs, exp)

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


def check_cython_cimports(t, exp):
    obs = ts.cython_cimports(t)
    assert_equal(exp, obs)

def test_cython_cimports():
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
        yield check_cython_cimports, t, exp  # Check that the case works,

def check_cython_import_tuples(t, exp):
    obs = ts.cython_import_tuples(t)
    assert_equal(exp, obs)

@with_setup(add_new_refined, del_new_refined)
def test_cython_import_tuples():
    cases = (
        ('str', set()),
        (('str',), set()),
        ('f4', set()),
        ('nucid', set([('pyne', 'nucname')])),
        (('nucid',), set([('pyne', 'nucname')])), 
        (('set', 'complex'), set([('stlcontainers',)])),
        (('map', 'nucid', 'float'), set([('stlcontainers',), 
                                         ('pyne', 'nucname')])),
        ('comp_map', set([('stlcontainers',), ('pyne', 'nucname')])),
        (('char', '*'), set()),
        (('char', 42), set()),
        (('map', 'nucid', ('set', 'nucname')), 
            set([('stlcontainers',), ('pyne', 'nucname')])),
        (('intrange', 1, 2), set()), 
        (('nucrange', 92000, 93000), set([('pyne', 'nucname')])),
        (('range', 'int32', 1, 2), set()), 
        (('range', 'nucid', 92000, 93000), set([('pyne', 'nucname')])), 
    )
    for t, exp in cases:
        yield check_cython_import_tuples, t, exp  # Check that the case works


def check_cython_imports(t, exp):
    obs = ts.cython_imports(t)
    assert_equal(exp, obs)

def test_cython_imports():
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
        yield check_cython_imports, t, exp  # Check that the case works,

def check_cython_cytype(t, exp):
    obs = ts.cython_cytype(t)
    assert_equal(exp, obs)

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
        (('llama', 'comp_map', 'self._inst'), 
            ('cdef stlcontainers._MapIntDouble llama_proxy\n', 
            ('if self._llama is None:\n'
             '    llama_proxy = stlcontainers.MapIntDouble(False, False)\n'
             '    llama_proxy.map_ptr = &self._inst.llama\n'
             '    self._llama = llama_proxy'), 'self._llama', True)),
        (('llama', ('char', '*'), None), (None, None, 'chr(<int> llama)', False)),
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
        (('frog', 'comp_map', 'self._inst'), (None, None, 
            'stlcontainers.dict_to_map_int_dbl(self._inst.frog)')),
        (('frog', ('char', '*'), None), ('cdef char * frog_proxy\n',
            'frog_bytes = frog.encode()', '(<char *> frog_bytes)[0]')),
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

p1 = ('float64', MatchAny)

type_matcher_cases = [
    [p1, ('float64', 0), True],
    [p1, ('float64', '*'), True],
    [p1, ('float64', '&'), True],
    [p1, ('float64', 'const'), True],
    [p1, 'float64', False],
    [p1, ('f8', 0), False],
    [p1, (('float64', 'const'), '&'), False],
    ]

def check_typematcher(pattern, t, exp):
    tm = ts.TypeMatcher(pattern)
    obs = tm.matches(t)
    assert_equal(exp, obs)

def test_typematcher():
    for pattern, t, exp in type_matcher_cases:
        yield check_typematcher, pattern, t, exp

def check_matches(pattern, t, exp):
    obs = ts.matches(pattern, t)
    assert_equal(exp, obs)

def test_matches():
    for pattern, t, exp in type_matcher_cases:
        yield check_matches, pattern, t, exp
