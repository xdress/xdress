import os
from xdress.utils import make_apiname

package = 'cppproj'
packagedir = 'cppproj'
includes = ['src']

plugins = ('xdress.autoall', 'xdress.pep8names', 'xdress.cythongen', 'xdress.stlwrap', 
    )

extra_types = 'cppproj_extra_types'  # non-default value

stlcontainers = [
    ('vector', 'float32'),
    ('vector', 'float64'),
    ('vector', 'str'),
    ('vector', 'int32'),
    ('vector', 'complex'),
    ('vector', ('vector', 'int32')),
    ('vector', ('vector', 'float64')),
    ('vector', 'ThreeNums'),
    ('set', 'int'),
    ('set', 'str'),
    ('set', 'uint'),
    ('set', 'char'),
    ('map', 'str', 'str'),
    ('map', 'str', 'int'),
    ('map', 'int', 'str'),
    ('map', 'str', 'uint'),
    ('map', 'uint', 'str'),
    ('map', 'uint', 'uint'),
    ('map', 'str', 'float'),
    ('map', 'int', 'int'),
    ('map', 'int', 'bool'),
    ('map', 'int', 'char'),
    ('map', 'int', 'float'),
    ('map', 'uint', 'float'),
    ('map', 'int', 'complex'),
    ('map', 'int', ('set', 'int')),
    ('map', 'int', ('set', 'str')),
    ('map', 'int', ('set', 'uint')),
    ('map', 'int', ('set', 'char')),
    ('map', 'int', ('vector', 'str')),
    ('map', 'int', ('vector', 'int')),
    ('map', 'int', ('vector', 'uint')),
    ('map', 'int', ('vector', 'char')),
    ('map', 'int', ('vector', 'bool')),
    ('map', 'int', ('vector', 'float')),
    ('map', 'int', ('vector', ('vector', 'float64'))),
    ('map', 'int', ('map', 'int', 'bool')),
    ('map', 'int', ('map', 'int', 'char')),
    ('map', 'int', ('map', 'int', 'float')),
    ('map', 'int', ('map', 'int', ('vector', 'bool'))),
    ('map', 'int', ('map', 'int', ('vector', 'char'))),
    ('map', 'int', ('map', 'int', ('vector', 'float'))),
    ]

stlcontainers_module = 'stlc'

_fromsrcdir = lambda x: os.path.join('src', x)
_inbasics = {'srcfiles': _fromsrcdir('basics.[ch]*'),
             'incfiles': 'basics.h',
             'language': 'c++',
             }
_indiscovery = {'srcfiles': _fromsrcdir('discovery*'),
                'incfiles': 'discovery.h',
                'language': 'c++',
                }

variables = [
    make_apiname('PersonID', tarbase='pybasics', **_inbasics),
    make_apiname('*', **_indiscovery),
    ]

functions = [
    make_apiname('voided', **_inbasics),
    {'srcname': 'func0', 
     'tarname': 'a_better_name',
     'srcfiles': _fromsrcdir('basics*')},
    make_apiname('func1', **_inbasics),
    make_apiname('func2', **_inbasics),
    make_apiname('func3', **_inbasics),
    make_apiname('func4', tarbase='pybasics', **_inbasics),
    make_apiname(('findmin', 'int32', 'float32',), **_inbasics), 
    make_apiname(('findmin', 'float64', 'float32',), **_inbasics), 
    {'srcname': ('findmin', 'int', 'int',), 
     'tarname': ('regmin', 'int', 'int',), 
     'srcfiles': _fromsrcdir('basics*')}, 
    {'srcname': ('findmin', 'bool', 'bool',), 
     'tarname': 'sillyBoolMin', 
     'srcfiles': _fromsrcdir('basics*')}, 
    make_apiname(('lessthan', 'int32', 3,), **_inbasics),
    make_apiname('call_threenums_op_from_c', tarbase='pybasics', **_inbasics),
    make_apiname('*', **_indiscovery),
    ]

classes = [
#    make_apiname('struct0', 'basics', 'pybasics', 'My_Struct_0'),  FIXME This needs more work
    make_apiname('A', **_inbasics),
    make_apiname('B', **_inbasics),
    make_apiname('C', **_inbasics),
    make_apiname(('TClass1', 'int32'), **_inbasics), 
    make_apiname(('TClass1', 'float64'), **_inbasics), 
    {'srcname': ('TClass1', 'float32'), 
     'tarname': 'TC1Floater', 
     'srcfiles': _fromsrcdir('basics*')}, 
    make_apiname(('TClass0', 'int32'), **_inbasics), 
    make_apiname(('TClass0', 'float64'), **_inbasics), 
    {'srcname': ('TClass0', 'bool'), 
     'tarname': ('TC0Bool', 'bool'),
     'srcfiles': _fromsrcdir('basics*')}, 
    make_apiname('Untemplated', **_inbasics), 
    make_apiname('ThreeNums', tarbase='pybasics', **_inbasics),
    make_apiname('*', **_indiscovery),
    ]

del os
del make_apiname

