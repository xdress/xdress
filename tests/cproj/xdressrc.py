import os
from xdress.utils import apiname

package = 'cproj'
packagedir = 'cproj'
includes = ['src']

plugins = ('xdress.autoall', 'xdress.pep8names', 'xdress.cythongen', 
           'xdress.extratypes')

extra_types = 'cproj_extra_types'  # non-default value

_fromsrcdir = lambda x: os.path.join('src', x)
_inbasics = {'srcfiles': _fromsrcdir('basics*'),
             'incfiles': 'basics.h',
             }
_indiscovery = {'srcfiles': _fromsrcdir('discovery*'),
                'incfiles': 'discovery.h',
                }

variables = [
    apiname('PersonID', tarbase='pybasics', **_inbasics),
    apiname('*', **_indiscovery),
    ]

functions = [
    apiname('voided', **_inbasics),
    {'srcname': 'func0', 
     'tarname': 'a_better_name',
     'srcfiles': _fromsrcdir('basics*'),
     'incfiles': 'basics.h',
     },
    apiname('func1', **_inbasics),
    apiname('func2', **_inbasics),
    apiname('func3', **_inbasics),
    apiname('func4', tarbase='pybasics', **_inbasics),
    apiname('call_threenums_op_from_c', tarbase='pybasics', **_inbasics),
    apiname('*', **_indiscovery),
    ]

classes = [
#    apiname('struct0', _fromsrcdir('basics*'), 'pybasics', 'My_Struct_0'),  #FIXME This needs more work
    apiname('ThreeNums', tarbase='pybasics', **_inbasics),
    apiname('*', **_indiscovery),
    ]

del os
del apiname
