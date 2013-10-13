import os

package = 'cproj'
sourcedir = 'src'
packagedir = 'cproj'

plugins = ('xdress.autoall', 'xdress.pep8names', 'xdress.cythongen', 
           'xdress.extratypes')

extra_types = 'cproj_extra_types'  # non-default value

_fromsrcdir = lambda x: os.path.join('src', x)

variables = [
    ('PersonID', _fromsrcdir('basics*'), 'pybasics'),
    ('*', _fromsrcdir('discovery*')),
    ]

functions = [
    ('voided', _fromsrcdir('basics*')),
    {'srcname': 'func0', 
     'tarname': 'a_better_name',
     'srcfiles': _fromsrcdir('basics*')},
    ('func1', _fromsrcdir('basics*')),
    ('func2', _fromsrcdir('basics*')),
    ('func3', _fromsrcdir('basics*')),
    ('func4', _fromsrcdir('basics*'), 'pybasics'),
    ('call_threenums_op_from_c', _fromsrcdir('basics*'), 'pybasics'),
    ('*', _fromsrcdir('discovery*')),
    ]

classes = [
#    ('struct0', _fromsrcdir('basics*'), 'pybasics', 'My_Struct_0'),  #FIXME This needs more work
    ('ThreeNums', _fromsrcdir('basics*'), 'pybasics'),
    ('*', _fromsrcdir('discovery*')),
    ]
