package = 'cproj'
sourcedir = 'src'
packagedir = 'cproj'

plugins = ('xdress.autoall', 'xdress.pep8names', 'xdress.cythongen', 
           'xdress.extratypes')

extra_types = 'cproj_extra_types'  # non-default value

variables = [
    ('PersonID', 'basics', 'pybasics'),
    ('*', 'discovery'),
    ]

functions = [
    ('voided', 'basics'),
    {'srcname': 'func0', 
     'tarname': 'a_better_name',
     'srcfile': 'basics'},
    ('func1', 'basics'),
    ('func2', 'basics'),
    ('func3', 'basics'),
    ('func4', 'basics', 'pybasics'),
    ('call_threenums_op_from_c', 'basics', 'pybasics'),
    ('*', 'discovery'),
    ]

classes = [
#    ('struct0', 'basics', 'pybasics', 'My_Struct_0'),  #FIXME This needs more work
    ('ThreeNums', 'basics', 'pybasics'),
    ('*', 'discovery'),
    ]
