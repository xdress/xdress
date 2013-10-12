package = 'cproj'
sourcedir = 'src'
packagedir = 'cproj'

plugins = ('xdress.autoall', 'xdress.pep8names', 'xdress.cythongen', 'xdress.stlwrap', 
    )

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
    (('findmin', 'int32', 'float32',), 'basics'), 
    (('findmin', 'float64', 'float32',), 'basics'), 
    {'srcname': ('findmin', 'int', 'int',), 
     'tarname': ('regmin', 'int', 'int',), 
     'srcfile': 'basics'}, 
    {'srcname': ('findmin', 'bool', 'bool',), 
     'tarname': 'sillyBoolMin', 
     'srcfile': 'basics'}, 
    (('lessthan', 'int32', 3,), 'basics'),
    ('call_threenums_op_from_c', 'basics', 'pybasics'),
    ('*', 'discovery'),
    ]

classes = [
#    ('struct0', 'basics', 'pybasics', 'My_Struct_0'),  FIXME This needs more work
    ('A', 'basics'),
    ('B', 'basics'),
    ('C', 'basics'),
    (('TClass1', 'int32'), 'basics'), 
    (('TClass1', 'float64'), 'basics'), 
    {'srcname': ('TClass1', 'float32'), 
     'tarname': 'TC1Floater', 
     'srcfile': 'basics'}, 
    (('TClass0', 'int32'), 'basics'), 
    (('TClass0', 'float64'), 'basics'), 
    {'srcname': ('TClass0', 'bool'), 
     'tarname': ('TC0Bool', 'bool'),
     'srcfile': 'basics'}, 
    ('Untemplated', 'basics'), 
    ('ThreeNums', 'basics', 'pybasics'),
    ('*', 'discovery'),
    ]
