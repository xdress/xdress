package = 'cppproj'
sourcedir = 'src'
packagedir = 'cppproj'

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
    ('func4', 'pybasics'),
    (('findmin', 'int32', 'float32',), 'basics'), 
    (('findmin', 'float64', 'float32',), 'basics'), 
    {'srcname': ('findmin', 'int', 'int',), 
     'tarname': ('regmin', 'int', 'int',), 
     'srcfile': 'basics'}, 
    {'srcname': ('findmin', 'bool', 'bool',), 
     'tarname': 'sillyBoolMin', 
     'srcfile': 'basics'}, 
    (('lessthan', 'int32', 3,), 'basics'),
    ('*', 'discovery'),
    ]

classes = [
    ('struct0', 'basics', 'pybasics', 'My_Struct_0'),
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
