package = 'xdtest'
sourcedir = 'src'
packagedir = 'xdtest'

extra_types = 'xdtest_extra_types'  # non-default value

stlcontainers = [
    ('vector', 'float32'),
    ('vector', 'float64'),
    ('vector', 'str'),
    ('vector', 'int32'),
    ('vector', 'complex'),
    ('vector', ('vector', 'int32')),
    ('vector', ('vector', 'float64')),
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
    # May be bug in Cython?
    #('map', 'int', ('vector', 'complex')),
    ]

stlcontainers_module = 'xdstlc'

variables = [
    ('ErrorStatus', 'device', 'pydevice'),
    ]

functions = [
    (('findmin', 'float64', 'int32'), 'bright'), 
    ('func', 'reprocess'),
    #('afunc', 'device', 'pydevice'),
    ('fillUraniumEnrichmentDefaults', 'enrichment_parameters'),
    ('*', 'device', 'pydevice'),
    ]

classes = [
    ('TwoNums', 'device', 'pydevice'),
    ('DeviceParamTag', 'device', 'pydevice'),
    ('DeviceDescriptorTag', 'device', 'pydevice'),
    ('FCComp', 'fccomp'), 
    ('EnrichmentParameters', 'enrichment_parameters'), 
    ('Enrichment', 'bright_enrichment', 'enrichment'), 
    ('*', 'reprocess'), 
    (('SparseMatrix', 'int32'), 'bright'), 
    (('SparseMatrix', 'float64'), 'bright'), 
    (('sparse_matrix_entry', 'int32'), 'bright'), 
    (('sparse_matrix_entry', 'float64'), 'bright'), 
#    ('*', 'bright'),  # doesn't work yet due to *_t types
    ]
