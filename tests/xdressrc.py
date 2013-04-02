package = 'xdtest'
sourcedir = 'src'
packagedir = 'xdtest'

extra_types = 'xdtest_extra_types'  # non-default value

stlcontainers = [
    ('vector', 'float32'),
    #('vector', 'float64'),  # doesn't work because double isn't a cython type 
    ('vector', 'str'),
    ('vector', 'int32'),
    #('vector', 'complex'),  # Need to define comparison opperators for complex_t
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
    #('map', 'int', ('vector', ('vector', 'float'))),
    ('map', 'int', ('map', 'int', 'bool')),
    ('map', 'int', ('map', 'int', 'char')),
    ('map', 'int', ('map', 'int', 'float')),
    ('map', 'int', ('map', 'int', ('vector', 'bool'))),
    ('map', 'int', ('map', 'int', ('vector', 'char'))),
    ('map', 'int', ('map', 'int', ('vector', 'float'))),
    # the right way to do this is register new numpy dtypes
    #('map', 'int', ('vector', 'complex')),
    ]

stlcontainers_module = 'xdstlc'

classes = [
    ('FCComp', 'fccomp'), 
    ('EnrichmentParameters', 'enrichment_parameters'), 
    ('Enrichment', 'bright_enrichment', 'enrichment'), 
    ('Reprocess', 'reprocess'), 
    ]

functions = [
    ('func', 'reprocess'),
    ('fillUraniumEnrichmentDefaults', 'enrichment_parameters'),
    ]
