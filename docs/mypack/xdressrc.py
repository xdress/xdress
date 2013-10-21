package = 'mypack'     # top-level python package name
packagedir = 'mypack'  # loation of the python package

stlcontainers = [
    ('vector', 'str'),
    ('set', 'uint'),
    ('map', 'int', 'float'),
    ]

classes = [
        ('A', ('src/hoover.h', 'src/hoover.cpp')),
        ('B', ('src/hoover.h', 'src/hoover.cpp'), 'hoover_b'),
        ]

functions = [('do_nothing_ab', ('src/hoover.h', 'src/hoover.cpp'), 'hoover_b')]

