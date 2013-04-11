package = 'mypack'     # top-level python package name
packagedir = 'mypack'  # loation of the python package
sourcedir = 'src'

stlcontainers = [
    ('vector', 'str'),
    ('set', 'uint'),
    ('map', 'int', 'float'),
    ]

classes = [
        ('A', 'hoover'),
        ('B', 'hoover', 'hoover_b'),
        ]

functions = [('do_nothing_ab', 'hoover', 'hoover_b')]

