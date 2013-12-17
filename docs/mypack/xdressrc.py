from xdress.utils import apiname

package = 'mypack'     # top-level python package name
packagedir = 'mypack'  # loation of the python package

extra_types = 'mypack_extra_types'

stlcontainers = [
    ('vector', 'str'),
    ('set', 'uint'),
    ('map', 'int', 'float'),
    ]

classes = [
    apiname('A', ('src/hoover.h', 'src/hoover.cpp'), incfiles='hoover.h'),
    apiname('B', ('src/hoover.h', 'src/hoover.cpp'), 'hoover_b', incfiles='hoover.h'),
    ]

functions = [apiname('do_nothing_ab', ('src/hoover.h', 'src/hoover.cpp'), 'hoover_b', 
                     incfiles='hoover.h')]

