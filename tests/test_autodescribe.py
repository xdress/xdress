from __future__ import print_function
import os

from xdress.typesystem import TypeSystem
from xdress import cythongen as cg
from xdress import autodescribe as ad
from xdress.astparsers import PARSERS_AVAILABLE
from xdress.utils import parse_global_rc, Arg

from tools import unit, assert_equal_or_diff, skip_then_continue, cleanfs

from numpy.testing import dec

ts = TypeSystem()

if not os.path.isdir('build'):
    os.mkdir('build')

base = ('Base', 'int32', 7, 0)
def exp_base_desc(parser):
    # TODO: The results depend on the parser since gccxml misses stuff
    bad = parser == 'gccxml'
    return {'name': base,
            'namespace': 'xdress',
            'parents': [],
            'construct': 'struct',
            'attrs': {} if bad else {'field': 'int32'},
            'methods': {(( 'Base', (Arg.TYPE, 'int32'), (Arg.LIT, 7)),): None,
                        (('~Base', (Arg.TYPE, 'int32'), (Arg.LIT, 7)),): None,
                        ('base', ('a', 'int32', (Arg.LIT, 1))): 'void'},
            'type': base,
            }

exp_point_desc = {
    'name': ('Point', True, 0),
    'namespace': 'xdress',
    'parents': [],
    'construct': 'class',
    'attrs': {},
    'methods': {(( 'Point', (Arg.LIT, True)),): None,
                (('~Point', (Arg.LIT, True)),): None},
    'type': ('Point', True, 0)}

exp_default_desc = {
    'name': 'Default',
    'type': 'Default',
    'namespace': 'xdress',
    'parents': [base],
    'construct': 'struct',
    'attrs': {},
    'methods': {( 'Default',): None,
                ('~Default',): None}}

exp_nodefault_base_desc = {
    'name': 'NoDefaultBase',
    'type': 'NoDefaultBase',
    'namespace': 'xdress',
    'parents': [],
    'construct': 'struct',
    'attrs': {},
    'methods': {( 'NoDefaultBase', ('i', 'int32')): None,
                ('~NoDefaultBase',): None}}

exp_nodefault_desc = {
    'name': 'NoDefault',
    'type': 'NoDefault',
    'namespace': 'xdress',
    'parents': ['NoDefaultBase'],
    'construct': 'struct',
    'attrs': {},
    'methods': {( 'NoDefault', ('i', 'int32')): None,
                ('~NoDefault',): None}}

choices = ('enum', 'Choices', (('CA', '0'), ('CB', '17')))
exp_choices_desc = {
    'name': 'Choices',
    'namespace': 'xdress',
    'type': choices}

exp_toaster_desc = {
    'name': 'Toaster',
    'namespace': 'xdress',
#    'parents': [base],
    'parents': [('Base', 'int32', 7, 0)],
    'construct': 'class',
    'attrs': {
        'nslices': 'uint32',
        'toastiness': 'str',
        'rate': 'float32',
        'fp': ('function_pointer', (('_0', 'float32'),), 'int32'),
        'vec': ('vector', (Arg.TYPE, 'char'), 0),
        },
    'methods': {
        ('Toaster', ('slices', 'int32', (Arg.LIT, 7)), 
                    ('flag', 'bool', (Arg.LIT, False))): None,
        ('Toaster', ('d', 'float64'), ('arg', 'str', (Arg.LIT, '\n'))): None,
        ('~Toaster',): None,
        ('make_choice', ('a', choices, (Arg.VAR, 'CA')), 
                        ('b', choices, (Arg.VAR, 'CB'))): 'void',
        ('make_toast', ('when', 'str'), ('nslices', 'uint32', (Arg.LIT, 1)), 
                       ('dub', 'float64', (Arg.LIT, 3e-8))): 'int32',
        ('templates', ('strange', ('Base', (Arg.TYPE, 'int32'), (Arg.LIT, 3), 0))): 
                                  ('Base', (Arg.TYPE, 'float32'), (Arg.LIT, 0), 0),
        ('const_', ('c', ('int32', 'const'))): ('int32', 'const'),
        ('pointers', ('a', ('int32', '*')), ('b', (('int32', 'const'), '*')),
                     ('c', (('int32', '*'), 'const')),
                     ('d', ((('int32', 'const'), '*'), 'const'))): ('int32', '*'),
        ('reference', ('a', ('int32', '&')), ('b', (('int32', 'const'), '&'))): ('int32', '&'),
        },
    'type': 'Toaster',
    }

exp_simple_desc = {
    'name': 'simple',
    'namespace': 'xdress',
    'signatures': {('simple', ('s', 'float32')): 'int32'}}

exp_twice_desc = {
    'name': 'twice',
    'namespace': 'xdress',
    'signatures': {('twice', ('x', 'int32')): 'void'}}

exp_conflict_desc = {
    'name': 'conflict',
    'namespace': 'xdress',
    'signatures': {('conflict', ('good', 'int32')): 'void'}}

def exp_lasso_desc(n):
    lasso_name = ('lasso', n, 'int32', 'float32')
    return {'name': lasso_name,
            'namespace': 'xdress',
            'signatures': {(('lasso', (Arg.LIT, n), (Arg.TYPE, 'int32'), 
                                      (Arg.TYPE, 'float32')), ('a', 'int32'), 
                                        ('b', (('float32', 'const'), '&'))): 'int32'}}

exp_merge_desc = {
    'name': 'Toaster',
    'namespace': 'xdress',
    'parents': [base],
    'construct': 'class',
    'attrs': {
        'nslices': 'uint32',
        'toastiness': 'str',
        'rate': 'float32',
        },
    'methods': {
        ('Toaster', ('slices', 'int32', 7)): None,
        ('~Toaster',): None,
        ('make_toast', ('when', 'str'), ('nslices', 'uint32', (Arg.LIT, 1))): 'int32',
        },
    'type': 'Toaster',
    }

meta_merge_desc = {
    'name': {'srcname': 'Toaster', 'tarname': 'Toaster'},
    'header_filename': 'toaster.h',
    'srcpxd_filename': 'cpp_toaster.pxd',
    'docstrings': {
        'module': "I am the Toaster lib! Hear me sizzle!", 
        'class': "I am the Toaster! FORKS DO NOT GO IN ME!",
        'attrs': {
            'toastiness': "white as snow or black as hell?", 
            'rate': "The rate at which the toaster can process slices.", 
            },
        'methods': {
            'make_toast': "I'll make you some toast you can't refuse...", 
            },
        },
    }

full_merge_desc = {
    'name': {'srcname': 'Toaster', 'tarname': 'Toaster'},
    'construct': 'class',
    'header_filename': 'toaster.h',
    'srcpxd_filename': 'cpp_toaster.pxd',
    'namespace': 'xdress',
    'docstrings': {
        'module': "I am the Toaster lib! Hear me sizzle!", 
        'class': "I am the Toaster! FORKS DO NOT GO IN ME!",
        'attrs': {
            'toastiness': "white as snow or black as hell?", 
            'rate': "The rate at which the toaster can process slices.", 
            },
        'methods': {
            'make_toast': "I'll make you some toast you can't refuse...", 
            },
        },
    'parents': [base],
    'attrs': {
        'nslices': 'uint32',
        'toastiness': 'str',
        'rate': 'float32',
        },
    'methods': {
        ('Toaster', ('slices', 'int32', 7)): None,
        ('~Toaster',): None,
        ('make_toast', ('when', 'str'), ('nslices', 'uint32', (Arg.LIT, 1))): 'int32',
        },
    'type': 'Toaster',
    }

@unit
def test_describe_cpp():
    rc = parse_global_rc()
    clang_includes = rc.clang_includes if 'clang_includes' in rc else ()
    testdir = os.path.dirname(__file__)
    fname = os.path.join(testdir, 'toaster.h')
    buildbase = os.path.join(testdir, 'build')
    ts.register_class('Base', ('T', 'i'), cpp_type='Base')
    ts.register_class('Point', ('T',), cpp_type='Point')
    ts.register_classname('Toaster', 'toaster', 'toaster', 'cpp_toaster')
    for name in 'NoDefaultBase', 'NoDefault', 'Default':
        ts.register_class(name, cpp_type=name)
    def check(parser):
        goals = (('class', ('Base', 'int32', 7, 0), exp_base_desc(parser)),
                 ('class', ('Point', True, 0), exp_point_desc),
                 ('class', 'Toaster', exp_toaster_desc),
                 ('func', 'simple', exp_simple_desc),
                 # Verify that we pick up parameter names from definitions
                 ('func', 'twice', exp_twice_desc), 
                 # Verify that the first parameter name declaration wins
                 ('func', 'conflict', exp_conflict_desc), 
                 ('func', ('lasso', 17, 'int32', 'float32'), exp_lasso_desc(17)),
                 ('func', ('lasso', 18, 'int32', 'float32'), exp_lasso_desc(18)),
                 ('class', 'Default', exp_default_desc),
                 ('class', 'NoDefaultBase', exp_nodefault_base_desc),
                 ('class', 'NoDefault', exp_nodefault_desc),
                 ('var', 'Choices', exp_choices_desc))
        for kind, name, exp in goals:
            obs = ad.describe(fname, name=name, kind=kind, parsers=parser, 
                              builddir=buildbase + '-' + parser, verbose=False, 
                              ts=ts, clang_includes=clang_includes)
            assert_equal_or_diff(obs, exp)
    #for parser in 'gccxml', 'clang':
    for parser in 'gccxml',:
        cleanfs(buildbase + '-' + parser)
        if PARSERS_AVAILABLE[parser]:
            yield check, parser
        else:
            yield skip_then_continue, parser + ' unavailable'

@unit
def test_merge_descriptions():
    obs = ad.merge_descriptions([exp_merge_desc, meta_merge_desc])
    exp = full_merge_desc
    assert_equal_or_diff(obs, exp)

@dec.skipif(ad.pycparser is None)
@unit
def test_pycparser_describe_device_measure():
    obs = ad.pycparser_describe('device.c', 'Device_measure', 'func', ts=ts)
    exp = {#'name': {'srcname': 'Device_measure', 'tarname': 'Device_measure'},
           'name': 'Device_measure',
           'namespace': None,
           'signatures': {
            ('Device_measure', ('_0', ('uint32', '*'))): ('enum', 
                                    'ErrorStatusTag', (('ERROR_OK', 0), 
                                                       ('ERROR_FAILED_INIT', 1))),
            ('Device_measure', ('aiValue', ('uint32', '*'))): ('enum', 
                                    'ErrorStatusTag', (('ERROR_OK', 0), 
                                                       ('ERROR_FAILED_INIT', 1))),
            ('Device_measure', ('deviceNumber', 'uchar'), 
                               ('aiValue', ('uint32', '*'))): ('enum', 
                                    'ErrorStatusTag', (('ERROR_OK', 0), 
                                                       ('ERROR_FAILED_INIT', 1))),
            ('Device_measure', ('_0', 'uchar'), 
                               ('_1', ('uint32', '*'))): ('enum', 
                                    'ErrorStatusTag', (('ERROR_OK', 0), 
                                                       ('ERROR_FAILED_INIT', 1))),
            }
           }
    assert_equal_or_diff(obs, exp)

@dec.skipif(ad.pycparser is None)
@unit
def test_pycparser_describe_device_init():
    obs = ad.pycparser_describe('device.c', 'Device_Init', 'func', ts=ts)
    exp = {#'name': {'srcname': 'Device_Init', 'tarname': 'Device_Init'},
           'name': 'Device_Init',
           'namespace': None,
           'signatures': {
            ('Device_Init', ('_0', ('DeviceParamTag', '*'))): ('enum', 
                                    'ErrorStatusTag', (('ERROR_OK', 0), 
                                                       ('ERROR_FAILED_INIT', 1))),
            ('Device_Init', ('param', ('DeviceParamTag', '*'))): ('enum', 
                                    'ErrorStatusTag', (('ERROR_OK', 0), 
                                                       ('ERROR_FAILED_INIT', 1))),
            }
           }
    assert_equal_or_diff(obs, exp)

@dec.skipif(ad.pycparser is None)
@unit
def test_pycparser_describe_device_descriptor_tag():
    ts.register_class('DeviceDescriptorTag')
    obs = ad.pycparser_describe('device.c', 'DeviceDescriptorTag', 'class', ts=ts)
    exp = {#'name': {'srcname': 'DeviceDescriptorTag', 'tarname': 'DeviceDescriptorTag'},
           'name': 'DeviceDescriptorTag',
           'type': 'DeviceDescriptorTag',
           'namespace': None,
           'construct': 'struct',
           'parents': [],
           'attrs': {
            'deviceNumber': 'uchar',
            'deviceMeasurement': ('function_pointer', (('_0', ('uint32', '*')),), 
                                    ('enum', 'ErrorStatusTag', (
                                        ('ERROR_OK', 0),
                                        ('ERROR_FAILED_INIT', 1)))),
            },
           'methods': {},
           }
    assert_equal_or_diff(obs, exp)

if __name__ == '__main__':
    import nose
    nose.runmodule()
