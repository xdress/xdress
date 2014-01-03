from __future__ import print_function
import os

from xdress.typesystem import TypeSystem
from xdress import cythongen as cg
from xdress import autodescribe as ad
from xdress.astparsers import PARSERS_AVAILABLE
from xdress.utils import parse_global_rc

from tools import unit, assert_equal_or_diff, skip_then_continue

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
            'methods': {(( 'Base', 'int32', 7),): None,
                        (('~Base', 'int32', 7),): None,
                        ('base', ('a', 'int32', 1)): 'void'},
            'type': base,
            }

exp_point_desc = {
    'name': ('Point', True, 0),
    'namespace': 'xdress',
    'parents': [],
    'construct': 'class',
    'attrs': {},
    'methods': {(( 'Point', True),): None,
                (('~Point', True),): None},
    'type': ('Point', True, 0)}

exp_toaster_desc = {
    'name': 'Toaster',
    'namespace': 'xdress',
    'parents': [base],
    'construct': 'class',
    'attrs': {
        'nslices': 'uint32',
        'toastiness': 'str',
        'rate': 'float32',
        'fp': ('function_pointer', (('_0', 'float32'),), 'int32'),
        'vec': ('vector', 'char', 0),
        },
    'methods': {
        ('Toaster', ('slices', 'int32', 7), ('flag', 'bool', False)): None,
        ('Toaster', ('d', 'float64'), ('arg', 'str', '\n')): None,
        ('~Toaster',): None,
        ('make_toast', ('when', 'str'), ('nslices', 'uint32', 1), ('dub', 'float64', 3e-8)): 'int32',
        ('templates', ('strange', ('Base', 'int32', 3, 0))): ('Base', 'float32', 0, 0),
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
    lasso_name = ('lasso',n,'int32','float32')
    return {'name': lasso_name,
            'namespace': 'xdress',
            'signatures': {(lasso_name, ('a', 'int32'), ('b', (('float32', 'const'), '&'))): 'int32'}}

exp_choices_desc = {
    'name': 'Choices',
    'namespace': 'xdress',
    'type': ('enum', 'Choices', (('CA', '0'), ('CB', '17')))}

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
        ('make_toast', ('when', 'str'), ('nslices', 'uint32', 1)): 'int32',
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
        ('make_toast', ('when', 'str'), ('nslices', 'uint32', 1)): 'int32',
        },
    'type': 'Toaster',
    }

@unit
def test_describe_cpp():
    rc = parse_global_rc()
    clang_includes = rc.clang_includes if 'clang_includes' in rc else None
    fname = os.path.join(os.path.split(__file__)[0], 'toaster.h')
    ts.register_class('Base', ('T', 'i'), cpp_type='Base')
    ts.register_classname('Toaster', 'toaster', 'toaster', 'cpp_toaster')
    def check(parser):
        goals = (('class',('Base','int32',7,0),exp_base_desc(parser)),
                 ('class',('Point',True,0),exp_point_desc),
                 ('class','Toaster',exp_toaster_desc),
                 ('func','simple',exp_simple_desc),
                 ('func','twice',exp_twice_desc), # Verify that we pick up parameter names from definitions
                 ('func','conflict',exp_conflict_desc), # Verify that the first parameter name declaration wins
                 ('func',('lasso',17,'int32','float32'),exp_lasso_desc(17)),
                 ('func',('lasso',18,'int32','float32'),exp_lasso_desc(18)),
                 ('var','Choices',exp_choices_desc))
        for kind,name,exp in goals:
            obs = ad.describe(fname, name=name, kind=kind, parsers=parser, verbose=False, ts=ts,
                              clang_includes=clang_includes)
            assert_equal_or_diff(obs, exp)
    for parser in 'gccxml','clang':
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
