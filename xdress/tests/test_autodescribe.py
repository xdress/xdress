from __future__ import print_function
import os

from xdress import typesystem as ts
from xdress import cythongen as cg
from xdress import autodescribe as ad

import nose
from nose.tools import assert_equal

from numpy.testing import dec

import pprint

if not os.path.isdir('build'):
    os.mkdir('build')

exp_toaster_desc = {
    'name': 'Toaster',
    'namespace': 'bright',
    'parents': ['FCComp'],
    'attrs': {
        'nslices': 'uint32',
        'toastiness': 'str',
        'rate': 'float32',
        },
    'methods': {
        ('Toaster',): None,
        ('~Toaster',): None, 
        ('make_toast', ('when', 'str'), ('nslices', 'uint32', 1)): 'int32',
        },
    }

meta_toaster_desc = {
    'name': 'Toaster',
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

full_toaster_desc = {
    'name': 'Toaster',
    'header_filename': 'toaster.h',
    'srcpxd_filename': 'cpp_toaster.pxd',
    'namespace': 'bright',
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
    'parents': ['FCComp'],
    'attrs': {
        'nslices': 'uint32',
        'toastiness': 'str',
        'rate': 'float32',
        },
    'methods': {
        ('Toaster',): None,
        ('~Toaster',): None, 
        ('make_toast', ('when', 'str'), ('nslices', 'uint32', 1)): 'int32',
        },
    }

# FIXME
#def test_describe_gccxml():
#    fname = os.path.join(os.path.split(__file__)[0], 'toaster.h')
#    obs = ad.describe(fname, classname='Toaster', parser='gccxml', verbose=False)
#    exp = exp_toaster_desc
#    assert_equal(obs, exp)


def test_merge_descriptions():
    obs = ad.merge_descriptions([exp_toaster_desc, meta_toaster_desc])
    exp = full_toaster_desc
    assert_equal(obs, exp)


@dec.skipif(ad.pycparser is None)
def test_pycparser_describe_device_measure():
    obs = ad.pycparser_describe('device.c', 'Device_measure', 'func')
    exp = {'name': 'Device_measure', 
           'namespace': None,
           'signatures': {
            ('Device_measure', ('aiValue', ('uint32', '*'))): ('enum', 
                                    'ErrorStatusTag', (('ERROR_OK', 0), 
                                                       ('ERROR_FAILED_INIT', 1))),
            ('Device_measure', ('deviceNumber', 'uchar'), 
                               ('aiValue', ('uint32', '*'))): ('enum', 
                                    'ErrorStatusTag', (('ERROR_OK', 0), 
                                                       ('ERROR_FAILED_INIT', 1))),
            }
           }
    pprint.pprint(obs)
    pprint.pprint(exp)
    assert_equal(obs, exp)

@dec.skipif(ad.pycparser is None)
def test_pycparser_describe_device_init():
    obs = ad.pycparser_describe('device.c', 'Device_Init', 'func')
    exp = {'name': 'Device_Init', 
           'namespace': None,
           'signatures': {
            ('Device_Init', ('param', ('DeviceParamTag', '*'))): ('enum', 
                                    'ErrorStatusTag', (('ERROR_OK', 0), 
                                                       ('ERROR_FAILED_INIT', 1))),
            }
           }
    pprint.pprint(obs)
    pprint.pprint(exp)
    assert_equal(obs, exp)

@dec.skipif(ad.pycparser is None)
def test_pycparser_describe_device_descriptor_tag():
    obs = ad.pycparser_describe('device.c', 'DeviceDescriptorTag', 'class')
    exp = {'name': 'DeviceDescriptorTag', 
           'namespace': None,
           'construct': 'struct',
           'parents': None,
           'attrs': {
            'deviceNumber': 'uchar',
            'deviceMeasurement': ('function_pointer', (('_0', ('uint32', '*')),), 
                                    ('enum', 'ErrorStatusTag', (
                                        ('ERROR_OK', 0),
                                        ('ERROR_FAILED_INIT', 1)))),
            },
           'methods': {},
           }
    pprint.pprint(obs)
    pprint.pprint(exp)
    assert_equal(obs, exp)

if __name__ == '__main__':
    nose.runmodule()
