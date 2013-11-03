from __future__ import print_function
import os
from pprint import pprint,pformat

from xdress.typesystem import TypeSystem
from xdress import cythongen as cg
from xdress import autodescribe as ad

import nose
from nose.tools import assert_equal
from tools import unit

from numpy.testing import dec

ts = TypeSystem()

if not os.path.isdir('build'):
    os.mkdir('build')

exp_toaster_desc = {
    'name': 'Toaster',
    'namespace': 'xdress',
    'parents': [],
    'construct': 'class',
    'attrs': {
        'nslices': 'uintc',
        'toastiness': 'str',
        'rate': 'float32',
        },
    'methods': {
        ('Toaster', ('slices', 'intc', 7)): None,
        ('~Toaster',): None, 
        ('make_toast', ('when', 'str'), ('nslices', 'uintc', 1)): 'intc',
        ('templates', ('strange', ('Base', 'intc', 3, 0))): ('Base', 'float32', 0, 0),
        },
    'type': 'Toaster',
    }

meta_toaster_desc = {
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

full_toaster_desc = {
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
    'parents': [],
    'attrs': {
        'nslices': 'uintc',
        'toastiness': 'str',
        'rate': 'float32',
        },
    'methods': {
        ('Toaster', ('slices', 'intc', 7)): None,
        ('~Toaster',): None, 
        ('make_toast', ('when', 'str'), ('nslices', 'uintc', 1)): 'intc',
        ('templates', ('strange', ('Base', 'intc', 3, 0))): ('Base', 'float32', 0, 0),
        },
    'type': 'Toaster',
    }

def show_diff(a,b,key=None):
    """Generated a colored diff between two strings.
    If key is passed, {0} and {1} are substituted with the colors of a and b, respectively."""
    red   = chr(27)+'[1;31m'
    green = chr(27)+'[1;32m'
    blue  = chr(27)+'[1;34m'
    clear = chr(27)+'[00m'
    import difflib
    m = difflib.SequenceMatcher(a=a,b=b,autojunk=0)
    r = []
    if key is not None:
        r.extend((green,key.format(blue+'blue'+green,red+'red'+green)))
    ia,ib = 0,0
    for ja,jb,n in m.get_matching_blocks():
        r.extend((blue, a[ia:ja],
                  red,  b[ib:jb],
                  clear,a[ja:ja+n]))
        ia = ja+n
        ib = jb+n
    return ''.join(r)

@unit
def test_describe_gccxml():
    fname = os.path.join(os.path.split(__file__)[0], 'toaster.h')
    ts.register_classname('Toaster', 'toaster', 'toaster', 'cpp_toaster')
    obs = ad.describe(fname, name='Toaster', parsers='gccxml', verbose=False, ts=ts)
    exp = exp_toaster_desc
    try:
        assert_equal(obs, exp)
    except:
        key = '\n\n# only expected = {0}, only computed = {1}\n'
        print(show_diff(pformat(exp),pformat(obs),key=key))
        raise

@unit
def test_describe_clang():
    fname = os.path.join(os.path.split(__file__)[0], 'toaster.h')
    obs = ad.describe(fname, name='Toaster', parsers='clang', verbose=False, ts=ts)
    exp = exp_toaster_desc
    try:
        assert_equal(obs, exp)
    except:
        key = '\n\n# only expected = {0}, only computed = {1}\n'
        print(show_diff(pformat(exp),pformat(obs),key=key))
        raise

@unit
def test_merge_descriptions():
    obs = ad.merge_descriptions([exp_toaster_desc, meta_toaster_desc])
    exp = full_toaster_desc
    #pprint(exp)
    #pprint(obs)
    assert_equal(obs, exp)

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
    #pprint(obs)
    #pprint(exp)
    assert_equal(obs, exp)

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
    #pprint(exp)
    #pprint(obs)
    assert_equal(exp, obs)

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
    #pprint(obs)
    #pprint(exp)
    assert_equal(obs, exp)

if __name__ == '__main__':
    nose.runmodule()
