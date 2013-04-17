from __future__ import print_function
import os

from bright.apigen import typesystem as ts
from bright.apigen import cythongen as cg
from bright.apigen import autodescribe as ad

from nose.tools import assert_equal

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
    'cpppxd_filename': 'cpp_toaster.pxd',
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
    'cpppxd_filename': 'cpp_toaster.pxd',
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

