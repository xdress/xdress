from bright.apigen import typesystem as ts
from bright.apigen import cythongen as cg

from nose.tools import assert_equal

toaster_desc = {
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
        'nslices': 'uint',
        'toastiness': 'str',
        'rate': 'float',
        },
    'methods': {
        ('Toaster',): None,
        ('~Toaster',): None, 
        ('make_toast', ('when', 'str'), ('nslices', 'uint', 1)): 'int',
        },
    }


exp_cpppxd = cg.AUTOGEN_WARNING + \
"""from libcpp.string cimport string as std_string
from pyne cimport extra_types

cdef extern from "toaster.h" namespace "bright":

    cdef cppclass Toaster(FCComp):
        # constructors
        Toaster() except +
        ~Toaster() except +

        # attributes
        extra_types.uint nslices
        double rate
        std_string toastiness

        # methods
        int make_toast(std_string) except +
        int make_toast(std_string, extra_types.uint) except +
"""

def test_gencpppxd():
    obs = cg.gencpppxd(toaster_desc).splitlines()
    exp = exp_cpppxd.splitlines()
    assert_equal(len(obs), len(exp))
    for o, e in zip(obs, exp):
        assert_equal(o, e)


exp_pxd = cg.AUTOGEN_WARNING + \
"""cimport cpp_toaster
cimport fccomp

cdef class Toaster(fccomp.FCComp):
    cdef cpp_toaster.Toaster * _inst
    cdef public bint _free_inst
"""

def test_genpxd():
    ts.register_class('FCComp', 
                      cython_c_type='cpp_fccomp.FCComp', 
                      cython_cimport='cpp_fccomp', 
                      cython_cy_type='fccomp.FCComp', 
                      cython_cyimport='fccomp')
    obs = cg.genpxd(toaster_desc).splitlines()
    ts.deregister_class('FCComp')
    exp = exp_pxd.splitlines()
    assert_equal(len(obs), len(exp))
    for o, e in zip(obs, exp):
        assert_equal(o, e)


exp_pyx = cg.AUTOGEN_WARNING + \
'''"""I am the Toaster lib! Hear me sizzle!
"""
from libcpp.string cimport string as std_string
from pyne cimport extra_types



cdef class Toaster(fccomp.FCComp):
    """I am the Toaster! FORKS DO NOT GO IN ME!"""

    # constuctors
    def __cinit__(self, *args, **kwargs):
        self._inst = NULL
        self._free_inst = True


    def __init__(self, *args, **kwargs):
        """"""
        self._inst = new cpp_toaster.Toaster()
    
    
    def __dealloc__(self):
        if self._free_inst:
            free(self._inst)


    # attributes
    property nslices:
        """no docstring for nslices, please file a bug report!"""
        def __get__(self):
            return int(self._inst.nslices)
    
        def __set__(self, value):
            self._inst.nslices = <extra_types.uint> long(value)
    
    
    property rate:
        """The rate at which the toaster can process slices."""
        def __get__(self):
            return float(self._inst.rate)
    
        def __set__(self, value):
            self._inst.rate = <double> value
    
    
    property toastiness:
        """white as snow or black as hell?"""
        def __get__(self):
            return str(<char *> self._inst.toastiness.c_str())
    
        def __set__(self, value):
            self._inst.toastiness = std_string(<char *> value)
    
    
    # methods
    def make_toast(self, when, nslices=1):
        """I'll make you some toast you can't refuse..."""
        cdef int rtnval
        rtnval = self._inst.make_toast(std_string(<char *> when), <extra_types.uint> long(nslices))
        return int(rtnval)
    
    
'''

def test_genpyx():
    ts.register_class('FCComp', 
                      cython_c_type='cpp_fccomp.FCComp', 
                      cython_cimport='cpp_fccomp', 
                      cython_cy_type='fccomp.FCComp', 
                      cython_cyimport='fccomp')
    obs = cg.genpyx(toaster_desc).splitlines()
    ts.deregister_class('FCComp')
    #print "\n".join(obs)
    exp = exp_pyx.splitlines()
    assert_equal(len(obs), len(exp))
    for o, e in zip(obs, exp):
        assert_equal(o, e)
