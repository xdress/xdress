cimport xdress_extra_types

cdef xdress_extra_types.complex_t py2c_complex(object pyv):
    cdef xdress_extra_types.complex_t cv = xdress_extra_types.complex_t()
    pyv = complex(pyv)
    cv.re = pyv.real
    cv.im = pyv.imag
    return cv

