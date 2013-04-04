"""C++ wrapper for extra types header."""

ctypedef unsigned int uint

cdef extern from "xdress_extra_types.h" namespace "xdress_extra_types":

    ctypedef struct complex_t:
        double re
        double im

cdef complex_t py2c_complex(object pyv)
