"""C++ wrapper for extra types header."""

ctypedef unsigned int uint

ctypedef struct complex_t:
    double re
    double im

cdef complex_t py2c_complex(object pyv)
