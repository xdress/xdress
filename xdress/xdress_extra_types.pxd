"""C++ wrapper for extra types header."""

ctypedef unsigned char uchar
ctypedef long long int64
ctypedef unsigned short uint16
ctypedef unsigned int uint32
ctypedef unsigned long long uint64
ctypedef long double float128

cdef extern from "{extra_types}.h" namespace "{extra_types}":

    ctypedef struct complex_t:
        double re
        double im

cdef complex_t py2c_complex(object pyv)
