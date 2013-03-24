"""Extra type conversions for Bright."""
from libcpp.map cimport map as cpp_map
from pyne cimport stlconverters as conv
from pyne import stlconverters as conv

cdef cpp_map[int, double] sepeff_py2c(object x)
