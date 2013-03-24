"""Extra type conversions for Bright."""
from libcpp.map cimport map as cpp_map

cdef cpp_map[int, double] sepeff_py2c(object x)
