"""Extra type conversions for Bright."""
from libcpp.map cimport map as cpp_map
from libcpp.string cimport string as std_string

from pyne cimport cpp_nucname
from pyne cimport nucname
from pyne import nucname

from pyne cimport stlconverters as conv
from pyne import stlconverters as conv

cimport typeconverters

cdef cpp_map[int, double] sepeff_py2c(object x):
    cdef int k
    cdef std_string ks
    cdef cpp_map[int, double] se = cpp_map[int, double]()
    for key, val in x.iteritems():
        if isinstance(key, int):
            k = key
            if 1000 < k:
                k = cpp_nucname.zzaaam(k)             
        elif isinstance(key, basestring):
            ks = std_string(<char *> key)
            if 0 < cpp_nucname.name_zz.count(ks):
                k = cpp_nucname.name_zz[ks]
            else:
                k = cpp_nucname.zzaaam(ks)
        else:
            raise TypeError("Separation keys must be strings or integers.")
        se[k] = <double> val
    return se


