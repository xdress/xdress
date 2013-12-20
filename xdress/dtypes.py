"""Generates numpy dtypes for C/C++ classes and structs.

This module is available as an xdress plugin by the name ``xdress.dtypes``.

:author: Anthony Scopatz <scopatz@gmail.com>

NumPy Dtypes Wrapper API
========================
"""
from __future__ import print_function
import os
import sys
import pprint

from .utils import newoverwrite, newcopyover, ensuredirs, indent, indentstr, \
    RunControl, NotSpecified
from .plugins import Plugin
from .typesystem import TypeSystem

if sys.version_info[0] >= 3: 
    basestring = str


#
# Vectors
#

_pyxvector = """# {ctype} dtype
cdef MemoryKnight[{ctype}] mk_{fncname} = MemoryKnight[{ctype}]()
cdef MemoryKnight[PyXD{clsname}_Type] mk_{fncname}_type = MemoryKnight[PyXD{clsname}_Type]()

cdef object pyxd_{fncname}_getitem(void * data, void * arr):
{c2pydecl.indent4}
{c2pybody.indent4}
    pyval = {c2pyrtn}
    return pyval

cdef int pyxd_{fncname}_setitem(object value, void * data, void * arr):
    cdef {ctype} * new_data
{py2cdecl.indent4}
    if {isinst}:
{py2cbody.indent8}
        new_data = mk_{fncname}.renew(data)
        new_data[0] = {py2crtn}
        return 0
    else:
        return -1

cdef void pyxd_{fncname}_copyswapn(void * dest, np.npy_intp dstride, void * src, np.npy_intp sstride, np.npy_intp n, int swap, void * arr):
    cdef np.npy_intp i
    cdef char * a 
    cdef char * b 
    cdef char c = 0
    cdef int j
    cdef int m
    cdef {ctype} * new_dest

    if src != NULL:
        if (sstride == sizeof({ctype}) and dstride == sizeof({ctype})):
            new_dest = mk_{fncname}.renew(dest)
            new_dest[0] = deref(<{ctype} *> src)
        else:
            a = <char *> dest
            b = <char *> src
            for i in range(n):
                new_dest = mk_{fncname}.renew(<void *> a)
                new_dest[0] = deref(<{ctype} *> b)
                a += dstride
                b += sstride
    if swap: 
        m = sizeof({ctype}) / 2
        a = <char *> dest
        for i in range(n, 0, -1):
            b = a + (sizeof({ctype}) - 1);
            for j in range(m):
                c = a[0]
                a[0] = b[0]
                a += 1
                b[0] = c
                b -= 1
            a += dstride - m

cdef void pyxd_{fncname}_copyswap(void * dest, void * src, int swap, void * arr):
    cdef char * a 
    cdef char * b 
    cdef char c = 0
    cdef int j
    cdef int m
    cdef {ctype} * new_dest
    if src != NULL:
        new_dest = mk_{fncname}.renew(dest)
        new_dest[0] = (<{ctype} *> src)[0]
    if swap:
        m = sizeof({ctype}) / 2
        a = <char *> dest
        b = a + (sizeof({ctype}) - 1);
        for j in range(m):
            c = a[0]
            a[0] = b[0]
            a += 1
            b[0] = c
            b -= 1

cdef np.npy_bool pyxd_{fncname}_nonzero(void * data, void * arr):
    return (data != NULL)
    # FIXME comparisons not defined for arbitrary types
    #cdef {ctype} zero = {ctype}()
    #return ((<{ctype} *> data)[0] != zero)

cdef int pyxd_{fncname}_compare(const void * d1, const void * d2, void * arr):
    return (d1 == d2) - 1
    # FIXME comparisons not defined for arbitrary types
    #if deref(<{ctype} *> d1) == deref(<{ctype} *> d2):
    #    return 0
    #else:
    #    return -1

cdef PyArray_ArrFuncs PyXD_{clsname}_ArrFuncs 
PyArray_InitArrFuncs(&PyXD_{clsname}_ArrFuncs)
PyXD_{clsname}_ArrFuncs.getitem = <PyArray_GetItemFunc *> (&pyxd_{fncname}_getitem)
PyXD_{clsname}_ArrFuncs.setitem = <PyArray_SetItemFunc *> (&pyxd_{fncname}_setitem)
PyXD_{clsname}_ArrFuncs.copyswapn = <PyArray_CopySwapNFunc *> (&pyxd_{fncname}_copyswapn)
PyXD_{clsname}_ArrFuncs.copyswap = <PyArray_CopySwapFunc *> (&pyxd_{fncname}_copyswap)
PyXD_{clsname}_ArrFuncs.nonzero = <PyArray_NonzeroFunc *> (&pyxd_{fncname}_nonzero)
PyXD_{clsname}_ArrFuncs.compare = <PyArray_CompareFunc *> (&pyxd_{fncname}_compare)

cdef object pyxd_{fncname}_type_alloc(PyTypeObject * self, Py_ssize_t nitems):
    cdef PyXD{clsname}_Type * cval
    cdef object pyval
    cval = mk_{fncname}_type.defnew()
    cval.ob_typ = self
    pyval = <object> cval
    return pyval

cdef void pyxd_{fncname}_type_dealloc(object self):
    cdef PyXD{clsname}_Type * cself = <PyXD{clsname}_Type *> self
    mk_{fncname}_type.deall(cself)
    return

cdef object pyxd_{fncname}_type_new(PyTypeObject * subtype, object args, object kwds):
    return pyxd_{fncname}_type_alloc(subtype, 0)

cdef void pyxd_{fncname}_type_free(void * self):
    return

cdef object pyxd_{fncname}_type_str(object self):
    cdef PyXD{clsname}_Type * cself = <PyXD{clsname}_Type *> self
{cself2pydecl.indent4}
{cself2pybody.indent4}
    pyval = {cself2pyrtn}
    s = str(pyval)
    return s

cdef object pyxd_{fncname}_type_repr(object self):
    cdef PyXD{clsname}_Type * cself = <PyXD{clsname}_Type *> self
{cself2pydecl.indent4}
{cself2pybody.indent4}
    pyval = {cself2pyrtn}
    s = repr(pyval)
    return s

cdef int pyxd_{fncname}_type_compare(object a, object b):
    return (a is b) - 1
    # FIXME comparisons not defined for arbitrary types
    #cdef PyXD{clsname}_Type * x
    #cdef PyXD{clsname}_Type * y
    #if type(a) is not type(b):
    #    raise NotImplementedError
    #x = <PyXD{clsname}_Type *> a
    #y = <PyXD{clsname}_Type *> b
    #if (x.obval == y.obval):
    #    return 0
    #elif (x.obval < y.obval):
    #    return -1
    #elif (x.obval > y.obval):
    #    return 1
    #else:
    #    raise NotImplementedError

cdef object pyxd_{fncname}_type_richcompare(object a, object b, int op):
    if op == Py_EQ:
        return (a is b)
    elif op == Py_NE:
        return (a is not b)
    else:
        return NotImplemented
    # FIXME comparisons not defined for arbitrary types
    #cdef PyXD{clsname}_Type * x
    #cdef PyXD{clsname}_Type * y
    #if type(a) is not type(b):
    #    return NotImplemented
    #x = <PyXD{clsname}_Type *> a
    #y = <PyXD{clsname}_Type *> b
    #if op == Py_LT:
    #    return (x.obval < y.obval)
    #elif op == Py_LE:
    #    return (x.obval <= y.obval)
    #elif op == Py_EQ:
    #    return (x.obval == y.obval)
    #elif op == Py_NE:
    #    return (x.obval != y.obval)
    #elif op == Py_GT:
    #    return (x.obval > y.obval)
    #elif op == Py_GE:
    #    return (x.obval >= y.obval)
    #else:
    #    return NotImplemented    

cdef long pyxd_{fncname}_type_hash(object self):
    return id(self)

cdef PyMemberDef pyxd_{fncname}_type_members[1]
pyxd_{fncname}_type_members[0] = PyMemberDef(NULL, 0, 0, 0, NULL)

cdef PyGetSetDef pyxd_{fncname}_type_getset[1]
pyxd_{fncname}_type_getset[0] = PyGetSetDef(NULL)

cdef bint pyxd_{fncname}_is_ready
cdef type PyXD_{clsname} = type("xd_{fncname}", ((<object> PyArray_API[10]),), {{}})
pyxd_{fncname}_is_ready = PyType_Ready(<object> PyXD_{clsname})
(<PyTypeObject *> PyXD_{clsname}).tp_basicsize = sizeof(PyXD{clsname}_Type)
(<PyTypeObject *> PyXD_{clsname}).tp_itemsize = 0
(<PyTypeObject *> PyXD_{clsname}).tp_doc = "Python scalar type for {ctype}"
(<PyTypeObject *> PyXD_{clsname}).tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_CHECKTYPES | Py_TPFLAGS_HEAPTYPE
(<PyTypeObject *> PyXD_{clsname}).tp_alloc = pyxd_{fncname}_type_alloc
(<PyTypeObject *> PyXD_{clsname}).tp_dealloc = pyxd_{fncname}_type_dealloc
(<PyTypeObject *> PyXD_{clsname}).tp_new = pyxd_{fncname}_type_new
(<PyTypeObject *> PyXD_{clsname}).tp_free = pyxd_{fncname}_type_free
(<PyTypeObject *> PyXD_{clsname}).tp_str = pyxd_{fncname}_type_str
(<PyTypeObject *> PyXD_{clsname}).tp_repr = pyxd_{fncname}_type_repr
(<PyTypeObject *> PyXD_{clsname}).tp_base = (<PyTypeObject *> PyArray_API[10])  # PyGenericArrType_Type
(<PyTypeObject *> PyXD_{clsname}).tp_hash = pyxd_{fncname}_type_hash
emit_ifpy2k()
(<PyTypeObject *> PyXD_{clsname}).tp_compare = &pyxd_{fncname}_type_compare
emit_endif()
(<PyTypeObject *> PyXD_{clsname}).tp_richcompare = pyxd_{fncname}_type_richcompare
(<PyTypeObject *> PyXD_{clsname}).tp_members = pyxd_{fncname}_type_members
(<PyTypeObject *> PyXD_{clsname}).tp_getset = pyxd_{fncname}_type_getset
pyxd_{fncname}_is_ready = PyType_Ready(<object> PyXD_{clsname})
Py_INCREF(PyXD_{clsname})
XD{clsname} = PyXD_{clsname}

cdef PyArray_Descr * c_xd_{fncname}_descr = <PyArray_Descr *> malloc(sizeof(PyArray_Descr))
(<PyObject *> c_xd_{fncname}_descr).ob_refcnt = 0 # ob_refcnt
(<PyObject *> c_xd_{fncname}_descr).ob_type = <PyTypeObject *> PyArray_API[3]
c_xd_{fncname}_descr.typeobj = <PyTypeObject *> PyXD_{clsname} # typeobj
c_xd_{fncname}_descr.kind = 'x'  # kind, for xdress
c_xd_{fncname}_descr.type = 'x'  # type
c_xd_{fncname}_descr.byteorder = '='  # byteorder
c_xd_{fncname}_descr.flags = 0    # flags
c_xd_{fncname}_descr.type_num = 0    # type_num, assigned at registration
c_xd_{fncname}_descr.elsize = sizeof({ctype})  # elsize, 
c_xd_{fncname}_descr.alignment = 8  # alignment
c_xd_{fncname}_descr.subarray = NULL  # subarray
c_xd_{fncname}_descr.fields = NULL  # fields
c_xd_{fncname}_descr.names = NULL
(<PyArray_Descr *> c_xd_{fncname}_descr).f = <PyArray_ArrFuncs *> &PyXD_{clsname}_ArrFuncs  # f == PyArray_ArrFuncs

cdef object xd_{fncname}_descr = <object> (<void *> c_xd_{fncname}_descr)
Py_INCREF(<object> xd_{fncname}_descr)
xd_{fncname} = xd_{fncname}_descr
cdef int xd_{fncname}_num = PyArray_RegisterDataType(c_xd_{fncname}_descr)
dtypes['{fncname}'] = xd_{fncname}
dtypes['xd_{fncname}'] = xd_{fncname}
dtypes[xd_{fncname}_num] = xd_{fncname}

"""

def genpyx_vector(t, ts):
    """Returns the pyx snippet for a vector of type t."""
    t = ts.canon(t)
    kw = dict(clsname=ts.cython_classname(t)[1], humname=ts.humanname(t)[1], 
              fncname=ts.cython_functionname(t)[1], 
              ctype=ts.cython_ctype(t), pytype=ts.cython_pytype(t), 
              cytype=ts.cython_cytype(t), stlcontainers=ts.stlcontainers, 
              extra_types=ts.extra_types)
    t0 = t
    while not isinstance(t0, basestring):
        t0 = t[0]
    fpt = ts.from_pytypes[t0]
    kw['isinst'] = " or ".join(["isinstance(value, {0})".format(x) for x in fpt])
    c2pykeys = ['c2pydecl', 'c2pybody', 'c2pyrtn']
    c2py = ts.cython_c2py("deref(<{0} *> data)".format(kw['ctype']), t, cached=False,
                          proxy_name="data_proxy")
    kw.update([(k, indentstr(v or '')) for k, v in zip(c2pykeys, c2py)])
    cself2pykeys = ['cself2pydecl', 'cself2pybody', 'cself2pyrtn']
    cself2py = ts.cython_c2py("(cself.obval)", t, cached=False, proxy_name="val_proxy")
    kw.update([(k, indentstr(v or '')) for k, v in zip(cself2pykeys, cself2py)])
    py2ckeys = ['py2cdecl', 'py2cbody', 'py2crtn']
    py2c = ts.cython_py2c("value", t)
    kw.update([(k, indentstr(v or '')) for k, v in zip(py2ckeys, py2c)])
    return _pyxvector.format(**kw)

_pxdvector = """# {ctype} dtype
ctypedef struct PyXD{clsname}_Type:
    Py_ssize_t ob_refcnt
    PyTypeObject *ob_typ
    {ctype} obval

cdef object pyxd_{fncname}_getitem(void * data, void * arr)
cdef int pyxd_{fncname}_setitem(object value, void * data, void * arr)
cdef void pyxd_{fncname}_copyswapn(void * dest, np.npy_intp dstride, void * src, np.npy_intp sstride, np.npy_intp n, int swap, void * arr)
cdef void pyxd_{fncname}_copyswap(void * dest, void * src, int swap, void * arr)
cdef np.npy_bool pyxd_{fncname}_nonzero(void * data, void * arr)
"""

def genpxd_vector(t, ts):
    """Returns the pxd snippet for a vector of type t."""
    t = ts.canon(t)
    kw = dict(clsname=ts.cython_classname(t)[1], humname=ts.humanname(t)[1], 
              ctype=ts.cython_ctype(t), pytype=ts.cython_pytype(t), 
              fncname=ts.cython_functionname(t)[1], 
              cytype=ts.cython_cytype(t),)
    return _pxdvector.format(**kw)


_testvector = """# Vector{clsname}
def test_vector_{fncname}():
    a = np.array({0}, dtype={stlcontainers}.xd_{fncname})
    #for x, y in zip(a, np.array({0}, dtype={stlcontainers}.xd_{fncname})):
    #    assert_equal(x, y)
    a[:] = {1}
    #for x, y in zip(a, np.array({1}, dtype={stlcontainers}.xd_{fncname})):
    #    assert_equal(x, y)
    a = np.array({2} + {3}, dtype={stlcontainers}.xd_{fncname})
    #for x, y in zip(a, np.array({2} + {3}, dtype={stlcontainers}.xd_{fncname})):
    #    assert_equal(x, y)
    b =  np.array(({2} + {3})[::2], dtype={stlcontainers}.xd_{fncname})
    #for x, y in zip(a[::2], b):
    #    assert_equal(x, y)
    a[:2] = b[-2:]
    print(a)

"""
def gentest_vector(t, ts):
    """Returns the test snippet for a set of type t."""
    t = ts.canon(t)
    if ('vector', t, 0) in testvals:
        s = _testvector.format(*[repr(i) for i in testvals['vector', t, 0]], 
                               clsname=ts.cython_classname(t)[1],
                               fncname=ts.cython_functionname(t)[1],
                               stlcontainers=ts.stlcontainers)
    else:
        s = ""
    return s



#
# Controlers 
#

_pyxheader = """###################
###  WARNING!!! ###
###################
# This file has been autogenerated

# Cython imports
from libcpp.set cimport set as cpp_set
from libcpp.vector cimport vector as cpp_vector
from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as inc
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libcpp.string cimport string as std_string
from libcpp.utility cimport pair
from libcpp.map cimport map as cpp_map
from libcpp.vector cimport vector as cpp_vector
from cpython.version cimport PY_MAJOR_VERSION
from cpython.ref cimport PyTypeObject
from cpython.type cimport PyType_Ready
from cpython.object cimport Py_LT, Py_LE, Py_EQ, Py_NE, Py_GT, Py_GE

# Python Imports
import collections

cimport numpy as np
import numpy as np

np.import_array()

cimport {extra_types}

# Cython Imports For Types
{cimports}

# Imports For Types
{imports}

dtypes = {{}}

if PY_MAJOR_VERSION >= 3:
    basestring = str


# Dirty ifdef, else, else preprocessor hack
# see http://comments.gmane.org/gmane.comp.python.cython.user/4080
cdef extern from *:
    cdef void emit_ifpy2k "#if PY_MAJOR_VERSION == 2 //" ()
    cdef void emit_ifpy3k "#if PY_MAJOR_VERSION == 3 //" ()
    cdef void emit_else "#else //" ()
    cdef void emit_endif "#endif //" ()

"""
def genpyx(template, header=None, ts=None):
    ts = ts or TypeSystem()
    """Returns a string of a pyx file representing the given template."""
    pyxfuncs = dict([(k[7:], v) for k, v in globals().items() \
                    if k.startswith('genpyx_') and callable(v)])
    pyx = _pyxheader if header is None else header
    with ts.swap_stlcontainers(None):
        import_tups = set()
        cimport_tups = set()
        for t in template:
            for arg in t[1:]:
                ts.cython_import_tuples(arg, import_tups)
                ts.cython_cimport_tuples(arg, cimport_tups)
        imports = "\n".join(ts.cython_import_lines(import_tups))
        cimports = "\n".join(ts.cython_cimport_lines(cimport_tups))
        pyx = pyx.format(extra_types=ts.extra_types, cimports=cimports, 
                         imports=imports)
        for t in template:
            pyx += pyxfuncs[t[0]](*t[1:], ts=ts) + "\n\n" 
    return pyx


_pxdheader = """###################
###  WARNING!!! ###
###################
# This file has been autogenerated

# Cython imports
from libcpp.set cimport set as cpp_set
from libcpp.vector cimport vector as cpp_vector
from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as inc
from libcpp.string cimport string as std_string
from libcpp.utility cimport pair
from libcpp.map cimport map as cpp_map
from libcpp.vector cimport vector as cpp_vector
from libc cimport stdio
from cpython.version cimport PY_MAJOR_VERSION
from cpython.ref cimport PyTypeObject, Py_INCREF, Py_XDECREF
from cpython.type cimport PyType_Ready
from cpython.object cimport PyObject
from cpython.object cimport Py_LT, Py_LE, Py_EQ, Py_NE, Py_GT, Py_GE

# Python Imports
cimport numpy as np

# Local imports
cimport {extra_types}

cimport numpy as np


# Cython Imports For Types
{cimports}

cdef extern from "Python.h":
    ctypedef Py_ssize_t Py_ssize_t

    cdef long Py_TPFLAGS_DEFAULT 
    cdef long Py_TPFLAGS_BASETYPE 
    cdef long Py_TPFLAGS_CHECKTYPES
    cdef long Py_TPFLAGS_HEAPTYPE

    ctypedef struct PyGetSetDef:
        char * name

    ctypedef struct PyTypeObject:
        char * tp_name
        int tp_basicsize
        int tp_itemsize
        object tp_alloc(PyTypeObject *, Py_ssize_t)
        void tp_dealloc(object)
        object tp_richcompare(object, object, int)
        object tp_new(PyTypeObject *, object, object)
        object tp_str(object)
        object tp_repr(object)
        long tp_hash(object)
        long tp_flags
        char * tp_doc
        PyMemberDef * tp_members
        PyGetSetDef * tp_getset
        PyTypeObject * tp_base
        void tp_free(void *)
        # This is a dirty hack by declaring to Cython both the Python 2 & 3 APIs
        int (*tp_compare)(object, object)      # Python 2
        void * (*tp_reserved)(object, object)  # Python 3

# structmember.h isn't included in Python.h for some reason
cdef extern from "structmember.h":
    ctypedef struct PyMemberDef:
        char * name
        int type
        Py_ssize_t offset
        int flags
        char * doc

cdef extern from "numpy/arrayobject.h":

    ctypedef object (*PyArray_GetItemFunc)(void *, void *)
    ctypedef int (*PyArray_SetItemFunc)(object, void *, void *)
    ctypedef void (*PyArray_CopySwapNFunc)(void *, np.npy_intp, void *, np.npy_intp, np.npy_intp, int, void *)
    ctypedef void (*PyArray_CopySwapFunc)(void *, void *, int, void *)
    ctypedef int (*PyArray_CompareFunc)(const void* d1, const void *, void *)
    ctypedef int (*PyArray_ArgFunc)(void *, np.npy_intp, np.npy_intp *, void *)
    ctypedef void (*PyArray_DotFunc)(void *, np.npy_intp, void *, np.npy_intp, void *, np.npy_intp, void *)
    ctypedef int (*PyArray_ScanFunc)(stdio.FILE *, void *, void *, void *)
    ctypedef int (*PyArray_FromStrFunc)(char *, void *, char **, void *)
    ctypedef np.npy_bool (*PyArray_NonzeroFunc)(void *, void *)
    ctypedef void (*PyArray_FillFunc)(void *, np.npy_intp, void *)
    ctypedef void (*PyArray_FillWithScalarFunc)(void *, np.npy_intp, void *, void *)
    ctypedef int (*PyArray_SortFunc)(void *, np.npy_intp, void *)
    ctypedef int (*PyArray_ArgSortFunc)(void *, np.npy_intp *, np.npy_intp, void *)
    ctypedef np.NPY_SCALARKIND (*PyArray_ScalarKindFunc)(np.PyArrayObject *)

    ctypedef struct PyArray_ArrFuncs:
        np.PyArray_VectorUnaryFunc ** cast
        PyArray_GetItemFunc *getitem
        PyArray_SetItemFunc *setitem
        PyArray_CopySwapNFunc *copyswapn
        PyArray_CopySwapFunc *copyswap
        PyArray_CompareFunc *compare
        PyArray_ArgFunc *argmax
        PyArray_DotFunc *dotfunc
        PyArray_ScanFunc *scanfunc
        PyArray_FromStrFunc *fromstr
        PyArray_NonzeroFunc *nonzero
        PyArray_FillFunc *fill
        PyArray_FillWithScalarFunc *fillwithscalar
        PyArray_SortFunc *sort
        PyArray_ArgSortFunc *argsort
        PyObject *castdict
        PyArray_ScalarKindFunc *scalarkind
        int **cancastscalarkindto
        int *cancastto
        int listpickle

    cdef void PyArray_InitArrFuncs(PyArray_ArrFuncs *)

    ctypedef struct PyArray_ArrayDescr:
        PyArray_Descr * base
        PyObject  *shape

    cdef void ** PyArray_API

    cdef PyTypeObject * PyArrayDescr_Type
    
    ctypedef struct PyArray_Descr:
        Py_ssize_t ob_refcnt
        PyTypeObject * ob_type
        PyTypeObject * typeobj
        char kind
        char type
        char byteorder
        int flags
        int type_num
        int elsize
        int alignment
        PyArray_ArrayDescr * subarray
        PyObject * fields
        PyObject * names
        PyArray_ArrFuncs * f

    cdef int PyArray_RegisterDataType(PyArray_Descr *)

    cdef object PyArray_Scalar(void *, PyArray_Descr *, object)

cdef extern from "{extra_types}.h" namespace "{extra_types}":
    cdef cppclass MemoryKnight[T]:
        MemoryKnight() nogil except +
        T * defnew() nogil except +
        T * renew(void *) nogil except +
        void deall(T *) nogil except +

"""
def genpxd(template, header=None, ts=None):
    """Returns a string of a pxd file representing the given template."""
    ts = ts or TypeSystem()
    pxdfuncs = dict([(k[7:], v) for k, v in globals().items() \
                    if k.startswith('genpxd_') and callable(v)])
    pxd = _pxdheader if header is None else header
    with ts.swap_stlcontainers(None):
        cimport_tups = set()
        for t in template:
            for arg in t[1:]:
                ts.cython_cimport_tuples(arg, cimport_tups, set(['c']))
        cimports = "\n".join(ts.cython_cimport_lines(cimport_tups))
        pxd = pxd.format(extra_types=ts.extra_types, cimports=cimports)
    for t in template:
        pxd += pxdfuncs[t[0]](*t[1:], ts=ts) + "\n\n" 
    return pxd


_testheader = '''"""Tests the part of stlconverters that is accessible from Python."""
###################
###  WARNING!!! ###
###################
# This file has been autogenerated
from __future__ import print_function
from unittest import TestCase
import nose

from nose.tools import assert_equal, assert_not_equal, assert_raises, raises, \\
    assert_almost_equal, assert_true, assert_false, assert_in

from numpy.testing import assert_array_equal, assert_array_almost_equal

import os
import numpy  as np
from collections import Container, Mapping

from {package} import {stlcontainers}


'''

_testfooter = '''if __name__ == '__main__':
    nose.run()
'''

def gentest(template, header=None, package='..', ts=None):
    """Returns a string of a test file representing the given template."""
    ts = ts or TypeSystem()
    testfuncs = dict([(k[8:], v) for k, v in globals().items() \
                    if k.startswith('gentest_') and callable(v)])
    test = _testheader if header is None else header
    test = test.format(stlcontainers=ts.stlcontainers, package=package)
    for t in template:
        test += testfuncs[t[0]](*t[1:], ts=ts) + "\n\n" 
    test += _testfooter
    return test


def genfiles(template, fname='temp', pxdname=None, testname=None, 
             pyxheader=None, pxdheader=None, testheader=None, package='..', 
             ts=None, verbose=False):
    """Generates all cython source files needed to create the wrapper."""
    ts = ts or TypeSystem()
    # munge some filenames
    fname = fname[:-4] if fname.endswith('.pyx') else fname
    pxdname = fname if pxdname is None else pxdname
    pxdname = pxdname + '.pxd' if not pxdname.endswith('.pxd') else pxdname
    testname = 'test_' + fname if testname is None else testname
    testname = testname + '.py' if not testname.endswith('.py') else testname
    fname += '.pyx'

    # register dtypes
    for t in template:
        if t[0] == 'vector':
            ts.register_numpy_dtype(t[1])

    pyx = genpyx(template, pyxheader, ts=ts)
    pxd = genpxd(template, pxdheader, ts=ts)
    test = gentest(template, testheader, package, ts=ts)

    newoverwrite(pyx, fname, verbose)
    newoverwrite(pxd, pxdname, verbose)
    newoverwrite(test, testname, verbose)


#
# XDress Plugin
#

class XDressPlugin(Plugin):
    """This class provides extra type functionality for xdress."""

    requires = ('xdress.base', 'xdress.extratypes')

    defaultrc = RunControl(
        stlcontainers=[],
        make_stlcontainers=True,
        )

    rcdocs = {
        "stlcontainers": "List of C++ standard library containers to wrap.",
        "stlcontainers_module": ("Module name for C++ standard library "
                                 "container wrappers."),
        "make_stlcontainers": ("Flag for enabling / disabling creating the "
                               "C++ standard library container wrappers."),
        }

    def update_argparser(self, parser):
        parser.add_argument('--stlcontainers-module', action='store', 
                dest='stlcontainers_module', help=self.rcdocs["stlcontainers_module"])
        parser.add_argument('--make-stlcontainers', action='store_true',
                    dest='make_stlcontainers', help="make C++ STL container wrappers")
        parser.add_argument('--no-make-stlcontainers', action='store_false',
              dest='make_stlcontainers', help="don't make C++ STL container wrappers")

    def setup(self, rc):
        print("stlwrap: registering C++ standard library types")
        ts = rc.ts
        ts.stlcontainers = rc.stlcontainers_module
        # register dtypes
        for t in rc.stlcontainers:
            if t[0] == 'vector':
                ts.register_numpy_dtype(t[1])


    def execute(self, rc):
        if not rc.make_stlcontainers:
            return
        print("stlwrap: generating C++ standard library wrappers & converters")
        fname = os.path.join(rc.packagedir, rc.stlcontainers_module)
        ensuredirs(fname)
        testname = 'test_' + rc.stlcontainers_module
        testname = os.path.join(rc.packagedir, 'tests', testname)
        ensuredirs(testname)
        genfiles(rc.stlcontainers, fname=fname, testname=testname, package=rc.package, 
                 ts=rc.ts, verbose=rc.verbose)


#
# Test main
#

if __name__ == "__main__":
    #t = [('set', 'int')]
    #t = [('set', 'str')]
    #t = [('py2c_map', 'int', 'int')]
    t = [('py2c_set', 'str')]
    print(genpyx(t))
