.. _advanced_tutorial:

*******************
Advanced Tutorial
*******************
The point of xdress is to wrap code.  The purpose of xdress is to generate 
idiomatic wrappers.  This tutorial describes some awesome things that you can 
do with xdress.

NumPy dtypes of Classes & Struct
================================
When using ``stlwrap``, the ``vector`` type actually creates custom 
numpy dtypes for the templated type.  This works not only on primitive types, 
other STL containers (map, set, vector), but also on any custom type that the
type system knows about!  Say we have the following run control file, which is 
used to wrap a C++ class Joan:

**xdressrc.py:**

.. code-block:: python

    package = "france"
    classes = [('Joan', 'france.cpp')]
    stlcontainers = [('vector', 'Joan')]

The classes list, with the autodescribe and cythongen plugins, will create
Python wrappers for he Joan class.  The stlcontainers will then see that the
vector template is dependent on Joan.  This will create and register a dtype 
called ``xd_joan`` that you can use with ndarrays.  You may then get and set 
elements out of this array via the wrapper class that was created with cythongen.
For exmaple::

    import numpy as np
    from france.stlcontainer import xd_joan

    x = np.zeros(10, dtype=xd_joan)

    # A scalar array with dtype xd_joan
    x[0]

    # A Joan wrapper object with a copy of the 0th element of the array
    x[0].item()

Note that the underlying memory is a contiguous block of Joans.  The Python
wrappers are regenerated on every getitem() call.  Thus this is more truly an 
array of structs, as opposed to a structured array ;).

Importable Type Systems
==========================

.. note:: This is only starting to be explored

The ``TypeSystem`` class has a mechanism to save and reload via ``dump()`` and
``load()`` methods.  Furthermore, type system variables by the name ``ts`` in 
run control and side car files are all merged together with the default type 
system.  This allows the users of xdress in project-beta to use 
the type system developed in project-alpha with out having to re-expose, 
re-register, or re-parse any project-alpha code at all! 

This is an import-esque mechanism for acting on the type systems themselves.
This should be a huge boon to mutlti-project systems.  You can also choose to 
provide xdress type systems to down stream users and a convenience or favor to 
them. Only pickle and gzip pickle are currently supported, though others may be 
forthcoming.
