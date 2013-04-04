.. _tutorial:

*******************
The XDress Tutorial
*******************
At its core, xdress is type system on which code generation utilites are written.
These utilites may be executed via the ``xdress`` command line interface.  This 
tutorial presents a brief walk through of the type system, the STL container wrapper
generator, and the C/C++ API wrapper generator.

===============
The Type System
===============
XDress provides an interface for denoting, describing, and converting
between various data types and the types coming from various systems.  This is
achieved by providing canonical abstractions of various kinds of types:

* Base types (int, str, float, non-templated classes)
* Refined types (even or odd ints, strings containing the letter 'a')
* Dependent types (templates such arrays, maps, sets, vectors)

All types are known by their name (a string identifier) and may be aliased with 
other names.  However, the string id of a type is not sufficient to fully describe
most types.  The system here implements a canonical form for all kinds of types.
This canonical form is itself hashable, being comprised only of strings, ints, 
and tuples.

These canonical forms are covered in detail in the :ref:`xdress_typesystem` 
documentation.  However, what are more important and useful from an end-user 
perspective are the short hand notations that should be used by mortals::

    # Base type are just their names
    'str'
    'int32'
    'float64'

    # Aliases may also be used
    'f4'     # -> 'float32'
    'uint'   # -> 'uint32'
    'float'  # -> 'float64'

    # Length-1 tuples expand to have scalar predicates
    ('int32',)  # -> ('int32', 0)

    # Refinement types may be listed by name only
    'posint'  # -> ('int32', 'posint')

    # Templates are given within tuples
    ('set', 'str')
    ('map', 'i4', 'float')

    # As are dependencies values
    ('intrange', 1, 2)
    ('range', 'int32', 1, 2)

    # And any combination of the above!
    (('map', 'posint', ('set', ('intrange', 1, 2))),)

========================
STL Containers (stlwrap)
========================
The first tool we discuss is the C++ STL container wrapper generator.  This tool 
relies soley on the type system.  XDress is governed by a run control file, called 
``xdressrc.py`` by default.  This is a pure Python file that should be placed 
in the directory where you will run the ``xdress`` command.  A simple stlwrap run
control file would contain the following variables.

**xdressrc.py**::

    package = 'mypack'     # top-level python package name
    packagedir = 'mypack'  # loation of the python package
    sourcedir = 'src'      # location of C/C++ source
    
    stlcontainers = [
        ('vector', 'str'),
        ('set', 'uint'),
        ('map', 'int', 'float'),
        ]

    # will be used later, but need to be present now
    classes = []
    functions = []

This would tell ``xdress`` to generate a numpy dtype for ``std::string`` (to be used
with normal numpy arrays), a wrapper class for ``std::set<unsigned int>``,  and a
wrapper class for ``std::map<int, double>``.  Suppose we started with an empty 
project,

.. code-block:: bash

    scopatz@ares ~/mypack $ ls *
    xdressrc.py

    mypack:
    __init__.py

    src:

We would then run xdress with the "no-cython" option to only execute stlwrap. 
This then generates the following files:

.. code-block:: bash

    scopatz@ares ~/mypack $ xdress --no-cython
    generating C++ standard library wrappers & converters
    scopatz@ares ~/mypack $ ls *
    xdressrc.py

    mypack:
    stlcontainers.pxd       stlcontainers.pyx  tests  xdress_extra_types.pxd  
    xdress_extra_types.pyx  __init__.py

    src:
    xdress_extra_types.h

It is then our job to pass these files off to Cython and a C++ compiler, typically 
as part of a larger build system.

================================
C/C++ API Generation (cythongen)
================================
The next tool that is built off of the xdress type system may be used for 
automatically creating Python wrappers of C/C++ APIs.  This requires that the user
has GCC-XML and lxml installed are their system.  Now suppose we had some C++ code
living in the ``src/`` directory.

**src/hoover.h**:

.. literalinclude:: mypack/src/hoover.h
   :language: cpp

**src/hoover.cpp**:

.. literalinclude:: mypack/src/hoover.cpp
   :language: cpp

To tell xdress that we what to wrap the A & B classes and the do nothing function, 
we simply need to tell xdress that they live in hoover.  We do this by adding to the
``classes`` and ``functions`` lists in the run control file.  

**xdressrc.py**::

    classes = [
        ('A', 'hoover'),
        ('B', 'hoover'),
        ]

    functions = [('do_nothing_ab', 'hoover')]

Note that do do this we need only give the construct names -- no signatures need 
be specified.  That is the point of API generation!  Also note that we only give
the base file name without the preceeding ``src/`` directory or the file extension
(``.cpp``, ``.h``).  We may then run xdress normally:

.. code-block:: bash


=======================
Putting It All Together
=======================
The foillowing is a more complete, realistic example of an xdressrc.py file that 
one might run across in a production level environment.

.. code-block:: python

    package = 'mypack'     # top-level python package name
    packagedir = 'mypack'  # loation of the python package
    sourcedir = 'src'      # location of C/C++ source

    # wrappers for non-standard types (uints, complex)
    extra_types = 'xdress_extra_types'  

    # List of C++ standard library container template types 
    # to instantiate and wrap with Cython. See the type 
    # system documentation for more details.  Note that 
    # vectors are wrapped as numpy arrays of the approriate
    # type.  If the type has no corresponding primitive C++
    # type, then a new numpy dtype is created to handle it.
    # For example, this allows the wrapping of vector< vector<int> >
    # as an np.array(..., dtype=xd_vector_int).
    stlcontainers = [
        ('vector', 'str'),
        ('vector', 'int32'),
        ('vector', 'complex'),
        ('vector', 'float32'),
        ('vector', 'float64'),
        ('vector', ('vector', 'float64')),
        ('set', 'int'),
        ('set', 'str'),
        ('set', 'uint'),
        ('set', 'char'),
        ('map', 'str', 'str'),
        ('map', 'str', 'int'),
        ('map', 'int', 'str'),
        ('map', 'str', 'uint'),
        ('map', 'uint', 'str'),
        ('map', 'uint', 'uint'),
        ('map', 'str', 'float'),
        ('map', 'int', 'int'),
        ('map', 'int', 'bool'),
        ('map', 'int', 'char'),
        ('map', 'int', 'float'),
        ('map', 'uint', 'float'),
        ('map', 'int', 'complex'),
        ('map', 'int', ('set', 'int')),
        ('map', 'int', ('set', 'str')),
        ('map', 'int', ('set', 'uint')),
        ('map', 'int', ('set', 'char')),
        ('map', 'int', ('vector', 'str')),
        ('map', 'int', ('vector', 'int')),
        ('map', 'int', ('vector', 'uint')),
        ('map', 'int', ('vector', 'char')),
        ('map', 'int', ('vector', 'bool')),
        ('map', 'int', ('vector', 'float')),
        ('map', 'int', ('vector', ('vector', 'float64'))),
        ('map', 'int', ('map', 'int', 'bool')),
        ('map', 'int', ('map', 'int', 'char')),
        ('map', 'int', ('map', 'int', 'float')),
        ('map', 'int', ('map', 'int', ('vector', 'bool'))),
        ('map', 'int', ('map', 'int', ('vector', 'char'))),
        ('map', 'int', ('map', 'int', ('vector', 'float'))),
        ]

    # name of the C++ standard library container module in
    # the packagedir
    #stlcontainers_module = 'stlcontainers'  # default value

    # List of classes to wrap.  These may take one of the following 
    # forms:
    #
    #   (classname, base source filename)
    #   (classname, base source filename, base package filename)
    #   (classname, base source filename, None)
    #
    # In the first case, the base source filename will be used as 
    # the base package name as well. In the last case, a None value
    # will register this class for the purpose of generating other 
    # APIs, but will not create the cooresponding bindings.
    classes = [
        ('FCComp', 'fccomp'), 
        ('EnrichmentParameters', 'enrichment_parameters'), 
        ('Enrichment', 'bright_enrichment', 'enrichment'), 
        ('DontWrap', 'bright_enrichment', None), 
        ('Reprocess', 'reprocess'), 
        ]

    # List of functions to wrap
    functions = [
        ('myfunc', 'reprocess'),
        ('fillUraniumEnrichmentDefaults', 'enrichment_parameters'),
        ]

