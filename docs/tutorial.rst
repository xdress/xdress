.. _tutorial:

*******************
Tutorial
*******************
At its core, xdress is type system on which code generation utilities are written.
These utilities may be executed via the ``xdress`` command line interface.  This
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
relies solely on the type system.  XDress is governed by a run control file, called
``xdressrc.py`` by default.  This is a pure Python file that should be placed
in the directory where you will run the ``xdress`` command.  A simple stlwrap run
control file would contain the following variables.

**xdressrc.py**::

    package = 'mypack'     # top-level python package name
    packagedir = 'mypack'  # location of the python package

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

    scopatz@ares ~/mypack $ mkdir src
    scopatz@ares ~/mypack $ mkdir mypack
    scopatz@ares ~/mypack $ ls *
    xdressrc.py

    mypack:
    __init__.py

    src:

We would then run xdress to execute stlwrap.  This then generates the following
files:

.. code-block:: bash

    scopatz@ares ~/mypack $ xdress
    generating C++ standard library wrappers & converters
    scopatz@ares ~/mypack $ ls *
    xdressrc.py

    mypack:
    stlcontainers.pxd       stlcontainers.pyx  tests        xdress_extra_types.pxd
    xdress_extra_types.pyx  __init__.pxd       __init__.py

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
        ('A', 'src/hoover.*'),
        ('B', 'src/hoover.*', 'hoover_b'),
        ]

    functions = [('do_nothing_ab', 'src/hoover.*')]

Note that to do this we need only give the construct names -- no signatures need
be specified.  That is the point of API generation!  Also note that we only give
the base file name with the preceding ``src/`` directory and the file extension
(``.cpp``, ``.h``).  Strings passed in here are globbed, so we can be a little lazy. 
Furthermore, the base names of the source and target files need not be the 
same...even for APIs which share the same source file!  We may then run 
xdress normally:

.. code-block:: bash

    scopatz@ares ~/mypack $ xdress
    generating C++ standard library wrappers & converters
    parsing A
    registering A
    parsing B
    registering B
    parsing B
    making cython bindings
    scopatz@ares ~/mypack $ ls *
    xdressrc.py

    build:
    desc.cache

    mypack:
    cpp_hoover.pxd    hoover.pyx    stlcontainers.pxd  xdress_extra_types.pxd
    cpp_hoover_b.pxd  hoover_b.pxd  stlcontainers.pyx  xdress_extra_types.pyx
    hoover.pxd        hoover_b.pyx  tests              __init__.pxd
    __init__.py

    src:
    hoover.cpp  hoover.h  xdress_extra_types.h

Since C/C++ API scraping may be an expensive task for large codes or files,
the descriptions of classes and functions that are generated are stored in the
``build/desc.cache``.  This cache is simply a pickled dictionary that maps
names, source files, and kinds to a hash of the source file and the description.
Thus API elements are not re-described if the source file has not changed.
You may view the contents of a description cache with the ``dumpdesc`` option.

.. code-block:: bash

    scopatz@ares ~/mypack $ xdress --dumpdesc
    {('A', 'src/hoover.cpp', 'class'): ('54a508b1e10845f26d9888a6ad2a470e',
                                        {'attrs': {'y': ('map',
                                                         'int32',
                                                         'float64')},
                                         'methods': {('A', ('x', 'int32', 5)): None,
                                                     ('~A',): None},
                                         'name': 'A',
                                         'namespace': 'hoover',
                                         'parents': None}),
     ('B', 'src/hoover.cpp', 'class'): ('54a508b1e10845f26d9888a6ad2a470e',
                                        {'attrs': {'z': 'int32'},
                                         'methods': {('B',): None,
                                                     ('~B',): None},
                                         'name': 'B',
                                         'namespace': 'hoover',
                                         'parents': ['A']}),
     ('do_nothing_ab', 'src/hoover.cpp', 'func'): ('54a508b1e10845f26d9888a6ad2a470e',
                                                   {'name': 'do_nothing_ab',
                                                    'namespace': 'hoover',
                                                    'signatures': {('do_nothing_ab', ('a', 'A'), ('b', 'B')): 'void'}})}

Be aware that the ``y`` member variable on class ``A`` -- which has type
``map<int, double>`` -- requires that stlwrap tool also have a matching container.
Luckily, we declared ``('map', 'int', 'float')`` in the ``stlcontainers`` list
previously =).

**Once again, it is up to the user to integrate the files created by xdress into their
own build system.**  However, for the above example the following ``setup.py`` file
will work:

**setup.py**:

.. literalinclude:: mypack/setup.py
   :language: py

Or, the following ``CMakeLists.txt`` files will work to build the modules with
`CMake <http://cmake.org>`_.

**CMakeLists.txt**:

.. literalinclude:: mypack/CMakeLists.txt
   :language: cmake

**mypack/CMakeLists.txt**:

.. literalinclude:: mypack/mypack/CMakeLists.txt
   :language: cmake

=============
Code Listings
=============
The following are code listings of the files generated above, since they are too
large to in-line into the tutorial text.  You may also find `this example implemented
in the xdress repo <https://github.com/scopatz/xdress/tree/master/docs/mypack>`_


.. toctree::
    :maxdepth: 4

    mypack/index


=======================
Putting It All Together
=======================
The following is a more complete, realistic example of an xdressrc.py file that
one might run across in a production level environment.

.. code-block:: python

    package = 'mypack'     # top-level python package name
    packagedir = 'mypack'  # location of the python package

    # wrappers for non-standard types (uints, complex)
    extra_types = 'xdress_extra_types'

    # List of C++ standard library container template types
    # to instantiate and wrap with Cython. See the type
    # system documentation for more details.  Note that
    # vectors are wrapped as numpy arrays of the appropriate
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
    # APIs, but will not create the corresponding bindings.  Additionally,
    # if the "xdress.autoall" plugin is enabled, you may also use an
    # asterix (or star) to tell xdress to search the source file for
    # all classes, functions, and/or variables:
    #
    #   ('*', base source filename)
    #   ('*', base source filename, base package filename)
    #   ('*', base source filename, None)
    #
    # This is useful for wrapping larger existing libraries.
    classes = [
        ('FCComp', 'src/fccomp.*'),
        ('EnrichmentParameters', 'src/enrichment_parameters.*'),
        ('Enrichment', 'src/bright_enrichment.*', 'enrichment'),
        ('DontWrap', 'src/bright_enrichment.*', None),
        ('Reprocess', 'src/reprocess.*'),
        ]

    # List of functions to wrap
    functions = [
        ('*', 'src/reprocess.*'),
        ('fillUraniumEnrichmentDefaults', 'src/enrichment_parameters.*'),
        ]

