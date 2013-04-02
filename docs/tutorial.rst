.. _tutorial:

*******************
The XDress Tutorial
*******************
XDress is governed by a run control file called ``xdressrc.py``.  
Place this file in the directory where you will run the ``xdress`` command.
This config file has the following form:

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

