XDress
==================================
XDress is an automatic wrapper generator for C/C++ written in pure Python. Currently,
xdress may generate Python bindings (via Cython) for C++ classes & functions 
and in-memory wrappers for C++ standard library containers (sets, vectors, maps).  
In the future, other tools and bindings will be supported.  

The main enabling feature of xdress is a dynamic type system that was designed with 
the purpose of API generation in mind.

`Go here for the latest version of the docs! <http://xdress.org/latest>`_

:ref:`Go here for previous versions of the code & documentation. <previous_versions>`

--------
Contents
--------

.. toctree::
    :maxdepth: 1

    tutorial
    advtut
    libref/index
    rcdocs
    previous/index
    other/index
    faq
    authors

============
Installation
============
Since xdress is pure Python code, the ``pip`` or ``easy_install`` may be used
to grab and install the code::

    $ pip install xdress

    $ easy_install xdress


The source code repository for xdress may be found at the 
`GitHub project site <http://github.com/scopatz/xdress>`_.
You may simply clone the development branch using git::

    git clone git://github.com/xdress/xdress.git

Also, if you wish to have the optional BASH completion, please add the 
following lines to your ``~/.bashrc`` file::

    # Enable completion for xdress
    eval "$(register-python-argcomplete xdress)"

============
Dependencies
============
XDress currently has the following external dependencies,

*Run Time:*

    #. `pycparser <https://bitbucket.org/eliben/pycparser>`_, optional for C
    #. `GCC-XML <http://www.gccxml.org/HTML/Index.html>`_, optional for C++
    #. `dOxygen <http://www.doxygen.org/>`_, optional for docstrings
    #. `lxml <http://lxml.de/>`_, optional (but nice!)
    #. `argcomplete <https://argcomplete.readthedocs.org/en/latest/>`_, optional for BASH completion

*Compile Time:*

    #. `Cython <http://cython.org/>`_
    #. `NumPy <http://numpy.scipy.org/>`_

*Test Time:*

    #. `nose <https://nose.readthedocs.org/en/latest/>`_
    #. `CMake <http://www.cmake.org/>`_, optional for integration tests

===============
Examples of Use
===============
To see examples of xdress in action (and sample run control files), here are a
few places to look:

* `xdress/tests/cproj <https://github.com/xdress/xdress/tree/master/tests/cproj>`_: 
  This is a fully functioning sample C project which uses xdress locally.
* `xdress/tests/cppproj <https://github.com/xdress/xdress/tree/master/tests/cppproj>`_: 
  This is a fully functioning sample C++ project which uses xdress locally.
* `PyNE <http://pynesim.org/>`_: This uses xdress to generate STL container wrappers.
* `Bright <http://bright-dev.github.com/>`_: This uses xdress to automatically
  wrap a suite of interacting C++ class.  This was the motivating use case for the
  xdress project.

=======
Testing
=======
XDress has two major test types: unit tests which test library functionality and 
integration tests which test the command line tool, the parsers, compilers, etc.  
The unit tests are generally fast while the integration are slower.  From the 
`tests/` directory you may use nose to run the tests together or individually::

    # Go into the tests dir
    $ cd tests

    # Run just the unit tests
    tests $ nosetests -a unit

    # Run just the integration tests
    tests $ nosetests -a integration

    # Run all of the tests together
    tests $ nosetests

Note that the integration tests require CMake in order to build the sample
projects.

==========
Contact Us
==========
If you have questions or comments, please sign up for the the mailing list
at https://groups.google.com/forum/#!forum/xdress and send an email to
xdress@googlegroups.com. Alternatively, please contact the authors directly or 
open an issue on GitHub.

============
Contributing
============
We highly encourage contributions to xdress!  If you would like to contribute, 
it is as easy as forking the repository on GitHub, making your changes, and 
issuing a pull request.  If you have any questions about this process don't 
hesitate to ask the mailing list (xdress@googlegroups.com).

=============
Helpful Links
=============

* `Documentation <http://xdress.org>`_
* `Mailing list website <https://groups.google.com/forum/#!forum/xdress>`_
* `Mailing list address <mailto:xdress@googlegroups.com>`_
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

