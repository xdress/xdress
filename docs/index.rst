XDress
==================================
XDress is an automatic wrapper generator for C/C++ written in pure Python. Currently,
xdress may generate Python binding (via Cython) for C++ classes and in-memory 
wrappers for C++ standard library containers (sets, vectors, maps).  In the future, 
functions and cyclus bindings will be supported.  

The main enabling feature of xdress is a dynamic type system that was designed with 
the purpose of API generation in mind.

XDress currently has the following external dependencies:

   #. `Cython <http://cython.org/>`_
   #. `NumPy <http://numpy.scipy.org/>`_
   #. `GCC-XML <http://www.gccxml.org/HTML/Index.html>`_

The source code for xdress may be found at the 
`GitHub project site <http://github.com/scopatz/xdress>`_.
Or you may simply clone the development branch using git::

    git clone git://github.com/scopatz/xdress.git

--------
Contents
--------

.. toctree::
    :maxdepth: 1

    tutorial
    libref/index

=============
Helpful Links
=============
	
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
