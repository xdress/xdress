XDress
==================================
XDress is an automatic wrapper generator for C/C++ written in pure Python. Currently,
xdress may generate Python bindings (via Cython) for C++ classes & functions 
and in-memory wrappers for C++ standard library containers (sets, vectors, maps).  
In the future, other tools and bindings will be supported.  

The main enabling feature of xdress is a dynamic type system that was designed with 
the purpose of API generation in mind.

XDress currently has the following external dependencies:

    #. `Cython <http://cython.org/>`_
    #. `NumPy <http://numpy.scipy.org/>`_
    #. `pycparser <https://bitbucket.org/eliben/pycparser>`_, optional for C
    #. `GCC-XML <http://www.gccxml.org/HTML/Index.html>`_, optional for C++
    #. `lxml <http://lxml.de/>`_, optional (but nice!)

The source code for xdress may be found at the 
`GitHub project site <http://github.com/scopatz/xdress>`_.
Or you may simply clone the development branch using git::

    git clone git://github.com/scopatz/xdress.git

`Go here for the latest version of the docs! <http://xdress.org/latest>`_

--------
Contents
--------

.. toctree::
    :maxdepth: 1

    tutorial
    libref/index
    previous/index
    other/index

===============
Examples of Use
===============
To see examples of xdress in action (and sample run control files), here are a
few places to look:

* `xdress/tests <https://github.com/scopatz/xdress/tree/master/tests>`_: This is 
  a fully functioning sample project which uses xdress locally (no install needed).
* `PyNE <http://pynesim.org/>`_: This uses xdress to generate STL container wrappers.
* `Bright <http://bright-dev.github.com/>`_: This uses xdress to automatically
  wrap a suite of interacting C++ class.  This was the motivating use case for the
  xdress project.

======
Author
======
This tool was written by `Anthony Scopatz <http://scopatz.com/>`_, who had many
type system discussions with John Bachan over coffee at the Div school, and was
polished up and released under the encouragement of Christopher Jordan-Squire at
`PyCon 2013 <https://us.pycon.org/2013/>`_.

==========
Contact Us
==========
If you have questions or comments, please send them to the mailing list
xdress@googlegroups.com or contact the author directly or open an issue on
GitHub.

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
* `Mailing list <mailto:xdress@googlegroups.com>`_
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

