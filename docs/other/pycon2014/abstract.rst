PyCon 2014 Abstract
===================

:title: Beautypeful XDress

:category: Science

:duration: 30 min

:description: XDress is an automatic wrapper generator for C/C++ written in 
    pure Python. It can generate Python bindings (via Cython) for classes, 
    structs, functions, and certain variables types. It can also create 
    in-memory wrappers for C++ sets, vectors, and maps. This is because it 
    can generate custom NumPy dtypes. This talk will go over the "zen of xdress" 
    and teach you how to generate your own wrappers.

:audience: Anyone who has to or wants to expose a C/C++ API to Python.

:Python level: Intermediate

:objectives: To teach people basic type theory concepts, that wrapping C/C++ 
    extension modules does not have to be painful, and that the results can be very 
    beautiful.

:detailed abstract: XDress is an automatic wrapper generator for C/C++ written 
    in pure Python. Currently, xdress may generate Python bindings (via Cython) for 
    C++ classes & functions and in-memory wrappers for C++ standard library 
    containers (sets, vectors, maps). In the future, other tools and bindings 
    will be supported.

    The main enabling feature of xdress is a dynamic type system that was designed 
    with the purpose of API generation in mind.  This type system provides a 
    canonical abstraction of various kinds of types: base types (int, str, float, 
    non-templated classes), refined types (even or odd ints, strings containing the 
    letter ‘a’), and dependent types (templates such arrays, maps, sets, vectors).
    This canonical form is itself hashable, being comprised only of strings, ints, 
    and tuples.

    On top of this type system, xdress provides a tool for auto-generating classes
    which are views into template instantiations of C++ standard library maps and sets.
    Additionally, this tool also creates custom numpy dtypes for any C++ type, class
    or struct.  This allows the user to have numpy array views into C++ vectors.

    Furthermore, xdress also has a tool which inspects a C++ code base and 
    automatically generates Cython wrappers for all user-specified classes and 
    functions.  This significantly eases the burden of supporting mixed language
    projects.

    Other powerful features include:

    * C++ template class & function support
    * Automatic docstring generation via dOxygen
    * Wrapped names may be different than source names, allowing automatic 
      PEP-8-ification during C/C++ to Python translation
    * First class support for C via pycparser
    * Python 3 support, by popular demand!

    The above code generators, however, are just the beginning.  The xdress type 
    system is flexible and powerful enough to engender a suite of other tools which
    take advantage of less obvious features.  For example, an automatic verification 
    & validation utility could take advantage of refinement type predicate functions 
    to interdict parameter constraints into the API right under the users nose!

    This talk will focus on xdress's type system and its use cases.  XDress is 
    licensed under 2-clause BSD. 

:outline: 

* Big Idea
* Motivation 
* What XDress Is and Is Not
* Type System
    * Kinds of Types
    * Base Types
    * Dependent Types
    * Refined Types
* C/C++ API Descriptions
* C++ Class Example
* C++ Standard Library Container Wrappers
* NumPy Dtype Generation
* Plugins
* Neat Concepts for Future Work
* Questions

:additional notes:  The main website is http://xdress.org/

    An earlier version of this talk which used an earlier version of xdress 
    was presented to SciPy 2013.  See this page for the slides and the video. 
    http://xdress.org/other/scipy2013/index.html

:additional requirements: none

:recording release: yes
