Using libclang
============================
libclang is an extension module which allows XDress to use the excellent Clang
compiler for parsing C and C++ source files.

**NOTE**: libclang is an official part of the Clang software distribution. As of
XDress version 0.4, the libclang used by XDress is a fork of the source code
in the main Clang distribution (due to missing features in the official version).
Patches have been submitted upstream, and the fork will be retired once those
changes are merged.


============
Installation
============

 * Start by downloading a source tarball for a current `Clang distribution <http://llvm.org/releases/>`_
 * We highly recommend following the detailed `build instructions <http://clang.llvm.org/get_started.html>`_ on the LLVM website.
 * Once you've built LLVM/Clang, install it in a location which is appropriate for
   your system. If that location is not part of your executable path, you will need
   to set the `LLVM_CONFIG` environment variable to point to the `llvm-config` program.
   XDress will check the environment variable before searching for an `llvm-config`
   program in your path.
 * Build XDress
 * Enjoy!


===========
OS X Issues
===========
If you are building on a recent version of OS X, you must take care to use the
same C++ library for both LLVM/Clang and libclang in XDress. Mixing libc++ and
libstdc++ will cause link errors and you will not be able to import libclang.

Because LLVM/Clang will likely use libc++ by default, the easiest solution is
to add `-stdlib=libc++` to the compiler flags for XDress. This might not be an
option depending on how your Python installation is configured. If you get an
error which looks like this:

`clang: error: invalid deployment target for -stdlib=libc++ (requires OS X 10.7 or later)`

Then you will need to build LLVM/Clang against libstdc++ in order to build XDress.

Building LLVM/Clang with libstdc++
----------------------------------
Building LLVM/Clang with libstdc++ is fairly straightforward. All you need to
do is set a `CPPFLAGS` environment variable before running the included
`configure` script::

    # "/usr/include/c++/4.2.1/" should be replaced with the location of the 
    # GNU C++ headers on your system.
    $ export CPPFLAGS="-I/usr/include/c++/4.2.1/ -stdlib=libstdc++"
    $ ../configure

You can confirm that you are linked against libstdc++ by running the following
command in the output library build directory::

    $ strings libLLVMSupport.a | grep __ZN4llvm27install_fatal_error_handler | c++filt

That command can be run fairly early in the build process because libLLVMSupport
is one of the first libraries which gets build. Here are the two possible results:

* libstdc++: `llvm::install_fatal_error_handler(void (*)(void*, std::string const&, bool), void*)`
* libc++: `llvm::install_fatal_error_handler(void (*)(void*, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > const&, bool), void*)`
