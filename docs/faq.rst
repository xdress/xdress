==========================
Frequently Asked Questions
==========================
1. Why not use an existing solution (eg, SWIG)?

    Their type systems don't support run-time, user provided refinement types,
    and thus are unsuited for verification & validation use cases that often
    arise in computational science.

    Furthermore, they tend to not handle C++ dependent types well (i.e. vector<T>
    does not come back as a np.view(..., dtype=T)).

2. Why GCC-XML and not Clang's AST?

    I tried using Clang's AST (and the remnants of a broken visitor class remain
    in the code base).  However, the official Clang AST Python bindings lack
    support for template argument types.  This is a really big deal. Other C++ ASTs
    may be supported in the future -- including Clang's.

3. I run xdress and it creates these files, now what?!

    It is your job to integrate the files created by xdress into your build system.

