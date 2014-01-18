A modified copy of libclang inside xdress
================================

The clang frontend of xdress needs access to several features of clang not
exposed by stock libclang, the C and Python bindings to clang.  The necessary
libclang modifications have not yet been merged into clang upstream, so we
include a copy of libclang as part of xdress.

DO NOT MODIFY THIS COPY DIRECTLY!  All changes to libclang should be made in
the xdress copy of clang, hosted on github at https://github.com/xdress/clang/tree/xdress.

Once a change has been made to an llvm+clang checkout, it can be copied over
to xdress by running

    cd xdress/xdress/clang
    ./sync <path-to-your-llvm-checkout>

This should be the only way our clang copy is ever modified.  The eventual goal
is to merge these upstream, and this copy can go away once they show up in a
clang version.  Email xdress@googlegroups.com for questions about the situation.
