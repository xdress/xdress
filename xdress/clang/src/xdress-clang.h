// Backwards compatibility utilities for use with libclang in xdress
#ifndef XDRESS_CLANG_H
#define XDRESS_CLANG_H

#include "clang/Basic/Version.h"

#define CLANG_VERSION_GE(major,minor) \
  (   !defined(CLANG_VERSION_MAJOR) \
   || (    CLANG_VERSION_MAJOR>(major) \
       || (CLANG_VERSION_MAJOR==(major) && CLANG_VERSION_MINOR>=(minor))))

#define CLANG_VERSION_EQ(major,minor) \
  (CLANG_VERSION_GE(major,minor) && !CLANG_VERSION_GE(major,minor+1))

extern void XDRESS_FATAL(const char* message);

#if CLANG_VERSION_GE(3,3)
template<class T> static inline const T* XDRESS_CONST_CAST(const T* x) {
  return x;
}
#else
template<class T> static inline T* XDRESS_CONST_CAST(const T* x) {
  return const_cast<T*>(x);
}
#endif

#if CLANG_VERSION_GE(3,5)
#define getResultType           getReturnType
#define getResultLoc            getReturnLoc
#define getResultTypeSourceInfo getReturnTypeSourceInfo
#endif

#endif // XDRESS_CLANG_H
