// Backwards compatibility utilities for use with libclang in xdress

#include "clang/Basic/Version.h"

#define CLANG_VERSION_LT(major,minor) \
  (   defined(CLANG_VERSION_MAJOR) \
   && (    CLANG_VERSION_MAJOR<(major) \
       || (CLANG_VERSION_MAJOR==(major) && CLANG_VERSION_MINOR<(minor))))

extern void XDRESS_FATAL(const char* message);

#if CLANG_VERSION_LT(3,3)
template<class T> static inline T* XDRESS_CONST_CAST(const T* x) {
  return const_cast<T*>(x);
}
#else
template<class T> static inline const T* XDRESS_CONST_CAST(const T* x) {
  return const_cast<T*>(x);
}
#endif
