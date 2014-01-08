// Backwards compatibility macros for use with libclang in xdress

#define CLANG_VERSION_LT(major,minor) \
  (CLANG_VERSION_MAJOR<(major) || (CLANG_VERSION_MAJOR==(major) && CLANG_VERSION_MINOR<(minor)))
