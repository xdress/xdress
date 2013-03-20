# Python extension modules
include_directories(${XDTEST_INCLUDE_DIRS})

# Print include dir
get_property(inc_dirs DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
message("-- C_INCLUDE_PATH for ${CMAKE_CURRENT_SOURCE_DIR}: ${inc_dirs}")

#include(XdtestIncludes)

set_source_files_properties(${PROJECT_SOURCE_DIR}/xdtest/xdtest_extra_types.pyx
                            PROPERTIES CYTHON_IS_CXX TRUE)
cython_add_module(xdtest_extra_types xdtest_extra_types.pyx)

set_source_files_properties(${PROJECT_SOURCE_DIR}/xdtest/stlcontainers.pyx
                            PROPERTIES CYTHON_IS_CXX TRUE)
cython_add_module(stlcontainers stlcontainers.pyx)

