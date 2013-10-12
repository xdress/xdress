# Python extension modules
include_directories(${CPROJ_INCLUDE_DIRS})

# Print include dir
get_property(inc_dirs DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
message("-- C_INCLUDE_PATH for ${CMAKE_CURRENT_SOURCE_DIR}: ${inc_dirs}")

set_source_files_properties(${PROJECT_SOURCE_DIR}/cproj/cproj_extra_types.pyx
                            PROPERTIES CYTHON_IS_CXX TRUE)
cython_add_module(cproj_extra_types cproj_extra_types.pyx)

set_source_files_properties(${PROJECT_SOURCE_DIR}/cproj/stlc.pyx
                            PROPERTIES CYTHON_IS_CXX TRUE)
cython_add_module(stlc stlc.pyx)

# basics
set_source_files_properties(${PROJECT_SOURCE_DIR}/cproj/basics.pyx
                            PROPERTIES CYTHON_IS_CXX TRUE)
cython_add_module(basics basics.pyx)
target_link_libraries(basics cproj_basics)

# basics
set_source_files_properties(${PROJECT_SOURCE_DIR}/cproj/pybasics.pyx
                            PROPERTIES CYTHON_IS_CXX TRUE)
cython_add_module(pybasics pybasics.pyx)
target_link_libraries(pybasics cproj_basics)

# discovery
set_source_files_properties(${PROJECT_SOURCE_DIR}/cproj/discovery.pyx
                            PROPERTIES CYTHON_IS_CXX TRUE)
cython_add_module(discovery discovery.pyx)
target_link_libraries(discovery cproj_discovery)
