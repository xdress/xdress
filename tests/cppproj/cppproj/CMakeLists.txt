# Python extension modules
include_directories(${CPPPROJ_INCLUDE_DIRS})

# Print include dir
get_property(inc_dirs DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
message("-- C_INCLUDE_PATH for ${CMAKE_CURRENT_SOURCE_DIR}: ${inc_dirs}")

set_source_files_properties(${PROJECT_SOURCE_DIR}/cppproj/cppproj_extra_types.pyx
                            PROPERTIES CYTHON_IS_CXX TRUE)
cython_add_module(cppproj_extra_types cppproj_extra_types.pyx)

set_source_files_properties(${PROJECT_SOURCE_DIR}/cppproj/stlc.pyx
                            PROPERTIES CYTHON_IS_CXX TRUE)
cython_add_module(stlc stlc.pyx)

# basics
set_source_files_properties(${PROJECT_SOURCE_DIR}/cppproj/basics.pyx
                            PROPERTIES CYTHON_IS_CXX TRUE)
cython_add_module(basics basics.pyx)
target_link_libraries(basics cppproj_basics)

# basics
set_source_files_properties(${PROJECT_SOURCE_DIR}/cppproj/pybasics.pyx
                            PROPERTIES CYTHON_IS_CXX TRUE)
cython_add_module(pybasics pybasics.pyx)
target_link_libraries(pybasics cppproj_basics)

# discovery
set_source_files_properties(${PROJECT_SOURCE_DIR}/cppproj/discovery.pyx
                            PROPERTIES CYTHON_IS_CXX TRUE)
cython_add_module(discovery discovery.pyx)
target_link_libraries(discovery cppproj_discovery)
