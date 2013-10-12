# Python extension modules
include_directories(${CPROJ_INCLUDE_DIRS})

# Print include dir
get_property(inc_dirs DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
message("-- C_INCLUDE_PATH for ${CMAKE_CURRENT_SOURCE_DIR}: ${inc_dirs}")

cython_add_module(cproj_extra_types cproj_extra_types.pyx)

# basics
cython_add_module(basics basics.pyx)
target_link_libraries(basics cproj_basics)

# basics
cython_add_module(pybasics pybasics.pyx)
target_link_libraries(pybasics cproj_basics)

# discovery
cython_add_module(discovery discovery.pyx)
target_link_libraries(discovery cproj_discovery)
