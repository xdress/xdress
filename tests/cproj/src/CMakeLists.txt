set(CPROJ_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${CPROJ_INCLUDE_DIRS} 
    PARENT_SCOPE)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# bright
add_library(cproj_basics basics.cpp)
set_target_properties(cproj_basics PROPERTIES 
                      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/cproj/lib")
target_link_libraries(cproj_basics)
install_lib(cproj_basics)

# discovery
add_library(cproj_discovery discovery.cpp)
set_target_properties(cproj_discovery PROPERTIES 
                      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/cproj/lib")
target_link_libraries(cproj_discovery)
install_lib(cproj_discovery)

# Print include dir
get_property(inc_dirs DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
message("-- C_INCLUDE_PATH for ${CMAKE_CURRENT_SOURCE_DIR}: ${inc_dirs}")
