set(CPPPROJ_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${CPPPROJ_INCLUDE_DIRS} 
    PARENT_SCOPE)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# bright
add_library(cppproj_basics basics.cpp)
set_target_properties(cppproj_basics PROPERTIES 
                      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/cppproj/lib")
target_link_libraries(cppproj_basics)
install_lib(cppproj_basics)

# discovery
add_library(cppproj_discovery discovery.cpp)
set_target_properties(cppproj_discovery PROPERTIES 
                      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/cppproj/lib")
target_link_libraries(cppproj_discovery)
install_lib(cppproj_discovery)

# Print include dir
get_property(inc_dirs DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
message("-- C_INCLUDE_PATH for ${CMAKE_CURRENT_SOURCE_DIR}: ${inc_dirs}")
