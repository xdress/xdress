set(XDTEST_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${XDTEST_INCLUDE_DIRS} 
    PARENT_SCOPE)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# bright
add_library(bright_bright bright.cpp)
set_target_properties(bright_bright PROPERTIES 
                      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/xdtest/lib")
target_link_libraries(bright_bright)
install_lib(bright_bright)

# fccomp
add_library(bright_fccomp fccomp.cpp)
set_target_properties(bright_fccomp PROPERTIES 
                      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/xdtest/lib")
target_link_libraries(bright_fccomp bright_bright)
install_lib(bright_fccomp)

# enrichment parameters
add_library(bright_enrichment_parameters enrichment_parameters.cpp)
set_target_properties(bright_enrichment_parameters PROPERTIES 
                      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/xdtest/lib")
target_link_libraries(bright_enrichment_parameters bright_bright)
install_lib(bright_enrichment_parameters)

# enrichment
add_library(bright_enrichment bright_enrichment.cpp)
set_target_properties(bright_enrichment PROPERTIES 
                      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/xdtest/lib")
target_link_libraries(bright_enrichment bright_bright bright_fccomp 
                      bright_enrichment_parameters)
install_lib(bright_enrichment)

# reprocess
add_library(bright_reprocess reprocess.cpp)
set_target_properties(bright_reprocess PROPERTIES 
                      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/xdtest/lib")
target_link_libraries(bright_reprocess bright_bright bright_fccomp)
install_lib(bright_reprocess)

# Print include dir
get_property(inc_dirs DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
message("-- C_INCLUDE_PATH for ${CMAKE_CURRENT_SOURCE_DIR}: ${inc_dirs}")
