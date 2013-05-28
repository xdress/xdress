# Python extension modules
include_directories(${XDTEST_INCLUDE_DIRS})

# Print include dir
get_property(inc_dirs DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
message("-- C_INCLUDE_PATH for ${CMAKE_CURRENT_SOURCE_DIR}: ${inc_dirs}")

#include(XdtestIncludes)

set_source_files_properties(${PROJECT_SOURCE_DIR}/xdtest/xdtest_extra_types.pyx
                            PROPERTIES CYTHON_IS_CXX TRUE)
cython_add_module(xdtest_extra_types xdtest_extra_types.pyx)

#add_library(new_inplace new_inplace.cpp)
#set_target_properties(new_inplace PROPERTIES
#                      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/xdtest/lib")
#target_link_libraries(new_inplace)
#install_lib(new_inplace)

set_source_files_properties(${PROJECT_SOURCE_DIR}/xdtest/xdstlc.pyx
                            PROPERTIES CYTHON_IS_CXX TRUE)
cython_add_module(xdstlc xdstlc.pyx)

set_source_files_properties(${PROJECT_SOURCE_DIR}/xdtest/typeconverters.pyx
                            PROPERTIES CYTHON_IS_CXX TRUE)
cython_add_module(typeconverters typeconverters.pyx)

# device
cython_add_module(pydevice pydevice.pyx)
target_link_libraries(pydevice cdevice)

# fccomp
set_source_files_properties(${PROJECT_SOURCE_DIR}/xdtest/fccomp.pyx
                            PROPERTIES CYTHON_IS_CXX TRUE)
cython_add_module(fccomp fccomp.pyx)
target_link_libraries(fccomp bright_fccomp)

# enrichment parameters
set_source_files_properties(${PROJECT_SOURCE_DIR}/xdtest/enrichment_parameters.pyx
                            PROPERTIES CYTHON_IS_CXX TRUE)
cython_add_module(enrichment_parameters enrichment_parameters.pyx)
target_link_libraries(enrichment_parameters bright_enrichment_parameters)

# enrichment
set_source_files_properties(${PROJECT_SOURCE_DIR}/xdtest/enrichment.pyx
                            PROPERTIES CYTHON_IS_CXX TRUE)
cython_add_module(enrichment enrichment.pyx)
target_link_libraries(enrichment bright_enrichment)

# reprocess
set_source_files_properties(${PROJECT_SOURCE_DIR}/xdtest/reprocess.pyx
                            PROPERTIES CYTHON_IS_CXX TRUE)
cython_add_module(reprocess reprocess.pyx)
target_link_libraries(reprocess bright_reprocess)

