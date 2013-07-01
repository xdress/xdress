cmake_minimum_required( VERSION 2.8 )

project( mypack )

# The files in the cmake directory must also be included in your project
# until they are merged into upstream CMake.
set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake )

include( UseCython )

# With CMake, a clean separation can be made between the source tree and the
# build tree.  When all source is compiled, as with pure C/C++, the source is
# no-longer needed in the build tree.  However, with pure *.py source, the
# source is processed directly.  To handle this, we reproduce the availability
# of the source files in the build tree.
add_custom_target( ReplicatePythonSourceTree ALL ${CMAKE_COMMAND} -P
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ReplicatePythonSourceTree.cmake
  ${CMAKE_CURRENT_BINARY_DIR}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )

include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/src )

enable_testing()
find_file( NOSETESTS_EXECUTABLE nosetests )
add_test( nosetests "${NOSETESTS_EXECUTABLE}" -v --with-xunit )

add_subdirectory( mypack )
