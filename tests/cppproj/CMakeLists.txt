# Defines the CMake commands/policies
if(APPLE)
    # OS X RPATH needs 2.8.12
    cmake_minimum_required( VERSION 2.8.12 )
    set(CMAKE_MACOSX_RPATH 1)
else(APPLE)
    cmake_minimum_required( VERSION 2.8.5 )
endif(APPLE)

# Set the project name
project( cppproj )

# Make the scripts available in the 'cmake' directory available for the
# 'include()' command, 'find_package()' command.
set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake )

# give an option for whether this should be built as a CI project
option(IS_CI "Is this project being built under continuous integration?" FALSE)

# Use new Python library finder
find_package(PythonInterp REQUIRED)
find_package(PythonLibsNew REQUIRED)
message("-- PYTHON_EXECUTABLE: ${PYTHON_EXECUTABLE}")

# Include the CMake script UseCython.cmake.  This defines add_cython_module().
# Instruction for use can be found at the top of cmake/UseCython.cmake.
include( UseCython )

include(CppprojMacros)

# This makes all the libraries build as SHARED
set(BUILD_SHARED_LIBS true)
message("-- CMake Install Prefix: ${CMAKE_INSTALL_PREFIX}")

# RPATH Settings
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE) 
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
if(APPLE)
    # This is the right thing to do for MacOSX 10.5+ --John
    # Set the install name of *.dylib files to @rpath/libname.dylib
    set(CMAKE_INSTALL_NAME_DIR "@rpath")
    # Set the RPATH for all *.so files to @loader_path/lib (and then put the
    # *.dylib files in a lib directory)
    set(CMAKE_INSTALL_RPATH "@loader_path/lib")
elseif(WIN32)
    if(MSVC)
        # ??? Who knows what to do here?! --Anthony
    elseif(CMAKE_COMPILER_IS_GNUC OR CMAKE_COMPILER_IS_GNUCXX)
        set(CMAKE_INSTALL_RPATH "\$ORIGIN/lib")
    endif(MSVC)
else(APPLE)
    # For linux
    set(CMAKE_INSTALL_RPATH "\$ORIGIN/lib")
endif(APPLE)
message("-- RPATH: ${CMAKE_INSTALL_RPATH}")


# find numpy and include the numpy headers
find_package(Numpy REQUIRED)
include_directories(${NUMPY_INCLUDE_DIR})

# Add some CI-dependent stuff
message("-- Is CI: ${IS_CI}")

# With CMake, a clean separation can be made between the source tree and the
# build tree.  When all source is compiled, as with pure C/C++, the source is
# no-longer needed in the build tree.  However, with pure *.py source, the
# source is processed directly.  To handle this, we reproduce the availability
# of the source files in the build tree.
add_custom_target( ReplicatePythonSourceTree ALL ${CMAKE_COMMAND} -P
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ReplicatePythonSourceTree.cmake
  ${CMAKE_CURRENT_BINARY_DIR}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )

add_subdirectory(src)
add_subdirectory(cppproj)

# Print include dir
get_property(inc_dirs DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
message("-- C_INCLUDE_PATH for ${CMAKE_CURRENT_SOURCE_DIR}: ${inc_dirs}")

message("-- Copying Cython header files.")
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/cppproj/ 
    DESTINATION ${CMAKE_BINARY_DIR}/cppproj/
    FILES_MATCHING PATTERN "*.pxd"
                   PATTERN "lib" EXCLUDE
                   PATTERN "include" EXCLUDE)
