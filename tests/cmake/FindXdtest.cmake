# - Find XDTEST libraries
# This module finds the libraries corresponding to the XDTEST library, using the Python
# interpreter.
# This code sets the following variables:
#
#  XDTEST_LIBS_FOUND            - have the XDTEST libs been found
#  XDTEST_PREFIX                - path to the XDTEST installation
#  XDTEST_LIBS_DIR              - path to the XDTEST libs dir
#  XDTEST_INCLUDE_DIR           - path to where XDTEST header files are
#
# To use XDTEST add the following lines to your main CMakeLists.txt file:
# 
#   # Find the XDTEST installation
#   find_package(XDTEST REQUIRED)
#   link_directories(${XDTEST_LIBS_DIR})
#   include_directories(${XDTEST_INCLUDE_DIR})
#
# Enjoy!

# Use the Python interpreter to find the libs.
if(XDTEST_FIND_REQUIRED)
    find_package(PythonInterp REQUIRED)
else()
    find_package(PythonInterp)
endif()

if(NOT PYTHONINTERP_FOUND)
    set(XDTEST_LIBS_FOUND FALSE)
    return()
endif()

execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
    "from XDTEST import XDTEST_config as pc; print(pc.prefix); print(pc.lib); print(pc.includes)"
    RESULT_VARIABLE _XDTEST_SUCCESS
    OUTPUT_VARIABLE _XDTEST_VALUES
    ERROR_VARIABLE _XDTEST_ERROR_VALUE
    OUTPUT_STRIP_TRAILING_WHITESPACE)

if(NOT _XDTEST_SUCCESS MATCHES 0)
    if(XDTEST_FIND_REQUIRED)
        message(FATAL_ERROR
            "XDTEST config failure:\n${_XDTEST_ERROR_VALUE}")
    endif()
    set(XDTEST_LIBS_FOUND FALSE)
    return()
endif()

# Convert the process output into a list
string(REGEX REPLACE ";" "\\\\;" _XDTEST_VALUES ${_XDTEST_VALUES})
string(REGEX REPLACE "\n" ";" _XDTEST_VALUES ${_XDTEST_VALUES})
list(GET _XDTEST_VALUES 0 XDTEST_PREFIX)
list(GET _XDTEST_VALUES 1 XDTEST_LIBS_DIR)
list(GET _XDTEST_VALUES 2 XDTEST_INCLUDE_DIR)

# Make sure all directory separators are '/'
string(REGEX REPLACE "\\\\" "/" XDTEST_PREFIX ${XDTEST_PREFIX})
string(REGEX REPLACE "\\\\" "/" XDTEST_LIBS_DIR ${XDTEST_LIBS_DIR})
string(REGEX REPLACE "\\\\" "/" XDTEST_INCLUDE_DIR ${XDTEST_INCLUDE_DIR})


MARK_AS_ADVANCED(
  XDTEST_LIBS_DIR
  XDTEST_INCLUDE_DIR
)

# We use XDTEST_INCLUDE_DIR, XDTEST_LIBS_DIR and XDTEST_PREFIX for the
# cache entries because they are meant to specify the location of a single
# library. We now set the variables listed by the documentation for this
# module.
SET(XDTEST_PREFIX      "${XDTEST_PREFIX}")
SET(XDTEST_LIBS_DIR    "${XDTEST_LIBS_DIR}")
SET(XDTEST_INCLUDE_DIR "${XDTEST_INCLUDE_DIR}")


find_package_message(XDTEST "Found XDTEST: ${XDTEST_PREFIX}" "${XDTEST_PREFIX}")
