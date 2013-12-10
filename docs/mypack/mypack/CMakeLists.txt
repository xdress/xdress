if( NOT NUMPY_INCLUDE_DIR )
  find_package( PythonInterp )
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE _numpy_include
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  find_path( NUMPY_INCLUDE_DIR numpy/arrayobject.h
    HINTS ${_numpy_include} )
endif()
include_directories( ${NUMPY_INCLUDE_DIR}  )

set( cxx_pyx_files
  hoover_b.pyx
  hoover.pyx
  stlcontainers.pyx
  mypack_extra_types.pyx
  )
set_source_files_properties( ${cxx_pyx_files}
  PROPERTIES CYTHON_IS_CXX TRUE )

cython_add_module( mypack_extra_types mypack_extra_types.pyx)
cython_add_module( stlcontainers stlcontainers.pyx )
cython_add_module( hoover hoover.pyx ../src/hoover.cpp )
cython_add_module( hoover_b hoover_b.pyx ../src/hoover.cpp )
