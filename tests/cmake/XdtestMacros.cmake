# set platform preprocessor macro
set(XDTEST_PLATFORM "__${CMAKE_SYSTEM_NAME}__")
if(APPLE)
    set(XDTEST_PLATFORM "__APPLE__")
elseif(WIN32)
    if(MSVC)
        set(XDTEST_PLATFORM "__WIN_MSVC__")
    elseif(CMAKE_COMPILER_IS_GNUC OR CMAKE_COMPILER_IS_GNUCXX)
        set(XDTEST_PLATFORM "__WIN_GNUC__")
    endif(MSVC)
else(APPLE)
    set(XDTEST_PLATFORM "__LINUX__")
endif(APPLE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D${XDTEST_PLATFORM}")
message("-- XDTEST platform defined as: ${XDTEST_PLATFORM}")

macro( add_lib_to_XDTEST _name _source )
  # add the library
  add_library(${_name}             ${_source})
  # add it to the list of XDTEST libraries
  set(XDTEST_LIBRARIES ${XDTEST_LIBRARIES} ${_name})
endmacro()

macro( install_lib _name )
  # install it
  set(lib_type LIBRARY)
  if(WIN32)
    set(lib_type RUNTIME)
  endif(WIN32)
  install(TARGETS ${_name} ${lib_type} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endmacro()


macro( print_logo )
  set(cat_prog cat)
  if(WIN32)
    set(cat_prog type)
  endif(WIN32)
  execute_process(COMMAND ${cat_prog} ${PROJECT_SOURCE_DIR}/cmake/logo.txt OUTPUT_VARIABLE variable)
  message("${variable}")
endmacro()
