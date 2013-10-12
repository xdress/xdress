# set platform preprocessor macro
set(CPROJ_PLATFORM "__${CMAKE_SYSTEM_NAME}__")
if(APPLE)
    set(CPROJ_PLATFORM "__APPLE__")
elseif(WIN32)
    if(MSVC)
        set(CPROJ_PLATFORM "__WIN_MSVC__")
    elseif(CMAKE_COMPILER_IS_GNUC OR CMAKE_COMPILER_IS_GNUCXX)
        set(CPROJ_PLATFORM "__WIN_GNUC__")
    endif(MSVC)
else(APPLE)
    set(CPROJ_PLATFORM "__LINUX__")
endif(APPLE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D${CPROJ_PLATFORM}")
message("-- CPROJ platform defined as: ${CPROJ_PLATFORM}")

macro( add_lib_to_CPROJ _name _source )
  # add the library
  add_library(${_name}             ${_source})
  # add it to the list of CPROJ libraries
  set(CPROJ_LIBRARIES ${CPROJ_LIBRARIES} ${_name})
endmacro()

macro( install_lib _name )
  # install it
  set(lib_type LIBRARY)
  if(WIN32)
    set(lib_type RUNTIME)
  endif(WIN32)
  install(TARGETS ${_name} ${lib_type} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endmacro()
