# set platform preprocessor macro
set(CPPPROJ_PLATFORM "__${CMAKE_SYSTEM_NAME}__")
if(APPLE)
    set(CPPPROJ_PLATFORM "__APPLE__")
elseif(WIN32)
    if(MSVC)
        set(CPPPROJ_PLATFORM "__WIN_MSVC__")
    elseif(CMAKE_COMPILER_IS_GNUC OR CMAKE_COMPILER_IS_GNUCXX)
        set(CPPPROJ_PLATFORM "__WIN_GNUC__")
    endif(MSVC)
else(APPLE)
    set(CPPPROJ_PLATFORM "__LINUX__")
endif(APPLE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D${CPPPROJ_PLATFORM}")
message("-- CPPPROJ platform defined as: ${CPPPROJ_PLATFORM}")

macro( add_lib_to_CPPPROJ _name _source )
  # add the library
  add_library(${_name}             ${_source})
  # add it to the list of CPPPROJ libraries
  set(CPPPROJ_LIBRARIES ${CPPPROJ_LIBRARIES} ${_name})
endmacro()

macro( install_lib _name )
  # install it
  set(lib_type LIBRARY)
  if(WIN32)
    set(lib_type RUNTIME)
  endif(WIN32)
  install(TARGETS ${_name} ${lib_type} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endmacro()
