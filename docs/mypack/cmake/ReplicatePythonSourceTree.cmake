# Note: when executed in the build dir, then CMAKE_CURRENT_SOURCE_DIR is the
# build dir.
file( COPY xdressrc.py src mypack DESTINATION "${CMAKE_ARGV3}"
  FILES_MATCHING PATTERN "*.py" )
