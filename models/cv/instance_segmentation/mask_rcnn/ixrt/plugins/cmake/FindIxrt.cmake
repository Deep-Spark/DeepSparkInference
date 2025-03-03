# This cmake file decides how to build with IxRT
# Custom IxRT Path
if(NOT "${IXRT_HOME}" STREQUAL "")
    set(IXRT_INCLUDE_DIR ${IXRT_HOME}/include)
    set(IXRT_LIB_DIR ${IXRT_HOME}/lib)
# From default paths
else()
  set(IXRT_INCLUDE_DIR /usr/local/corex/include)
  set(IXRT_LIB_DIR /usr/local/corex/lib)
endif()

message(STATUS "IXRT_INCLUDE_DIR:   ${IXRT_INCLUDE_DIR}")
message(STATUS "IXRT_LIB_DIR:   ${IXRT_LIB_DIR}")

if(EXISTS ${IXRT_INCLUDE_DIR} AND EXISTS ${IXRT_LIB_DIR})
    include_directories(${IXRT_INCLUDE_DIR})
else()
    message( FATAL_ERROR "IxRT library doesn't exist!")
endif()
