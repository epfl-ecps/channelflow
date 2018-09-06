#
# Define cfbasics as an INTERFACE (i.e. header only) library
#
# This permits to use it to set dependencies on targets, rather than
# setting global include directories
#
# This file is a part of channelflow version 2.0, https://channelflow.ch .
# License is GNU GPL version 2 or later: ./LICENSE
#

set(
    cfbasics_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/arglist.h
    ${CMAKE_CURRENT_SOURCE_DIR}/brent.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cfarray.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cfbasics.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cfvector.h
    ${CMAKE_CURRENT_SOURCE_DIR}/complexdefs.h
    ${CMAKE_CURRENT_SOURCE_DIR}/mathdefs.h
)

# Define the target with appropriate dependencies
add_library(cfbasics INTERFACE)
target_include_directories(cfbasics SYSTEM INTERFACE ${FFTW_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR})
target_link_fftw(cfbasics INTERFACE)

# Install header files
install(FILES ${cfbasics_HEADERS} DESTINATION include/cfbasics)
