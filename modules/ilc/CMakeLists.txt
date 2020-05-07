#
# This file is a part of channelflow version 2.0 https://channelflow.ch.
# License is GNU GPL version 2 or later: ./LICENCE
#  
# Include and configure the Inclined Layer Convection module of channelflow.
#
set(
    ilc_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/ilcflags.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/obe.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ilc.cpp
)

set(
    ilc_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/ilcflags.h
    ${CMAKE_CURRENT_SOURCE_DIR}/obe.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ilc.h
)

# Define the target with appropriate dependencies
install_channelflow_library(ilc)
target_link_fftw(ilc PRIVATE)
target_link_libraries(ilc PUBLIC chflow)
if (WITH_NSOLVER)
    target_link_libraries(ilc PUBLIC nsolver)
endif ()

# Install header files
install(FILES ${ilc_HEADERS} DESTINATION include/modules/ilc)
