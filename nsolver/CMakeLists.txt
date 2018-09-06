#
# Include and configure the NSolver part of channelflow.
#
# This file is a part of channelflow version 2.0 https://channelflow.ch.
# License is GNU GPL version 2 or later: ./LICENCE
#

set(
    nsolver_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/lanczos.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/arnoldi.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bicgstab.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/continuation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dsi.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gmres.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fgmres.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/newtonalgorithm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/newton.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/eigenvals.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/multiShootingDSI.cpp
)

set(
    nsolver_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/arnoldi.h
    ${CMAKE_CURRENT_SOURCE_DIR}/lanczos.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bicgstab.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bicgstabl.h
    ${CMAKE_CURRENT_SOURCE_DIR}/continuation.h
    ${CMAKE_CURRENT_SOURCE_DIR}/dsi.h
    ${CMAKE_CURRENT_SOURCE_DIR}/gmres.h
    ${CMAKE_CURRENT_SOURCE_DIR}/fgmres.h
    ${CMAKE_CURRENT_SOURCE_DIR}/newtonalgorithm.h
    ${CMAKE_CURRENT_SOURCE_DIR}/newton.h
    ${CMAKE_CURRENT_SOURCE_DIR}/nsolver.h
    ${CMAKE_CURRENT_SOURCE_DIR}/eigenvals.h
    ${CMAKE_CURRENT_SOURCE_DIR}/multiShootingDSI.h
)

# Define the target with appropriate dependencies
install_channelflow_library(nsolver)
# FIXME: These errors should be taken care of
if ( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND ${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER "6.0.0")
    target_compile_options(nsolver PUBLIC "-Wno-error=misleading-indentation")
endif()

# Install header files
install(FILES ${nsolver_HEADERS} DESTINATION include/nsolver)
