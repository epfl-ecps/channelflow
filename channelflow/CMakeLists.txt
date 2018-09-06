#
# Compile the Channelflow source files
# 
# This file is a part of channelflow version 2.0, https://channelflow.ch .
# License is GNU GPL version 2 or later: ./LICENSE
#
set(
    chflow_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/bandedtridiag.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/basisfunc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/chebyshev.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cfmpi.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/diffops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dnsflags.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dnsalgo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nse.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dns.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/flowfield.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/helmholtz.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/periodicfunc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/poissonsolver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/realprofile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/realprofileng.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/symmetry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tausolver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/turbstats.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utilfuncs.cpp
    ${CMAKE_BINARY_DIR}/GitSHA1.cpp
)

set(
    chflow_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/bandedtridiag.h
    ${CMAKE_CURRENT_SOURCE_DIR}/basisfunc.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cfmpi.h
    ${CMAKE_CURRENT_SOURCE_DIR}/chebyshev.h
    ${CMAKE_CURRENT_SOURCE_DIR}/diffops.h
    ${CMAKE_CURRENT_SOURCE_DIR}/dnsflags.h
    ${CMAKE_CURRENT_SOURCE_DIR}/dnsalgo.h
    ${CMAKE_CURRENT_SOURCE_DIR}/nse.h
    ${CMAKE_CURRENT_SOURCE_DIR}/dns.h
    ${CMAKE_CURRENT_SOURCE_DIR}/flowfield.h
    ${CMAKE_CURRENT_SOURCE_DIR}/helmholtz.h
    ${CMAKE_CURRENT_SOURCE_DIR}/periodicfunc.h
    ${CMAKE_CURRENT_SOURCE_DIR}/poissonsolver.h
    ${CMAKE_CURRENT_SOURCE_DIR}/realprofile.h
    ${CMAKE_CURRENT_SOURCE_DIR}/realprofileng.h
    ${CMAKE_CURRENT_SOURCE_DIR}/symmetry.h
    ${CMAKE_CURRENT_SOURCE_DIR}/tausolver.h
    ${CMAKE_CURRENT_SOURCE_DIR}/turbstats.h
    ${CMAKE_CURRENT_SOURCE_DIR}/utilfuncs.h
)

if (WITH_NSOLVER)
    set(
        chflow_SOURCES ${chflow_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/cfdsi.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/laurettedsi.cpp
    )
    set(
        chflow_HEADERS ${chflow_HEADERS}
        ${CMAKE_CURRENT_SOURCE_DIR}/cfdsi.h
        ${CMAKE_CURRENT_SOURCE_DIR}/laurettedsi.h
    )
endif ()

# Define the target with appropriate dependencies
install_channelflow_library(chflow)
target_link_fftw(chflow PUBLIC)

if (NETCDF_FOUND)
    target_include_directories(chflow SYSTEM PRIVATE ${NETCDF_INCLUDE_DIRS})
    target_link_libraries(chflow PRIVATE ${NETCDF_LIBRARIES})
endif ()

if (WITH_HDF5CXX)
    target_include_directories(chflow SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(chflow PRIVATE ${HDF5_LIBRARIES})
endif ()

if (WITH_NSOLVER)
    target_link_libraries(chflow PUBLIC nsolver)
endif ()

# TODO: fix the following error in source files
# target_compile_options(chflow PRIVATE "-Wno-error=misleading-indentation;-Wno-error=vla")

# Install header files
install(FILES ${chflow_HEADERS} DESTINATION include/channelflow)
install(FILES ${PROJECT_BINARY_DIR}/channelflow/config.h DESTINATION include/channelflow)
