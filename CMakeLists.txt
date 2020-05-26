#
# This file is a part of channelflow version 2.0, https://channelflow.ch .
# License is GNU GPL version 2 or later: ./LICENSE
#  
#  Include and configure this version of channelflow.
#
cmake_minimum_required(VERSION 3.1)
cmake_policy(SET CMP0057 NEW)
project(chflow)

######## Channelflow version ########
set(CF_VERSION_MAJOR "2")
set(CF_VERSION_MINOR "0")
set(CF_VERSION_PATCH "2")
set(CHANNELFLOW_VERSION "${CF_VERSION_MAJOR}.${CF_VERSION_MINOR}.${CF_VERSION_PATCH}")

message(STATUS "")
message(STATUS "configuring channelflow version " ${CF_VERSION_MAJOR} . ${CF_VERSION_MINOR} . ${CF_VERSION_PATCH})

######## Command line options ########
OPTION(USE_MPI "use mpi, if found" ON) # use -DUSE_MPI=off to force disable use of MPI
OPTION(BUILD_SHARED_LIBS "Builds shared libraries if ON, static libraries if OFF" ON)
OPTION(SET_RPATH "set the rpath of the executable to the found libraries" ON)
OPTION(WITH_NSOLVER "compile support for the nsolver module" ON)
OPTION(WITH_ILC "compile programs for inclined layer convection" OFF)
OPTION(WITH_PYTHON "build a python wrapper, using boost-python" OFF)
OPTION(WITH_HDF5CXX "enable legacy file format using hdf5 cxx" OFF)
OPTION(WITH_GTEST "enable gtest unit testing" ON)

# Set compiler flags that are build-type related
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

# Netcdf support
if (NOT WITH_NETCDF)
    set(WITH_NETCDF "Serial" CACHE STRING "Either OFF, Serial or Parallel.")
endif ()

if (WITH_NETCDF AND
    NOT (WITH_NETCDF STREQUAL "Serial" OR WITH_NETCDF STREQUAL "Parallel"))
    message(FATAL_ERROR "Allowed values for WITH_NETCDF are 'OFF', 'Serial' or 'Parallel [current value is '${WITH_NETCDF}']")
endif()



######## Basic settings ########

# Ask for C++ 11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
    message(STATUS "Using GNU compiler options")
elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL Intel)
    message(STATUS "Using Intel compiler options")
endif ()
set(COMPILER_VERSION "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

# rpath settings
# From: http://www.itk.org/Wiki/CMake_RPATH_handling
if (SET_RPATH)
    set(CMAKE_SKIP_BUILD_RPATH FALSE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
    if ("${isSystemDir}" STREQUAL "-1")
        set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
    endif ("${isSystemDir}" STREQUAL "-1")
else ()
    message(STATUS "not setting any rpath")
    set(CMAKE_SKIP_RPATH TRUE)
endif ()

######## Library finding and system introspection ########

message(STATUS "")
message(STATUS "Starting system introspection")
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules" ${CMAKE_MODULE_PATH})

if (BUILD_SHARED_LIBS)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES} .so)
else ()
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES} .a)
endif ()

# Basic system checks
include(CheckLibraryExists)
include(CheckFunctionExists)
include(CheckIncludeFiles)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/compiler_helper.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/target_helper.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_helper.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_mpi_compiles.cmake)

CHECK_FUNCTION_EXISTS(drand48 HAVE_DRAND48)
CHECK_INCLUDE_FILES(wordexp.h HAVE_WORDEXP_H)

# Find all the dependencies
find_package(Eigen3 REQUIRED)
find_package(FFTW REQUIRED)
set(CMAKE_REQUIRED_INCLUDES ${FFTW_INCLUDE_DIR})
CHECK_INCLUDE_FILES(fftw3.h HAVE_FFTW_H)

if (USE_MPI)
    # In principle channelflow only needs MPI_C libraries and not MPI_CXX, but
    # the legacy file format with HDF5 C++ requires to link against MPI_CXX
    find_package(MPI COMPONENTS C CXX REQUIRED)
    set(HAVE_MPI TRUE)
    message(STATUS "MPI libs: ${MPI_CXX_LIBRARIES}")
    message(STATUS "FFTW_MPI library: ${FFTW_MPI_LIBRARY}")

    # Require MPI-capable compiler
    CHECK_MPI_COMPILES(MPI_COMPILES)
    if (NOT MPI_COMPILES)
        if (MPI_CXX_COMPILER AND (NOT CMAKE_CXX_COMPILER STREQUAL MPI_CXX_COMPILER))
            SET(COMPILER_SUGGESTION " (such as ${MPI_CXX_COMPILER})")
        endif()
        message(FATAL_ERROR "Option USE_MPI is set and MPI is found, but the compiler (${CMAKE_CXX_COMPILER}) failed to compile MPI code. To set another compiler${COMPILER_SUGGESTION}, use cmake option -DCMAKE_CXX_COMPILER=<compiler> or set environment variable CXX. If you do not want to use MPI, clean up and rerun cmake with -DUSE_MPI=off.")
    endif ()

    # Require FFTW MPI library
    if (NOT FFTW_MPI_LIBRARY)
        message(FATAL_ERROR "Option USE_MPI is set and MPI is found, but cmake couldn't find libfftw_mpi, which is required. If you do not want to use MPI, clean up and rerun cmake with -DUSE_MPI=off.")
    endif ()
else ()
    message(STATUS "Not using MPI")
endif ()

if (WITH_NETCDF)
    find_package(NetCDF REQUIRED)
    message(STATUS "NetCDF include dir: ${NETCDF_INCLUDE_DIRS}")
    set(CMAKE_REQUIRED_INCLUDES ${NETCDF_INCLUDE_DIRS})
    CHECK_INCLUDE_FILES(netcdf.h HAVE_NETCDF_H)
    if (NOT HAVE_NETCDF_H)
        message(FATAL ERROR "could not include netcdf.h [Required by WITH_NETCDF]")
    endif ()
    # Check for the parallel header of NetCDF only if we are building with MPI
    if (WITH_NETCDF STREQUAL "Parallel")
        CHECK_INCLUDE_FILES(netcdf_par.h HAVE_NETCDF_PAR_H)
        if (NOT HAVE_NETCDF_PAR_H)
            message(FATAL ERROR "could not include netcdf_par.h [Required by WITH_NETCDF=Parallel]")
        endif ()
    endif ()
endif ()

if (WITH_HDF5CXX)
    find_package(HDF5 COMPONENTS CXX REQUIRED)
    get_filename_component(HDF5_LIBRARY_DIR "${HDF5_CXX_LIBRARY_hdf5}" DIRECTORY)
    message(STATUS "Checking if hdf5 library functions in ${HDF5_LIBRARY_DIR} are working correctly.")
    set(CMAKE_REQUIRED_INCLUDES ${HDF5_INCLUDE_DIRS})
    CHECK_INCLUDE_FILES(hdf5.h HAVE_HDF5_H)
    CHECK_LIBRARY_EXISTS(hdf5 H5D_init "${HDF5_LIBRARY_DIR}" HAVE_LIBHDF5)
    set(CMAKE_REQUIRED_LIBRARIES hdf5) # need hdf5 to run hdf5_hl and hdf5_cpp
    CHECK_LIBRARY_EXISTS(hdf5_hl H5Dopen2 "${HDF5_LIBRARY_DIR}" HAVE_LIBHDF5_HL)
    CHECK_LIBRARY_EXISTS(hdf5_cpp H5Eprint2 "${HDF5_LIBRARY_DIR}" HAVE_LIBHDF5_CPP)
endif ()

######## Create config.h from config.h.in ########
configure_file("${PROJECT_SOURCE_DIR}/config.h.in" "${PROJECT_BINARY_DIR}/channelflow/config.h")
if (WITH_NSOLVER)
    configure_file("${PROJECT_SOURCE_DIR}/config.h.in" "${PROJECT_BINARY_DIR}/nsolver/config.h")
endif ()

######## Write git revision into a generated cpp file ########
# We could also directly write the git revision into the config.h,
# but this would cause everything to rebuild everytime the git
# revision changes
include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_SHA1)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/GitSHA1.cpp.in" "${CMAKE_CURRENT_BINARY_DIR}/GitSHA1.cpp" @ONLY)
list(APPEND SOURCES "${CMAKE_CURRENT_BINARY_DIR}/GitSHA1.cpp" GitSHA1.h)
get_directory_property(OUT_VAR INCLUDE_DIRECTORIES)

message(STATUS "")
message(STATUS "Include directories: ${OUT_VAR}")
message(STATUS "Git revision ${GIT_SHA1}")

# Set up bundled projects
add_subdirectory(bundled)

# Compile core libraries and executables
add_subdirectory(cfbasics)
if (WITH_NSOLVER)
    add_subdirectory(nsolver)
endif ()
add_subdirectory(channelflow)
add_subdirectory(programs)
add_subdirectory(tools)
add_subdirectory(examples)

if (WITH_ILC)
    if (NOT WITH_NSOLVER)
        message(FATAL_ERROR "Compiling the ILC module requires -DWITH_NSOLVER=ON")
    endif()
    add_subdirectory(modules/ilc)
    add_subdirectory(modules/ilc/programs)
    add_subdirectory(modules/ilc/examples)
endif ()

# Check if we want to build the python wrapper and have boost-python
# If Python and boost.python are there, build python interface
if (WITH_PYTHON)
    if (USE_MPI)
            message(FATAL_ERROR "Compiling the python wrapper requires -DUSE_MPI=OFF")
    endif()
    find_package(Boost COMPONENTS python REQUIRED)
    find_package(PythonLibs 2.7 REQUIRED)
    message(STATUS "Building with Python interface.")
    add_subdirectory(python-wrapper)
endif ()

enable_testing()
add_subdirectory(tests)
if (WITH_ILC)
    add_subdirectory(modules/ilc/tests)
endif ()
if (WITH_PYTHON)
    add_subdirectory(python-wrapper/tests)
endif()

################################################################
# Doxygen
################################################################

find_package(Doxygen)
if (DOXYGEN_FOUND)
    set(DOXYGEN_GENERATE_LATEX NO)
    set(DOXYGEN_RECURSIVE YES)
    set(DOXYGGEN_EXTRACT_ALL YES)
    set(DOXYGEN_WARNINGS YES)
    set(DOXYGEN_WARN_NO_PARAMDOC YES)
    set(DOXYGEN_WARN_IF_DOC_ERROR YES)
    set(DOXYGEN_WARN_IF_UNDOCUMENTED YES)
    set(DOXYGEN_GENERATE_TREEVIEW YES)
    set(DOXYGEN_GENERATE_HTML YES)
    set(DOXYGEN_GENERATE_MAN NO)
    set(DOXYGEN_HAVE_DOT YES)
    set(DOXYGEN_CALL_GRAPH YES)
    set(DOXYGEN_CALLER_GRAPH YES)
    set(DOXYGEN_FULL_PATH_NAMES YES)
    set(DOXYGEN_EXCLUDE_PATTERNS "${CMAKE_SOURCE_DIR}/bundled/*" "${CMAKE_SOURCE_DIR}/cmake/*" "${CMAKE_SOURCE_DIR}/share/*")
    
    # to set other options, read: https://cmake.org/cmake/help/v3.9/module/FindDoxygen.html
    doxygen_add_docs(
        doxygen
        ${PROJECT_SOURCE_DIR}
        COMMENT "Generate html pages"
    )
    add_custom_target(doc DEPENDS doxygen)
endif(DOXYGEN_FOUND)


######## Summary ########
message("\n")
message("   ###############################################")
message("   ###########  Configuration summary  ###########")
message("   ###############################################")
message("   Compiler:                       ${COMPILER_VERSION}")
message("   Build type:                     ${CMAKE_BUILD_TYPE}")
message("   Install prefix:                 ${CMAKE_INSTALL_PREFIX}")

if (BUILD_SHARED_LIBS)
    message("   Building shared library:        yes")
    message("   Linking programs to:            shared library")
else ()
    message("   Building shared library:        no")
    message("   Linking programs to:            static library")
endif ()

message("\n   Libraries")

if (MPI_FOUND)
    message("   MPI:                            enabled")
else ()
    message("   MPI:                            disabled")
endif ()

if (HAVE_NETCDF_H)
    message("   netcdf (default file format):   enabled")
else ()
    message("   netcdf (default file format):   disabled")
endif ()

if (HAVE_NETCDF_PAR_H)
    message("   Parallel netcdf:                enabled")
else ()
    message("   Parallel netcdf:                disabled")
endif ()


if (WITH_HDF5CXX)
    message("   HDF5 C++ (legacy file format):  enabled")
else ()
    message("   HDF5 C++ (legacy file format):  disabled")
endif ()

message("\n   Channelflow components")

if (WITH_NSOLVER)
    message("   nsolver:                        enabled")
else ()
    message("   nsolver:                        disabled")
endif ()

if (WITH_ILC)
    message("   Inclined layer convection:      enabled")
else ()
    message("   Inclined layer convection:      disabled")
endif ()

if (WITH_PYTHON)
    message("   Python wrapper:                 enabled")
else ()
    message("   Python wrapper:                 disabled")
endif ()

if (WITH_GTEST)
    message("   GTest unit testing:             enabled")
else ()
    message("   GTest unit testing:             disabled")
endif ()

message("\n")
