#!/bin/bash -e
#
# Description:
#    This script is used to build Channelflow in Travis under different configurations

# Print ccache version and re-initialize statistics
ccache -V
ccache -z

# Serial build
mkdir ${TRAVIS_BUILD_DIR}/cmake-build-debug-serial && cd ${TRAVIS_BUILD_DIR}/cmake-build-debug-serial
CC="ccache ${OMPI_CC}" CXX="ccache ${OMPI_CXX}" cmake -DUSE_MPI:BOOL=OFF -DCMAKE_BUILD_TYPE:STRING=Debug -DWITH_HDF5CXX:BOOL=ON -DWITH_PYTHON:BOOL=OFF -DBUILD_SHARED_LIBS:BOOL=ON -DWITH_NSOLVER:BOOL=ON -DWITH_NETCDF:STRING=Serial ${TRAVIS_BUILD_DIR}
make -j 4

# Parallel build
mkdir ${TRAVIS_BUILD_DIR}/cmake-build-debug && cd ${TRAVIS_BUILD_DIR}/cmake-build-debug
CC="ccache mpicc" CXX="ccache mpicxx" cmake -DCMAKE_BUILD_TYPE:STRING=Debug -DWITH_HDF5CXX:BOOL=ON -DWITH_PYTHON:BOOL=OFF -DBUILD_SHARED_LIBS:BOOL=ON -DWITH_NSOLVER:BOOL=ON -DWITH_NETCDF:STRING=Serial ${TRAVIS_BUILD_DIR}
make -j 4

# Print ccache statistics
ccache -s
