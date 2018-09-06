#!/bin/bash -e
#
# Description:
#    This script is used to run unit and integration tests in Travis

# Print ccache version and re-initialize statistics
ccache -V
ccache -z

mkdir ${TRAVIS_BUILD_DIR}/cmake-build-release && cd ${TRAVIS_BUILD_DIR}/cmake-build-release
# Tests done for coverage are built in debug mode, to hit assertions.
# We use -Og not to slow down excessively the executables and tests
if [[ "${OMPI_CC}" == "gcc" ]]
then
    # Parallel build in Release mode
    CXXFLAGS="--coverage -ggdb3 -Og" CC="ccache mpicc" CXX="ccache mpicxx" cmake -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=ON -DCMAKE_BUILD_TYPE:STRING=Debug -DWITH_HDF5CXX:BOOL=ON -DWITH_PYTHON:BOOL=OFF -DBUILD_SHARED_LIBS:BOOL=ON -DWITH_NSOLVER:BOOL=ON -DWITH_NETCDF:STRING=Serial ${TRAVIS_BUILD_DIR}
else
    # Parallel build in Release mode
    CC="ccache mpicc" CXX="ccache mpicxx" cmake -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=ON -DCMAKE_BUILD_TYPE:STRING=Release -DWITH_HDF5CXX:BOOL=ON -DWITH_PYTHON:BOOL=OFF -DBUILD_SHARED_LIBS:BOOL=ON -DWITH_NSOLVER:BOOL=ON -DWITH_NETCDF:STRING=Serial ${TRAVIS_BUILD_DIR}
fi
make -j 4

# Print ccache statistics
ccache -s

# Run tests
make test


