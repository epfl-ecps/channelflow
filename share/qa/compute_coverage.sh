#!/bin/bash -e

if [[ "${OMPI_CC}" == "gcc" ]]
then
    # Go to the build directory and compute coverage
    cd ${TRAVIS_BUILD_DIR}/cmake-build-release
    lcov --version
    echo $PWD
    lcov --rc lcov_branch_coverage=1 --directory . --capture --output-file coverage.info
    lcov --rc lcov_branch_coverage=1 --remove coverage.info '/*tests/*' --output-file coverage.info
    lcov --rc lcov_branch_coverage=1 --remove coverage.info '/*/bundled/*' --output-file coverage.info
    lcov --rc lcov_branch_coverage=1 --remove coverage.info '/usr/*' --output-file coverage.info
    lcov --rc lcov_branch_coverage=1 --list coverage.info

    # Upload to codecov website
    bash <(curl -s https://codecov.io/bash) || echo "Codecov did not collect coverage reports"
fi
