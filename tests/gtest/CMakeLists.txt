#
# Include and configure unit tests for this version of channelflow.
#
# This file is a part of channelflow version 2.0, https://channelflow.ch .
# License is GNU GPL version 2 or later: ./LICENSE
#


include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
set(
    test_sources
    periodicfunctest.cpp
    timesteptest.cpp
    cfbasicstest.cpp
)

if (WITH_NSOLVER)
   set(
       test_sources
       ${test_sources}
       msDSItest.cpp
       arnolditest.cpp
       lanczostest.cpp
   )
endif()

add_executable( runUnitTests ${test_sources})

# Link test executable against gtest & gtest_main
target_link_libraries(runUnitTests chflow gtest gtest_main)
add_dependencies(runUnitTests chflow)

if (WITH_NSOLVER)
   target_link_libraries(runUnitTests chflow nsolver gtest gtest_main)
   add_dependencies(runUnitTests chflow nsolver)
endif()

# Run specific test with `test/gtest/runUnitTests --gtest_filter=Periodic*`
add_custom_target(unitTests
    COMMAND runUnitTests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running unit tests"
    VERBATIM)
