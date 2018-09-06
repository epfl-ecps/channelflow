#  
# Include and configure the test code for the python wrapper
#
# This file is a part of channelflow version 2.0 https://channelflow.ch.
# License is GNU GPL version 2 or later: ./LICENCE
#

set(pw_TESTS pythonWrapperTest)

find_package(PythonInterp REQUIRED)
set(CFWRAPPER_PYTHON_PATH "${CMAKE_CURRENT_BINARY_DIR}/../")

foreach (program ${pw_TESTS})

    # Set path during copy
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/${program}.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/${program}.py
    )
    
    add_test(NAME ${program}
             COMMAND ${PYTHON_EXECUTABLE}
             ${CMAKE_CURRENT_BINARY_DIR}/${program}.py
             ${program})
endforeach (program)

