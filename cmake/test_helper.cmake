
# Adds an MPI test to the test suite
#
# add_mpi_test(name executable [arguments])
#
# Args:
#   name: name of the test
#   executable: name of the mpi executable
#   aurguments (optional): optional arguments to the executable
function(add_mpi_test name executable)
    add_test(${name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${CMAKE_CURRENT_BINARY_DIR}/${executable} ${ARGN})
endfunction()

# Adds a serial test to the test suite
#
# add_serial_test(name executable [arguments])
#
# Args:
#   name: name of the test
#   executable: name of the executable
#   aurguments (optional): optional arguments to the executable
function(add_serial_test name executable)
    if (USE_MPI)
        add_test(${name} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${CMAKE_CURRENT_BINARY_DIR}/${executable} ${ARGN})
    else ()
        add_test(${name} ${CMAKE_CURRENT_BINARY_DIR}/${executable} ${ARGN})
    endif ()
endfunction()
