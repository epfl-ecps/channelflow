# Attempts to compile a test C++ file with the current build system configuration.
#
# CHECK_MPI_COMPILES(MPI_COMPILES)
#
# Args:
#     MPI_COMPILES: boolean return value
#

function(CHECK_MPI_COMPILES MPI_COMPILES)

    SET(TEMP_DIRECTORY ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/)
    FILE(WRITE ${TEMP_DIRECTORY}/mpi_test.cxx 
        "#include <mpi.h>\n"
        "int main(int argc, char** argv){MPI_Init(&argc,&argv); MPI_Finalize();return 0;}\n")
    TRY_COMPILE(MPI_COMPILES ${TEMP_DIRECTORY} ${TEMP_DIRECTORY}/mpi_test.cxx OUTPUT_VARIABLE OUTPUT)

endfunction()
