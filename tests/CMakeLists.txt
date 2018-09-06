#
# Include and configure integration tests for channelflow.
#
# This file is a part of channelflow version 2.0, https://channelflow.ch .
# License is GNU GPL version 2 or later: ./LICENSE
#  
# gtest unit test framework
if(WITH_GTEST)
    add_subdirectory(gtest)
endif()

# Integration tests:
set(
    test_APPS
    chebyTest
    dnsOrrsommTest
    dnsParabolaTest
    dnsSinusoidTest
    dnsZeroTest
    helmholtzTest
    ioTest
    poissonTest
    pressureTest
    realprofileTest
    tausolverTest
    timeIntegrationTest
    tridiagTest
    vector2fieldTest
)

if (WITH_NSOLVER)
   set(test_APPS ${test_APPS}
   nsolvertest
   #continuationTest
   #findsolnTest
   #lauretteTest
   )
endif()

# Copy data file needed for tests from source folder to binary folder
set(data_files eq.nc sigmabest.asc uinit.nc ufinal.nc)
foreach (filename ${data_files})
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data/${filename}
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/data)
endforeach ()

# Register the executables
foreach (program ${test_APPS})
    install_channelflow_application(${program} OFF)
endforeach (program)

# Run unit tests before integration tests
if(WITH_GTEST)
    add_serial_test(unitTests gtest/runUnitTests)
endif()

add_serial_test(tridiag tridiagTest)
add_serial_test(cheby chebyTest)
add_serial_test(helmholtz helmholtzTest)
add_serial_test(tausolver tausolverTest)
add_serial_test(poisson poissonTest)
add_serial_test(io ioTest)
add_serial_test(pressure pressureTest)
add_serial_test(vector2field vector2fieldTest)

add_serial_test(dnsZero01 dnsZeroTest --cnfe1 --bulk)
add_serial_test(dnsZero02 dnsZeroTest --cnfe1 --gradp)
add_serial_test(dnsZero03 dnsZeroTest --cnab2 --bulkv)
add_serial_test(dnsZero04 dnsZeroTest --cnab2 --gradp)
add_serial_test(dnsZero05 dnsZeroTest --cnrk2 --bulkv)
add_serial_test(dnsZero06 dnsZeroTest --cnrk2 --gradp)
add_serial_test(dnsZero07 dnsZeroTest --smrk2 --bulkv)
add_serial_test(dnsZero08 dnsZeroTest --smrk2 --gradp)
add_serial_test(dnsZero09 dnsZeroTest --sbdf2 --bulkv)
add_serial_test(dnsZero10 dnsZeroTest --sbdf2 --gradp)
add_serial_test(dnsZero11 dnsZeroTest --sbdf3 --bulkv)
add_serial_test(dnsZero12 dnsZeroTest --sbdf3 --gradp)
add_serial_test(dnsZero13 dnsZeroTest --sbdf4 --bulkv)
add_serial_test(dnsZero14 dnsZeroTest --sbdf4 --gradp)

add_serial_test(dnsParabola01 dnsParabolaTest --cnfe1 --fluc --bulk)
add_serial_test(dnsParabola02 dnsParabolaTest --cnfe1 --fluc --gradp)
add_serial_test(dnsParabola03 dnsParabolaTest --cnfe1 --base --bulkv)
add_serial_test(dnsParabola04 dnsParabolaTest --cnfe1 --base --gradp)
add_serial_test(dnsParabola05 dnsParabolaTest --cnab2 --fluc --bulkv)
add_serial_test(dnsParabola06 dnsParabolaTest --cnab2 --fluc --gradp)
add_serial_test(dnsParabola07 dnsParabolaTest --cnab2 --base --bulkv)
add_serial_test(dnsParabola08 dnsParabolaTest --cnab2 --base --gradp)
add_serial_test(dnsParabola09 dnsParabolaTest --cnrk2 --fluc --bulkv)
add_serial_test(dnsParabola10 dnsParabolaTest --cnrk2 --fluc --gradp)
add_serial_test(dnsParabola11 dnsParabolaTest --cnrk2 --base --bulkv)
add_serial_test(dnsParabola12 dnsParabolaTest --cnrk2 --base --gradp)
add_serial_test(dnsParabola13 dnsParabolaTest --smrk2 --fluc --bulkv)
add_serial_test(dnsParabola14 dnsParabolaTest --smrk2 --fluc --gradp)
add_serial_test(dnsParabola15 dnsParabolaTest --smrk2 --base --bulkv)
add_serial_test(dnsParabola16 dnsParabolaTest --smrk2 --base --gradp)
add_serial_test(dnsParabola17 dnsParabolaTest --sbdf2 --fluc --bulkv)
add_serial_test(dnsParabola18 dnsParabolaTest --sbdf2 --fluc --gradp)
add_serial_test(dnsParabola19 dnsParabolaTest --sbdf2 --base --bulkv)
add_serial_test(dnsParabola20 dnsParabolaTest --sbdf2 --base --gradp)
add_serial_test(dnsParabola21 dnsParabolaTest --sbdf3 --fluc --bulkv)
add_serial_test(dnsParabola22 dnsParabolaTest --sbdf3 --fluc --gradp)
add_serial_test(dnsParabola23 dnsParabolaTest --sbdf3 --base --bulkv)
add_serial_test(dnsParabola24 dnsParabolaTest --sbdf3 --base --gradp)
add_serial_test(dnsParabola25 dnsParabolaTest --sbdf4 --fluc --bulkv)
add_serial_test(dnsParabola26 dnsParabolaTest --sbdf4 --fluc --gradp)
add_serial_test(dnsParabola27 dnsParabolaTest --sbdf4 --base --bulkv)
add_serial_test(dnsParabola28 dnsParabolaTest --sbdf4 --base --gradp)

#if(WITH_NSOLVER)
#    add_serial_test(continuation01 continuationTest --SolverEigen)
#    add_serial_test(continuation02 continuationTest --SolverGMRES)
#    add_serial_test(continuation03 continuationTest --SolverGMRES --OptHookstep)
#endif()

# FIXME: these tests need NetCDF

add_serial_test(timeIntegration timeIntegrationTest)

# if (${WITH_NSOLVER})
#   add_test(continuation continuationTest)
#   add_test(laurette lauretteTest)
#   add_test(findsoln findsolnTest -eqb -xrel -T 10 -sigma data/sigmabest.asc -Nn 2 -es 5e-8)
# endif()

# add_test(dnsSinusoid01 dnsSinusoidTest --cnfe1 --zero  --bulkv)
# add_test(dnsSinusoid02 dnsSinusoidTest --cnfe1 --zero  --gradp)
# add_test(dnsSinusoid03 dnsSinusoidTest --cnfe1 --parab --bulkv)
# add_test(dnsSinusoid04 dnsSinusoidTest --cnfe1 --parab --gradp)
# add_test(dnsSinusoid05 dnsSinusoidTest --cnab2 --zero  --bulkv)
# add_test(dnsSinusoid06 dnsSinusoidTest --cnab2 --zero  --gradp)
# add_test(dnsSinusoid07 dnsSinusoidTest --cnab2 --parab --bulkv)
# add_test(dnsSinusoid08 dnsSinusoidTest --cnab2 --parab --gradp)
# add_test(dnsSinusoid09 dnsSinusoidTest --cnrk2 --zero  --bulkv)
# add_test(dnsSinusoid10 dnsSinusoidTest --cnrk2 --zero  --gradp)
# add_test(dnsSinusoid11 dnsSinusoidTest --cnrk2 --parab --bulkv)
# add_test(dnsSinusoid12 dnsSinusoidTest --cnrk2 --parab --gradp)
# add_test(dnsSinusoid13 dnsSinusoidTest --smrk2 --zero  --bulkv)
# add_test(dnsSinusoid14 dnsSinusoidTest --smrk2 --zero  --gradp)
# add_test(dnsSinusoid15 dnsSinusoidTest --smrk2 --parab --bulkv)
# add_test(dnsSinusoid16 dnsSinusoidTest --smrk2 --parab --gradp)
# add_test(dnsSinusoid17 dnsSinusoidTest --sbdf2 --zero  --bulkv)
# add_test(dnsSinusoid18 dnsSinusoidTest --sbdf2 --zero  --gradp)
# add_test(dnsSinusoid19 dnsSinusoidTest --sbdf2 --parab --bulkv)
# add_test(dnsSinusoid20 dnsSinusoidTest --sbdf2 --parab --gradp)
# add_test(dnsSinusoid21 dnsSinusoidTest --sbdf3 --zero  --bulkv)
# add_test(dnsSinusoid22 dnsSinusoidTest --sbdf3 --zero  --gradp)
# add_test(dnsSinusoid23 dnsSinusoidTest --sbdf3 --parab --bulkv)
# add_test(dnsSinusoid24 dnsSinusoidTest --sbdf3 --parab --gradp)
# add_test(dnsSinusoid25 dnsSinusoidTest --sbdf4 --zero  --bulkv)
# add_test(dnsSinusoid26 dnsSinusoidTest --sbdf4 --zero  --gradp)
# add_test(dnsSinusoid27 dnsSinusoidTest --sbdf4 --parab --bulkv)
# add_test(dnsSinusoid28 dnsSinusoidTest --sbdf4 --parab --gradp)

# add_test(dnsOrrsomm01 dnsOrrsommTest  --linU  --base  --bulkv)
# add_test(dnsOrrsomm02 dnsOrrsommTest  --linU  --base  --gradp)
# add_test(dnsOrrsomm03 dnsOrrsommTest  --linu  --base  --bulkv)
# add_test(dnsOrrsomm04 dnsOrrsommTest  --linu  --base  --gradp)
# add_test(dnsOrrsomm05 dnsOrrsommTest  --rot   --base  --bulkv)
# add_test(dnsOrrsomm06 dnsOrrsommTest  --rot   --base  --gradp)
# add_test(dnsOrrsomm07 dnsOrrsommTest  --rot   --fluc  --bulkv)
# add_test(dnsOrrsomm08 dnsOrrsommTest  --rot   --fluc  --gradp)
# add_test(dnsOrrsomm09 dnsOrrsommTest  --skw   --base  --bulkv)
# add_test(dnsOrrsomm10 dnsOrrsommTest  --skw   --base  --gradp)
# add_test(dnsOrrsomm11 dnsOrrsommTest  --skw   --fluc  --bulkv)
# add_test(dnsOrrsomm12 dnsOrrsommTest  --skw   --fluc  --gradp)
# add_test(dnsOrrsomm13 dnsOrrsommTest  --alt   --base  --bulkv)
# add_test(dnsOrrsomm14 dnsOrrsommTest  --alt   --base  --gradp)
# add_test(dnsOrrsomm15 dnsOrrsommTest  --alt   --fluc  --bulkv)
# add_test(dnsOrrsomm16 dnsOrrsommTest  --alt   --fluc  --gradp)
# add_test(dnsOrrsomm17 dnsOrrsommTest  --div   --base  --bulkv)
# add_test(dnsOrrsomm18 dnsOrrsommTest  --div   --base  --gradp)
# add_test(dnsOrrsomm19 dnsOrrsommTest  --div   --fluc  --bulkv)
# add_test(dnsOrrsomm20 dnsOrrsommTest  --div   --fluc  --gradp)
# add_test(dnsOrrsomm21 dnsOrrsommTest  --cnv   --base  --bulkv)
# add_test(dnsOrrsomm22 dnsOrrsommTest  --cnv   --base  --gradp)
# add_test(dnsOrrsomm23 dnsOrrsommTest  --cnv   --fluc  --bulkv)
# add_test(dnsOrrsomm24 dnsOrrsommTest  --cnv   --fluc  --gradp)

if (USE_MPI)
    add_mpi_test(mpi_tridiag tridiagTest)
    add_mpi_test(mpi_cheby chebyTest)
    add_mpi_test(mpi_helmholtz helmholtzTest)
    add_mpi_test(mpi_tausolver tausolverTest)
    add_mpi_test(mpi_poisson poissonTest)
    add_mpi_test(mpi_io ioTest)
    add_mpi_test(mpi_pressure pressureTest)
    add_mpi_test(mpi_vector2field vector2fieldTest)

    add_mpi_test(mpi_dnsZero01 dnsZeroTest --cnfe1 --bulk)
    add_mpi_test(mpi_dnsZero02 dnsZeroTest --cnfe1 --gradp)
    add_mpi_test(mpi_dnsZero03 dnsZeroTest --cnab2 --bulkv)
    add_mpi_test(mpi_dnsZero04 dnsZeroTest --cnab2 --gradp)
    add_mpi_test(mpi_dnsZero05 dnsZeroTest --cnrk2 --bulkv)
    add_mpi_test(mpi_dnsZero06 dnsZeroTest --cnrk2 --gradp)
    add_mpi_test(mpi_dnsZero07 dnsZeroTest --smrk2 --bulkv)
    add_mpi_test(mpi_dnsZero08 dnsZeroTest --smrk2 --gradp)
    add_mpi_test(mpi_dnsZero09 dnsZeroTest --sbdf2 --bulkv)
    add_mpi_test(mpi_dnsZero10 dnsZeroTest --sbdf2 --gradp)
    add_mpi_test(mpi_dnsZero11 dnsZeroTest --sbdf3 --bulkv)
    add_mpi_test(mpi_dnsZero12 dnsZeroTest --sbdf3 --gradp)
    add_mpi_test(mpi_dnsZero13 dnsZeroTest --sbdf4 --bulkv)
    add_mpi_test(mpi_dnsZero14 dnsZeroTest --sbdf4 --gradp)

    add_mpi_test(mpi_dnsParabola01 dnsParabolaTest --cnfe1 --fluc --bulk)
    add_mpi_test(mpi_dnsParabola02 dnsParabolaTest --cnfe1 --fluc --gradp)
    add_mpi_test(mpi_dnsParabola03 dnsParabolaTest --cnfe1 --base --bulkv)
    add_mpi_test(mpi_dnsParabola04 dnsParabolaTest --cnfe1 --base --gradp)
    add_mpi_test(mpi_dnsParabola05 dnsParabolaTest --cnab2 --fluc --bulkv)
    add_mpi_test(mpi_dnsParabola06 dnsParabolaTest --cnab2 --fluc --gradp)
    add_mpi_test(mpi_dnsParabola07 dnsParabolaTest --cnab2 --base --bulkv)
    add_mpi_test(mpi_dnsParabola08 dnsParabolaTest --cnab2 --base --gradp)
    add_mpi_test(mpi_dnsParabola09 dnsParabolaTest --cnrk2 --fluc --bulkv)
    add_mpi_test(mpi_dnsParabola10 dnsParabolaTest --cnrk2 --fluc --gradp)
    add_mpi_test(mpi_dnsParabola11 dnsParabolaTest --cnrk2 --base --bulkv)
    add_mpi_test(mpi_dnsParabola12 dnsParabolaTest --cnrk2 --base --gradp)
    add_mpi_test(mpi_dnsParabola13 dnsParabolaTest --smrk2 --fluc --bulkv)
    add_mpi_test(mpi_dnsParabola14 dnsParabolaTest --smrk2 --fluc --gradp)
    add_mpi_test(mpi_dnsParabola15 dnsParabolaTest --smrk2 --base --bulkv)
    add_mpi_test(mpi_dnsParabola16 dnsParabolaTest --smrk2 --base --gradp)
    add_mpi_test(mpi_dnsParabola17 dnsParabolaTest --sbdf2 --fluc --bulkv)
    add_mpi_test(mpi_dnsParabola18 dnsParabolaTest --sbdf2 --fluc --gradp)
    add_mpi_test(mpi_dnsParabola19 dnsParabolaTest --sbdf2 --base --bulkv)
    add_mpi_test(mpi_dnsParabola20 dnsParabolaTest --sbdf2 --base --gradp)
    add_mpi_test(mpi_dnsParabola21 dnsParabolaTest --sbdf3 --fluc --bulkv)
    add_mpi_test(mpi_dnsParabola22 dnsParabolaTest --sbdf3 --fluc --gradp)
    add_mpi_test(mpi_dnsParabola23 dnsParabolaTest --sbdf3 --base --bulkv)
    add_mpi_test(mpi_dnsParabola24 dnsParabolaTest --sbdf3 --base --gradp)
    add_mpi_test(mpi_dnsParabola25 dnsParabolaTest --sbdf4 --fluc --bulkv)
    add_mpi_test(mpi_dnsParabola26 dnsParabolaTest --sbdf4 --fluc --gradp)
    add_mpi_test(mpi_dnsParabola27 dnsParabolaTest --sbdf4 --base --bulkv)
    add_mpi_test(mpi_dnsParabola28 dnsParabolaTest --sbdf4 --base --gradp)

    add_mpi_test(mpi_timeIntegration timeIntegrationTest)
    
    # continuationTest does not use an MPI test, because the solver is not parallel
    #   if (${WITH_NSOLVER})
    #     add_test(mpi_findsoln mpirun -np 4 findsolnTest -eqb -xrel -T 10 -sigma data/sigmabest.asc -Nn 2 -es 5e-8)
    #   endif()

    # add_test(mpi_dnsSinusoid01 mpirun -np 4 dnsSinusoidTest --cnfe1 --zero  --bulkv)
    # add_test(mpi_dnsSinusoid02 mpirun -np 4 dnsSinusoidTest --cnfe1 --zero  --gradp)
    # add_test(mpi_dnsSinusoid03 mpirun -np 4 dnsSinusoidTest --cnfe1 --parab --bulkv)
    # add_test(mpi_dnsSinusoid04 mpirun -np 4 dnsSinusoidTest --cnfe1 --parab --gradp)
    # add_test(mpi_dnsSinusoid05 mpirun -np 4 dnsSinusoidTest --cnab2 --zero  --bulkv)
    # add_test(mpi_dnsSinusoid06 mpirun -np 4 dnsSinusoidTest --cnab2 --zero  --gradp)
    # add_test(mpi_dnsSinusoid07 mpirun -np 4 dnsSinusoidTest --cnab2 --parab --bulkv)
    # add_test(mpi_dnsSinusoid08 mpirun -np 4 dnsSinusoidTest --cnab2 --parab --gradp)
    # add_test(mpi_dnsSinusoid09 mpirun -np 4 dnsSinusoidTest --cnrk2 --zero  --bulkv)
    # add_test(mpi_dnsSinusoid10 mpirun -np 4 dnsSinusoidTest --cnrk2 --zero  --gradp)
    # add_test(mpi_dnsSinusoid11 mpirun -np 4 dnsSinusoidTest --cnrk2 --parab --bulkv)
    # add_test(mpi_dnsSinusoid12 mpirun -np 4 dnsSinusoidTest --cnrk2 --parab --gradp)
    # add_test(mpi_dnsSinusoid13 mpirun -np 4 dnsSinusoidTest --smrk2 --zero  --bulkv)
    # add_test(mpi_dnsSinusoid14 mpirun -np 4 dnsSinusoidTest --smrk2 --zero  --gradp)
    # add_test(mpi_dnsSinusoid15 mpirun -np 4 dnsSinusoidTest --smrk2 --parab --bulkv)
    # add_test(mpi_dnsSinusoid16 mpirun -np 4 dnsSinusoidTest --smrk2 --parab --gradp)
    # add_test(mpi_dnsSinusoid17 mpirun -np 4 dnsSinusoidTest --sbdf2 --zero  --bulkv)
    # add_test(mpi_dnsSinusoid18 mpirun -np 4 dnsSinusoidTest --sbdf2 --zero  --gradp)
    # add_test(mpi_dnsSinusoid19 mpirun -np 4 dnsSinusoidTest --sbdf2 --parab --bulkv)
    # add_test(mpi_dnsSinusoid20 mpirun -np 4 dnsSinusoidTest --sbdf2 --parab --gradp)
    # add_test(mpi_dnsSinusoid21 mpirun -np 4 dnsSinusoidTest --sbdf3 --zero  --bulkv)
    # add_test(mpi_dnsSinusoid22 mpirun -np 4 dnsSinusoidTest --sbdf3 --zero  --gradp)
    # add_test(mpi_dnsSinusoid23 mpirun -np 4 dnsSinusoidTest --sbdf3 --parab --bulkv)
    # add_test(mpi_dnsSinusoid24 mpirun -np 4 dnsSinusoidTest --sbdf3 --parab --gradp)
    # add_test(mpi_dnsSinusoid25 mpirun -np 4 dnsSinusoidTest --sbdf4 --zero  --bulkv)
    # add_test(mpi_dnsSinusoid26 mpirun -np 4 dnsSinusoidTest --sbdf4 --zero  --gradp)
    # add_test(mpi_dnsSinusoid27 mpirun -np 4 dnsSinusoidTest --sbdf4 --parab --bulkv)
    # add_test(mpi_dnsSinusoid28 mpirun -np 4 dnsSinusoidTest --sbdf4 --parab --gradp)
endif ()
