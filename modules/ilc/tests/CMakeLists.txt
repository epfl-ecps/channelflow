set(ilc_TESTS ilc_timeIntegrationTest)

file(
    COPY
    ${CMAKE_CURRENT_SOURCE_DIR}/data/uinit.nc ${CMAKE_CURRENT_SOURCE_DIR}/data/ufinal.nc
    DESTINATION
    ${CMAKE_CURRENT_BINARY_DIR}/data
)

file(
    COPY
    ${CMAKE_CURRENT_SOURCE_DIR}/data/tinit.nc ${CMAKE_CURRENT_SOURCE_DIR}/data/tfinal.nc
    DESTINATION
    ${CMAKE_CURRENT_BINARY_DIR}/data
)

foreach (program ${ilc_TESTS})
    install_channelflow_application(${program} OFF)
    target_link_libraries(${program}_app PUBLIC ilc)
endforeach (program)

add_serial_test(ilc_timeIntegration ilc_timeIntegrationTest)

if (USE_MPI)
    add_mpi_test(mpi_ilc_timeIntegration ilc_timeIntegrationTest)
endif ()
