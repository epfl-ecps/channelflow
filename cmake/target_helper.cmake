include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/compiler_helper.cmake)

# Defines a new target library ${name}_${type}, either shared or static.
#
# Installs it in ${CMAKE_INSTALL_PREFIX}/lib
#
# install_channelflow_library(name)
#
# Args:
#     name: name of the library to be added as a target. Expects a
#         variable named ${name}_SOURCES to be defined and to contain
#         the list of sources.
#
function(install_channelflow_library name)

    # Uncomment to debug calls to this function
    # message(STATUS "Building library: ${name}")

    # Create the target with the required type
    add_library(${name} ${${name}_SOURCES})

    target_include_directories(${name} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
        $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(${name} PUBLIC cfbasics)
    install(TARGETS ${name} DESTINATION lib)
    set_warning_flags(${name})

endfunction()

# Adds FFTW3 to the target link libraries and handles whether or not
# to activate MPI support.
#
# target_link_fftw(name, export_type)
#
# Args:
#    name: name of the target that needs to be linked with FFTW
#    export_type: either PUBLIC, PRIVATE or INTERFACE
#
function(target_link_fftw name export_type)

    # Uncomment to debug calls to this function
    # message(STATUS "Linking ${name} to FFTW")

    if (USE_MPI)
        target_link_libraries(${name} ${export_type} ${FFTW_MPI_LIBRARY})
    endif ()
    target_link_libraries(${name} ${export_type} ${FFTW_LIBRARY})

endfunction()

# Adds and install an application that depends on the channelflow library. Sets compiler
# warning levels.
#
# install_channelflow_application(name destination)
#
# Args:
#     name: name of the application to be installed
#     destination: where to install the application. If the destination is OFF
#         the application is not installed
function(install_channelflow_application name destination)

    set(app_name "${name}_app")

    # Uncomment to debug calls to this function
    # message(STATUS "Building application: ${name} (with target name "${app_name}")

    add_executable(${app_name} ${name}.cpp)
    target_link_libraries(${app_name} PUBLIC chflow)
    set_warning_flags(${app_name})
    set_target_properties(${app_name} PROPERTIES OUTPUT_NAME ${name})

    if (destination)
        install(TARGETS ${app_name} DESTINATION ${destination})
    endif ()

endfunction()
