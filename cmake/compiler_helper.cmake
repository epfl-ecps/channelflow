# Inspects the copmiler type and version, then adds compiler warning flags to a given target.
#
# set_warning_flags(target)
#
# Args:
#     target: target name
#
function(set_warning_flags target)
    # Sets the warning flags for a given target


    if( CMAKE_CXX_COMPILER_ID STREQUAL "GNU")

        target_compile_options(
            ${target}
            PRIVATE
            "-Wall;-Werror;-pedantic"
        )

    elseif( CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" )

        target_compile_options(
            ${target}
            PRIVATE
            "-Wall;-Werror;-pedantic;-Wdocumentation"
        )

    elseif( CMAKE_CXX_COMPILER_ID STREQUAL "Intel" )

        target_compile_options(
            ${target}
            PRIVATE
            "-Wall;-Werror"
        )

    elseif( CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" )

        target_compile_options(
            ${target}
            PRIVATE
            "/W4"
        )

    else()

        message(WARNING "No warning flags for this compiler.")

    endif()

endfunction()
