set(
    ilc_APPS
    ilc_simulateflow
    ilc_findsoln
    ilc_continuesoln
    ilc_findeigenvals
    ilc_edgetracking
    ilc_addbaseflow
)

foreach (program ${ilc_APPS})
    install_channelflow_application(${program} bin)
    target_link_libraries(${program}_app PUBLIC ilc)
endforeach (program)

