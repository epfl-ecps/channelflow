set(ilc_EXAMPLES benchmark_ilc)

foreach (program ${ilc_EXAMPLES})
    install_channelflow_application(${program} examples)
    target_link_libraries(${program}_app PUBLIC ilc)
endforeach (program)
