#  
# Include and configure the channelflow tools
#
# This file is a part of channelflow version 2.0, https://channelflow.ch .
# License is GNU GPL version 2 or later: ./LICENSE
#
set(tool_APPS
  addfields
  benchmark
  changegrid
  diffop
  extrapolatefields
  fieldconvert
  fieldprops
  findsymmetries
  L2op
  optphaseshift
  perturbfield
  pressure
  randomfield
  symmetrize
  symmetryop
  )

foreach (program ${tool_APPS})
    install_channelflow_application(${program} bin)
endforeach ()
