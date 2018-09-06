#
# Configure the channelflow python wrapper pycflib
#
# This file is a part of channelflow version 2.0 https://channelflow.ch.
# License is GNU GPL version 2 or later: ./LICENCE

set(
    pycf_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/cfwrapper.cpp
)

# Define the target with appropriate dependencies
install_channelflow_library(pycf)
target_include_directories(pycf SYSTEM PUBLIC ${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
target_link_libraries(pycf PUBLIC ${Boost_LIBRARIES} nsolver chflow)
