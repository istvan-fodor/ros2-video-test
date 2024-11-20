#!/bin/bash

colcon build --symlink-install
source ./install/setup.bash
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp  ros2 launch analog_gauge_reader analog_gauge_reader.launch.py
