#!/bin/bash

source /opt/ros/jazzy/setup.bash
colcon build --cmake-args -DPython3_EXECUTABLE="$VIRTUAL_ENV/bin/python" -DPython3_FIND_VIRTUALENV=ONLY

