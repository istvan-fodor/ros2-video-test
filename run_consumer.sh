#!/bin/bash

export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_ROCM_ARCH=gfx1031

. ./install/setup.sh
ros2 run fusiongrid_robot camera_consumer