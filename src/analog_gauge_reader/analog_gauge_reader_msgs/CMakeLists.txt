cmake_minimum_required(VERSION 3.8)
project(analog_gauge_reader_msgs)

if(POLICY CMP0148)
  cmake_policy(SET CMP0148 OLD)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/GaugeReading.msg"
  "msg/GaugeReadings.msg"
  "srv/GaugeReader.srv"
  DEPENDENCIES std_msgs
)

ament_package()
