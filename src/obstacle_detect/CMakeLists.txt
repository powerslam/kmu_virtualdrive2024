cmake_minimum_required(VERSION 3.0.2)
project(obstacle_detect)

find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  std_msgs
  message_generation
)

add_message_files(
  FILES

  Rotary.msg
  LidarObstacleInfo.msg
  LidarObstacleInfoArray.msg

  ObstacleInfo.msg
  ObstacleInfoArray.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
#  INCLUDE_DIRS include
  LIBRARIES lane_detection
  CATKIN_DEPENDS rospy std_msgs message_runtime
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(
  PROGRAMS
  src/lidar_obstacle_detect.py
  src/camera_obstacle_detect.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )
