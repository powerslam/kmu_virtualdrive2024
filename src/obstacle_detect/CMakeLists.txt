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
  Obstacle.msg
  ObstacleArray.msg

  DynamicObstacle.msg
  DynamicObstacleArray.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
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
