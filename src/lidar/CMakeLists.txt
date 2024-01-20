cmake_minimum_required(VERSION 3.0.2)
project(lidar)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  message_generation
)

add_message_files(
  FILES  
  Obstacle.msg
  ObstacleArray.msg
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
  src/obstacle.py
  src/index.py
  src/main.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )
