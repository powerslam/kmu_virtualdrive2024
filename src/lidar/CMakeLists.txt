cmake_minimum_required(VERSION 3.0.2)
project(lidar)

find_package(catkin REQUIRED COMPONENTS
  rospy
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES lidar
#  CATKIN_DEPENDS rospy
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
   src/obstacle.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )
