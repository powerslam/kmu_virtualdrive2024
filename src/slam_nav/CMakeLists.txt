cmake_minimum_required(VERSION 3.0.2)
project(slam_nav)

set(CMAKE_CXX_STANDARD 14)
find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  sensor_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES slam_nav
#  CATKIN_DEPENDS roscpp rospy
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
add_executable(lidar_convert src/convert_lidar.cpp)
target_link_libraries(lidar_convert ${catkin_LIBRARIES})

catkin_install_python(PROGRAMS
   scripts/total.py
   scripts/pub_odom.py
   scripts/save_waypoint.py
   scripts/navigation_client.py
   scripts/lane_drive.py
   scripts/pure_pursuit_test.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)