cmake_minimum_required(VERSION 3.0.2)
project(kitting_robot)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/perception_node.py
  scripts/notify_node.py
  scripts/dobot_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
