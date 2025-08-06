cmake_minimum_required(VERSION 3.0.2)
project(simulator)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    rostime
    sensor_msgs
    message_filters
)

find_package(OpenCV REQUIRED)
find_package(OpenMP)
find_package(PCL REQUIRED)

catkin_package()

# ##########
# # Build ##
# ##########

include_directories(
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

add_executable(publish_pointcloud src/publish_pointcloud.cpp)

target_link_libraries(publish_pointcloud
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${PCL_LIBRARIES}
)

# ############
# # Install ##
# ############

catkin_install_python(PROGRAMS
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS publish_pointcloud
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)