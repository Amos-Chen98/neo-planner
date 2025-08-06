cmake_minimum_required(VERSION 3.0.2)
project(planner)

find_package(
    catkin REQUIRED COMPONENTS
    roscpp
    rospy
    octomap_ros
    geometry_msgs
    actionlib
    actionlib_msgs
)

# Generate actions in the 'action' folder
add_action_files(
    FILES
    Plan.action
)

# Generate added messages and services with any dependencies listed here
generate_messages(
    DEPENDENCIES
    geometry_msgs
    actionlib_msgs
)

# ##########
# # Build ##
# ##########

include_directories(${catkin_INCLUDE_DIRS})

catkin_install_python(PROGRAMS
    scripts/ros_node/esdf_vis_node.py
    scripts/ros_node/manager_node.py
    scripts/ros_node/traj_planner_node.py
    scripts/ros_node/odom2base_tf_node.py
    scripts/ros_node/visualizer_node.py
    scripts/ros_node/tracker_manager_node.py
    scripts/ros_node/tracker_planner_node.py
    scripts/ros_node_test/manager_test_node.py
    scripts/ros_node_test/traj_planner_test_node.py
    scripts/ros_node_test/traj_planner_demo_node.py
    scripts/ros_node_test/traj_planner_all_demo_node.py
    scripts/ros_node_test/visualizer_demo_node.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
