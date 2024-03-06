#!/bin/bash
echo "=============================="

selected_planner=${1:-"enhanced"}  # 'basic', 'batch', 'record', 'nn', 'enhanced', or 'warmstart'
echo "The selected planner is $selected_planner"

gazebo_world=${2:-"poles"}  # Default to empty world if no argument is provided
echo "The gazebo world is $gazebo_world"

replan_mode=${3:-"periodic"}  # 'global', 'online', or 'periodic'
echo "The replan mode is $replan_mode"

is_save_rosbag=${4:-"false"}  # Default to true if no argument is provided
echo "The is_save_rosbag is $is_save_rosbag"

echo "=============================="

max_target_find_time=45 # Maximum simulation time in seconds

# Launch roslaunch in a new GNOME terminal tab and capture its PID
gnome-terminal --tab --title="bring up" --command="roslaunch planner bringup.launch headless:=true is_save_metric:=true max_target_find_time:=$max_target_find_time gazebo_world:=$gazebo_world selected_planner:=$selected_planner replan_mode:=$replan_mode" &
GNOME_PID=$!

# Sleep for 25 seconds to allow roslaunch to start and stabilize
sleep 25

# rosbag record in the background, assuming this is a once-off command and does not need to be explicitly killed
time_now=$(date +"%Y-%m-%d-%H-%M-%S")
if [ "$is_save_rosbag" = "true" ]; then
    rosbag record -a -x "/camera.*" -O /tmp/demo_"$time_now".bag &
fi

# Publish a goal, assuming this is a once-off command and does not need to be explicitly killed
rostopic pub -1 /move_base_simple/goal geometry_msgs/PoseStamped '{header: {stamp: now, frame_id: "map"}, pose: {position: {x: 30.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}' &

# Sleep for $max_target_find_time+5 to allow some processing after publishing the goal
sleep $((max_target_find_time+5))

# If there are any specific ROS nodes you want to ensure are stopped, you could also use rosnode kill here
# Example (not typically necessary with proper terminal shutdown):
rosnode kill -a

# Check if the GNOME terminal process exists before trying to kill it
if kill -0 $GNOME_PID 2>/dev/null; then
    kill $GNOME_PID
    wait $GNOME_PID  # Wait for the process to be terminated
else
    echo "Process $GNOME_PID not found."
fi
