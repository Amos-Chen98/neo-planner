#!/bin/bash
max_target_find_time=45 # Maximum simulation time in seconds

# Launch roslaunch in a new GNOME terminal tab and capture its PID
gnome-terminal --tab --title="bring up" --command="roslaunch planner bringup.launch headless:=true is_save_metric:=true max_target_find_time:=$max_target_find_time gazebo_world:=poles" &
GNOME_PID=$!

# Sleep for 25 seconds to allow roslaunch to start and stabilize
sleep 25

# Publish a goal, assuming this is a once-off command and does not need to be explicitly killed
rostopic pub -1 /move_base_simple/goal geometry_msgs/PoseStamped '{header: {stamp: now, frame_id: "map"}, pose: {position: {x: 30.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}' &

# Sleep for $max_target_find_time+5 to allow some processing after publishing the goal
sleep $(($max_target_find_time+5))

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
