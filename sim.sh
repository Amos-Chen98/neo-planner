gnome-terminal --tab --title="simulator" --command="bash -c 'roslaunch simulator sim_onboard.launch; exec bash'" &
sleep 3
gnome-terminal --tab --title="map_server" --command="bash -c 'roslaunch planner map_server_onboard.launch; exec bash'" &
gnome-terminal --tab --title="planner" --command="bash -c 'roslaunch planner planner.launch; exec bash'" &
gnome-terminal --tab --title="manager" --command="bash -c 'roslaunch planner manager.launch; exec bash'" &
sleep 17
rostopic pub /move_base_simple/goal geometry_msgs/PoseStamped '{header: {stamp: now, frame_id: "map"}, pose: {position: {x: 30.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}' &
wait
