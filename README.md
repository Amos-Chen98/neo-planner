# drone_ws
Motion planning simulator for drones based on ROS-Noetic and PX4

## Environments

When using this project, make sure the following dependencies have been successfully installed and configured.

* ROS1 with Gazebo
* PX4-Autopilot
* Mavros

Please also make sure that this ROS workspace has been sourced in `.bashrc`.

## Trajectory tracking demo

updated 01/17/2023.

Step1, launch QGC, and launch the simulator and Mavros using the following command.

```bash
roslaunch px4_controller run_simulator.launch 
```

Step2, use the `take off` command in QGC to make the drone take off.

Step3, start trajecory tracking.

```
rosrun px4_controller traj_tracking.py
```



