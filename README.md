# drone_ws
## Introduction

This project (as a ROS workspace) provides a motion planning framework for drones based on ROS and PX4-SITL.

## Installation

### Prerequisites

1. Install required framework and softwares.

   Before using this project, make sure the following dependencies have been successfully installed and configured.

* ROS1 with Gazebo: https://wiki.ros.org/noetic/Installation/Ubuntu
* PX4-Autopilot: https://docs.px4.io/main/en/dev_setup/dev_env_linux_ubuntu.html

```bash
git clone https://github.com/PX4/PX4-Autopilot.git
cd PX4-Autopilot
git checkout v1.13.2
bash ./PX4-Autopilot/Tools/setup/ubuntu.sh
make px4_sitl gazebo
```

* Mavros: https://docs.px4.io/main/en/ros/mavros_installation.html
* QGC: https://docs.qgroundcontrol.com/master/en/getting_started/download_and_install.html

2. Install required ROS packages

```bash
sudo apt-get install ros-noetic-octomap
sudo apt-get install ros-noetic-octomap-msgs
sudo apt-get install ros-noetic-octomap-ros
sudo apt-get install ros-noetic-octovis
```

3. Download all Gazebo models (optional)

```
cd ~/.gazebo/
mkdir -p models
cd models
wget http://file.ncnynl.com/ros/gazebo_models.txt
wget -i gazebo_models.txt
ls model.tar.g* | xargs -n1 tar xzvf
```

4. Clone and build this repo:

```
git clone git@github.com:Amos-Chen98/drone_ws.git
cd drone_ws
catkin build
```

5. Configure the environment variables

Add the following lines to `.bashrc`

```bash
source <path_to_drone_ws>/devel/setup.bash
source ~/PX4-Autopilot/Tools/setup_gazebo.bash ~/PX4-Autopilot ~/PX4-Autopilot/build/px4_sitl_default
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/PX4-Autopilot
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/PX4-Autopilot/Tools/sitl_gazebo
export GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH:/usr/lib/x86_64-linux-gnu/gazebo-11/plugins
```

## Usage

### Trajectory tracking demo

updated 01/17/2023.

Step1: launch QGC, and launch the simulator and Mavros using the following command.

```bash
roslaunch px4_controller run_simulator.launch 
```

Step2: use the `take off` command in QGC to make the drone take off.

Step3: start trajecory tracking.

```
rosrun px4_controller traj_tracking.py
```

### Generate octomap from Gazebo world

updated 01/23/2023

This function is based on package `sim_gazebo_plugins`. To use the plugin, you need to edit your desired .world file to recognize the plugin. Simply open your .world file in a text editor and add the following line just before the final `<world>` tag (i. e. in between the `<world>` tags):

```xml
<plugin name='gazebo_octomap' filename='libBuildOctomapPlugin.so'/>
```

To build an octomap .bt file, open three separate terminals:

```
In one terminal:
$ roscore

In second terminal:
$ rosrun gazebo_ros gazebo <your_world_file>.world
**NOTE: replace <your_world_file> with the filename of the world you wish to build a map of**

In third terminal, once Gazebo has loaded the world above:
$ rosservice call /world/build_octomap '{bounding_box_origin: {x: 0, y: 0, z: 15}, bounding_box_lengths: {x: 30, y: 30, z: 30}, leaf_size: 0.5, filename: output_filename.bt}'
```

Note that the above rosservice call has a few adjustable variables. The bounding box origin can be set as desired (in meters) as well as the bounding box lengths (in meters) relative to the bounding box origin. The bounding box lengths are done in both (+/-) directions relative to the origin. For example, in the `rosservice` call above, from `(0, 0, 0)`, our bounding box will start at **-15 meters** and end at **+15 meters** in the X and Y directions. In the Z direction, we will start at **0 meters** and end at **30 meters**.
