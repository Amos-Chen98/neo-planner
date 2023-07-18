# drone_ws
## About

This project (as a ROS workspace) provides a motion planning framework for drones based on ROS and PX4-SITL.

## Installation

### Prerequisites

1. Install required framework and softwares.

   Before using this project, please make sure the following dependencies have been successfully installed and configured.

* ROS1 with Gazebo: https://wiki.ros.org/noetic/Installation/Ubuntu
* PX4-Autopilot:

```bash
git clone https://github.com/PX4/PX4-Autopilot.git
cd PX4-Autopilot
git checkout v1.13.2
git submodule update --init --recursive
bash Tools/setup/ubuntu.sh
make px4_sitl gazebo
```

Ref:  https://docs.px4.io/main/en/dev_setup/dev_env_linux_ubuntu.html

* Mavros: https://docs.px4.io/main/en/ros/mavros_installation.html

* QGC: https://docs.qgroundcontrol.com/master/en/getting_started/download_and_install.html

  **Note: Remember to enable Virtual Joystick in Application Settings in QGC, otherwise, the drone will refuse to enter OFFBOARD mode.**

2. Install required packages

```bash
sudo apt-get install ros-noetic-octomap*
sudo apt-get install ros-noetic-octovis
```

```
pip install octomap-python
pip install pyquaternion
pip install scipy
pip install transitions
```

3. Install dependencies for neural network

```
pip install onnx
pip install onnxruntime-gpu
```

4. Download all Gazebo models (optional)

```
cd ~/.gazebo/
mkdir -p models
cd models
wget http://file.ncnynl.com/ros/gazebo_models.txt
wget -i gazebo_models.txt
ls model.tar.g* | xargs -n1 tar xzvf
```

### Install this project (as a ROS workspace)

1. Clone and build this repo:

```
git clone git@github.com:Amos-Chen98/drone_ws.git
cd drone_ws
catkin build
```

2. Configure the environment variables: Add the following lines to `.bashrc`

```bash
source <path_to_drone_ws>/devel/setup.bash
source ~/PX4-Autopilot/Tools/setup_gazebo.bash ~/PX4-Autopilot ~/PX4-Autopilot/build/px4_sitl_default
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/PX4-Autopilot
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/PX4-Autopilot/Tools/sitl_gazebo
export GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH:/usr/lib/x86_64-linux-gnu/gazebo-11/plugins
```

## Usage

### 1. Generate octomap from Gazebo world

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
$ rosservice call /world/build_octomap '{bounding_box_origin: {x: 0, y: 0, z: 15}, bounding_box_lengths: {x: 30, y: 30, z: 30}, leaf_size: 0.1, filename: output_filename.bt}'
```

Note that the above rosservice call has a few adjustable variables. The bounding box origin can be set as desired (in meters) as well as the bounding box lengths (in meters) relative to the bounding box origin. The bounding box lengths are done in both (+/-) directions relative to the origin. For example, in the `rosservice` call above, from `(0, 0, 0)`, our bounding box will start at **-15 meters** and end at **+15 meters** in the X and Y directions. In the Z direction, we will start at **0 meters** and end at **30 meters**.

### 2. Trajectory tracking

updated 02/07/2023.

Step 1: launch QGC, and launch the simulator and Mavros using the following command.

```bash
roslaunch simulator sim_empty.launch 
```

Step 2: use the `take off` command in QGC to make the drone take off.

Step 3: start trajecory tracking.

```
rosrun px4_controller traj_tracking.py
```

Then the drone will follow an '8' pattern with time-varying velocity.

### 3. Online octomap building

updated 02/07/2023.

Step 1: launch QGC, and run the following line.

```bash
roslaunch simulator sim_onboard.launch 
```

Step 2: use the Virtual Joystick to move the drone to generate octomap.

### 4. Trajectory planning and tracking

updated 05/18/2023.

Step 1: launch QGC, 

Step 2: Launch Gazebo simulator, Mavros, and other related nodes.

​	Use globally known maps:

```bash
roslaunch simulator sim_global.launch 
```

​	Or, construct the map incrementally based on onboard senseing:

```
roslaunch simulator sim_onboard.launch 
```

Step 3: 

In one terminal:

```
rosrun planner traj_planner_node.py 
```

In another terminal: 

```
rosrun planner manager_node.py
```

The above command will take the drone off and run the FSM.

Step 4: Set goal point with `2D Nav Goal` in RViz, and you will see the drone perform trajectory planning and tracking.



`rqt_graph` when using onboard senseing:

![](https://raw.githubusercontent.com/Amos-Chen98/Image_bed/main/2023/202304262012099.png)
