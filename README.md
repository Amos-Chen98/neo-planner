# **drone_ws**

## **About**

This project (as a ROS workspace) provides a motion planning framework for drones based on ROS and PX4-SITL.

## **Installation**

### **Prerequisites**

Before using this project, please make sure the following dependencies have been successfully installed and configured.

- ROS1 with Gazebo: https://wiki.ros.org/noetic/Installation/Ubuntu
- PX4-Autopilot:

```bash
git clone https://github.com/PX4/PX4-Autopilot.git
cd PX4-Autopilot
git reset --hard v1.13.2  # harder than git checkout
git submodule update --init --recursive
bash Tools/setup/ubuntu.sh
make px4_sitl gazebo
```

Ref:  https://docs.px4.io/main/en/dev_setup/dev_env_linux_ubuntu.html

- Mavros: https://docs.px4.io/main/en/ros/mavros_installation.html

- QGC: https://docs.qgroundcontrol.com/master/en/getting_started/download_and_install.html

  **Note: Remember to enable Virtual Joystick in Application Settings in QGC, otherwise, the drone will refuse to enter OFFBOARD mode.**

- PyTorch (GPU): https://pytorch.org/get-started/locally/

Besides, install the following dependencies.

```bash
sudo apt install ros-noetic-octomap*
sudo apt install ros-noetic-octovis
pip install octomap-python
pip install pyquaternion
pip install scipy
sudo apt-get install graphviz graphviz-dev
pip install transitions[diagrams]
pip install onnx
pip install onnxruntime-gpu
pip install torchinfo
pip install torchvision
```

[Optional] Download all Gazebo models:

```bash
cd ~/.gazebo/
mkdir models
cd models
wget http://file.ncnynl.com/ros/gazebo_models.txt
wget -i gazebo_models.txt
ls model.tar.g* | xargs -n1 tar xzvf
rm model.tar.g*
```

### **Install this project (as a ROS workspace)**

1. Clone and build this repo:

```bash
git clone https://github.com/Amos-Chen98/drone_ws.git
cd drone_ws
catkin build
```

1. Configure the environment variables: Add the following lines to `.bashrc`/`.zshrc`

```bash
alias drone_ws_go='source <path_to_drone_ws>/devel/setup.bash;
source ~/PX4-Autopilot/Tools/setup_gazebo.bash ~/PX4-Autopilot ~/PX4-Autopilot/build/px4_sitl_default;
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/PX4-Autopilot;
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/PX4-Autopilot/Tools/sitl_gazebo;
export GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH:/usr/lib/x86_64-linux-gnu/gazebo-11/plugins'
```

Run `drone_ws_go`  in shell to activate.

## **Usage**

### **1. Trajectory planning and tracking**

updated 08/16/2023.

Step 1: launch QGC

Step 2: Launch the following files:

```bash
roslaunch simulator sim_onboard.launch
roslaunch planner map_server_onboard.launch
roslaunch planner planner.launch
roslaunch planner manager.launch
```

Step 3: Set goal point with `2D Nav Goal` in RViz, and you will see the drone perform trajectory planning and tracking.

### **2. Generate octomap from Gazebo world**

If you want to generate ground truth octomap for test or training, follow the instructions.

updated 01/23/2023

This function is based on package `sim_gazebo_plugins`. To use the plugin, you need to edit your desired .world file to recognize the plugin. Simply open your .world file in a text editor and add the following line just before the final `<world>` tag (i. e. in between the `<world>` tags):

```
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

## 3. Object tracking

updated 08/20/2023

This is an application of the planner: using the planner to perform object tracking while avoiding obstacles.

Step 1: launch QGC

Step 2: Launch the following files:

```bash
roslaunch simulator sim_onboard.launch
roslaunch planner map_server_onboard.launch
roslaunch roslaunch planner tracker_planner.launch
roslaunch roslaunch planner tracker_manager.launch
```

By default, the planner takes in the moving object's pose through the topic `/move_base_simple/goal`. You can dynamically send the target pose to this topic for tracking.
