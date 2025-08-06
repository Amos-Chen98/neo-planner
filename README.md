# **NEO-Planner**

## 1 **About**

This is the code repository for the IROS'25 paper:

**Learning to Initialize Trajectory Optimization for Vision-Based Autonomous Flight in Unknown Environments**

Video: https://youtu.be/UoroRe-euDk

![Simulation Demo](media/sim.gif)

![Experiment Demo](media/exp.gif)

This repository (as a ROS workspace) provides a test environment for drone navigation based on ROS and PX4 software-in-the-loop (SITL).

## **2 Installation**

This project has been tested on Ubuntu 20.04 with ROS Noetic.

### 2.1 **Dependencies**

Please install the dependencies following each link.

* PX4, ROS1, and MAVROS: https://docs.px4.io/main/en/ros/mavros_installation.html

- QGroundControl (QGC): https://docs.qgroundcontrol.com/master/en/qgc-user-guide/getting_started/download_and_install.html#ubuntu

  **Note:** Remember to enable Virtual Joystick in Application Settings in QGC, otherwise, the drone will refuse to enter OFFBOARD mode.

- PyTorch (GPU): https://pytorch.org/get-started/locally

Besides, install the following dependencies.

```
sudo apt install ros-noetic-octomap* ros-noetic-octovis graphviz graphviz-dev
pip install octomap-python pyquaternion scipy transitions[diagrams] onnx onnxruntime-gpu torchinfo torchvision
```

### 2.2 Install this project (as a ROS workspace)

1. Clone and build this repository:

```bash
git clone https://github.com/Amos-Chen98/neo-planner.git
cd neo-planner
catkin build
```

2. Configure the environment variables: Add the following lines to `.bashrc`/`.zshrc`

```bash
source ~/PX4-Autopilot/Tools/simulation/gazebo-classic/setup_gazebo.bash ~/PX4-Autopilot ~/PX4-Autopilot/build/px4_sitl_default
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/PX4-Autopilot
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/PX4-Autopilot/Tools/simulation/gazebo-classic/sitl_gazebo-classic
```

Ref: https://docs.px4.io/main/en/simulation/ros_interface.html#launching-gazebo-classic-with-ros-wrappers

## 3 **Usage**

Ensure **QGC** is running before launching any ROS nodes for this project.

### 3.1 Quick start

#### 3.1.1 Autonomous navigation

Step 1: Launch the following file:

```bash
roslaunch planner bringup.launch
```

The drone will automatically take off and enter HOVER mode.

Step 2: Set a goal point with `2D Nav Goal` in RViz. Or, if you want to set a precise goal point, use the ROS command:

```
rostopic pub /move_base_simple/goal geometry_msgs/PoseStamped '{header: {stamp: now, frame_id: "map"}, pose: {position: {x: 30.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}'
```

Then the drone will perform trajectory planning and tracking.

The above Step 1-2 is equal to running the following commands:

```
roscd planner
./scripts/bash/demo.sh
```

**Configurable parameters**:

The parameters of the planner node is defined in `src/planner/launch/config/planner_config.yaml`

The parameters of the manager node is defined in `src/planner/launch/config/manager_config.yaml`

The parameters of the octomap_server is defined in `src/planner/launch/map_server_onboard.launch`

#### 3.1.2 Object tracking

This is an extended application of the planner: using the planner to perform object tracking while avoiding obstacles.

Launch the following files:

```bash
roslaunch simulator sim_onboard.launch
roslaunch planner map_server_onboard.launch
roslaunch planner tracker_planner.launch
roslaunch planner tracker_manager.launch
```

By default, the planner takes in the moving object's pose through the topic `/move_base_simple/goal`. You can dynamically send the target pose to this topic for tracking.

### 3.2 Customized development

#### 3.2.1 Batch random generation of Gazebo world

Run `src/simulator/scripts/generate_worlds.py`

This command will automatically generate a batch of gazebo world files.

The configurable parameters are listed in `src/simulator/scripts/generator_config.yaml`

#### 3.2.2 Generate ground-truth pointcloud and octomap from Gazebo world

This function is based on the package `sim_gazebo_plugins`. To use the plugin, you need to edit your desired .world file to incorporate the plugin. Simply open your .world file in a text editor and add the following line just before the final `<world>` tag (i. e. in between the `<world>` tags):

```
<plugin name='gazebo_octomap' filename='libBuildOctomapPlugin.so'/>
```

**Note:** If you are using the existing world files, or any worlds generated from `generate_worlds.py`, this step is not required because the plugin has already been included

To generate a .pcd file and a .bt file , execute the following commands:

```bash
roslaunch simulator load_world.launch gazebo_world:=<your_world_file>
# Replace <your_world_file> with the filename of the world you wish to build a map from. This name should not not contain ".world"

rosservice call /world/build_octomap '{bounding_box_origin: {x: 0, y: 0, z: 15}, bounding_box_lengths: {x: 30, y: 30, z: 30}, leaf_size: 0.1, filename: output_filename.bt}'
```

Then the .pcd file and the .bt file will be generated in your .ros folder under the home folder.

The `rosservice` call includes adjustable variables: the bounding box origin and lengths, both in meters. 

The lengths extend symmetrically in both (+/-) directions from the origin. For example, with an origin at `(0, 0, 0)` and bounding_box_lengths `{x: 30, y: 30, z: 30}`, the bounding box spans **-15 to +15 meters** in the X and Y directions, and **0 to 30 meters** in the Z direction.

#### 3.2.3 Collect data and train your own network

**(1) Use the expert planner to collect data**

Launch the following files:

```bash
roslaunch simulator sim_global.launch
roslaunch planner map_server_global.launch
roslaunch planner manager.launch mission_mode:=random
roslaunch planner planner.launch selected_planner:=record
```

Set a goal point with `2D Nav Goal` in RViz to trigger the first mission. When the drone reaches its goal, the manager node automatically samples a new goal. The system runs continuously, saving training samples to a local directory. You could end it anytime by killing the nodes.

**(2) Train your network based on the collected data**

Run `src/planner/scripts/nn_trainer/nn_trainer.py` with your customized file paths.

