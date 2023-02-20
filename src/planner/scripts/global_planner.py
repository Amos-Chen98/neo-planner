#! /usr/bin/env python
import os
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_path)
from matplotlib import pyplot as plt
from nav_msgs.msg import Odometry
from visualizer import Visualizer
from visualization_msgs.msg import MarkerArray
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, TwistStamped, Point, Vector3
from mavros_msgs.msg import Altitude, ExtendedState, HomePosition, State, WaypointList, PositionTarget
from mavros_msgs.srv import CommandBool, ParamGet, SetMode, WaypointClear, WaypointPush
from sensor_msgs.msg import NavSatFix, Imu
from octomap_msgs.msg import Octomap
from traj_planner import MinJerkPlanner
from pyquaternion import Quaternion
import time
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import TransformStamped
import tf2_ros
from ESDF import ESDF



class Config():
    def __init__(self):
        self.v_max = rospy.get_param("~v_max", 5.0)
        self.T_min = rospy.get_param("~T_min", 1.0)
        self.T_max = rospy.get_param("~T_max", 20)
        self.safe_dis = rospy.get_param("~safe_dis", 0.3)
        self.kappa = rospy.get_param("~kappa", 50)
        # self.weights = rospy.get_param("~weights", [1.0, 1.0, 0.001, 1])
        self.weights = rospy.get_param("~weights", [10, 1.0, 0, 10000])


class GlobalPlanner():
    def __init__(self, node_name="global_planner"):
        # Node
        rospy.init_node(node_name, anonymous=False)

        # Members
        self.octomap = None
        self.odom = None
        self.drone_state = np.zeros((3, 3))  # p,v,a in inertia frame
        planner_config = Config()
        self.planner = MinJerkPlanner(planner_config)
        self.state_cmd = PositionTarget()
        self.state_cmd.coordinate_frame = 1
        self.ODOM_RECEIVED = False
        self.des_path = Path()
        self.real_path = Path()
        self.map = ESDF()
        self.visualizer = Visualizer()

        # Subscribers
        self.occupancy_map_sub = rospy.Subscriber('/projected_map', OccupancyGrid, self.map.occupancy_map_cb)
        self.odom_sub = rospy.Subscriber('/mavros/local_position/odom', Odometry, self.odom_cb)

        # Publishers
        self.local_pos_cmd_pub = rospy.Publisher("/mavros/setpoint_raw/local", PositionTarget, queue_size=10)
        self.drone_snapshots_pub = rospy.Publisher('/robotMarker', MarkerArray, queue_size=10)
        self.des_wpts_pub = rospy.Publisher('/des_wpts', MarkerArray, queue_size=10)
        self.des_path_pub = rospy.Publisher('/des_path', MarkerArray, queue_size=10)
        # self.des_path_pub = rospy.Publisher('/des_path', Path, queue_size=10)

        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

    def odom_cb(self, data):
        '''
        1. store the drone's global status
        2. publish dynamic tf transform from map frame to camera frame
        (Currently, regard camera frame as drone body frame)
        '''
        # get global drone_state
        self.ODOM_RECEIVED = True
        self.odom = data
        local_pos = np.array([data.pose.pose.position.x,
                              data.pose.pose.position.y,
                              data.pose.pose.position.z])
        global_pos = local_pos
        local_vel = np.array([data.twist.twist.linear.x,
                              data.twist.twist.linear.y,
                              data.twist.twist.linear.z,
                              ])
        global_vel = local_vel
        # quat = Quaternion(data.pose.pose.orientation.w,
        #                   data.pose.pose.orientation.x,
        #                   data.pose.pose.orientation.y,
        #                   data.pose.pose.orientation.z)
        # global_vel = quat.inverse.rotate(local_vel)
        self.drone_state[0] = global_pos
        self.drone_state[1] = global_vel

    def plan(self, tail_state):
        while not self.ODOM_RECEIVED:
            time.sleep(0.01)

        drone_state_2d = self.drone_state[:, 0:2]
        self.des_pos_z = self.drone_state[0][2]  # use current height
        self.planner.plan(self.map, drone_state_2d, tail_state)  # 2D planning, z is fixed

        rospy.loginfo("Trajectory planning finished!")

    def warm_up(self):
        # Send a few setpoints before switching to OFFBOARD mode
        self.state_cmd.position.x = self.drone_state[0][0]
        self.state_cmd.position.y = self.drone_state[0][1]
        self.state_cmd.position.z = self.drone_state[0][2]
        rate = rospy.Rate(100)
        for _ in range(5):  # set 5 points
            if (rospy.is_shutdown()):
                break
            self.local_pos_cmd_pub.publish(self.state_cmd)
            rate.sleep()

    def publish_state_cmd(self):
        '''
        When triggered, start to publish full state cmd
        '''
        self.des_state, self.traj_time, hz = self.planner.get_full_state_cmd()
        self.des_state_index = 0
        self.start_time = rospy.get_time()
        self.timer = rospy.Timer(rospy.Duration(1/hz), self.timer_cb)

    def timer_cb(self, event):
        '''
        Publish state cmd, height is fixed to current height
        '''
        self.state_cmd.position.x = self.des_state[self.des_state_index][0][0]
        self.state_cmd.position.y = self.des_state[self.des_state_index][0][1]
        self.state_cmd.position.z = self.des_pos_z

        self.state_cmd.velocity.x = self.des_state[self.des_state_index][1][0]
        self.state_cmd.velocity.y = self.des_state[self.des_state_index][1][1]
        self.state_cmd.velocity.z = 0

        self.state_cmd.acceleration_or_force.x = self.des_state[self.des_state_index][2][0]
        self.state_cmd.acceleration_or_force.y = self.des_state[self.des_state_index][2][1]
        self.state_cmd.acceleration_or_force.z = 0

        self.state_cmd.yaw = 0

        self.local_pos_cmd_pub.publish(self.state_cmd)

        self.des_state_index += 1 if self.des_state_index < len(self.des_state)-1 else 0

        # if event.current_real.to_sec() - self.start_time > self.traj_time:
        #     self.des_state_index -= 1  # keep publishing the last des_state
        #     # self.timer.shutdown()

    def visualize_drone_snapshots(self):
        '''
        publish snapshots of drone model (mesh) along the trajectory
        '''
        pos_array = self.planner.get_pos_array()
        pos_array = np.hstack((pos_array, self.des_pos_z * np.ones([len(pos_array), 1])))
        drone_snapshots = self.visualizer.get_marker_array(pos_array, 10, 2)
        self.drone_snapshots_pub.publish(drone_snapshots)
        rospy.loginfo("Drone_snapshots published!")

    def visualize_des_wpts(self):
        '''
        Visualize the desired waypoints as markers
        '''
        # time_start = time.time()
        pos_array = self.planner.int_wpts  # shape: (2,n)
        pos_array = np.vstack((pos_array, self.des_pos_z * np.ones([1, pos_array.shape[1]]))).T
        des_wpts = self.visualizer.get_marker_array(pos_array, 2, 0.4)
        self.des_wpts_pub.publish(des_wpts)
        rospy.loginfo("Desired waypoints published!")
        # time_end = time.time()
        # print("time cost of visualize_des_wpts: ", time_end - time_start)

    def visualize_des_path(self):
        '''
        Visualize the desired path, where high-speed pieces and low-speed pieces are colored differently
        '''
        # time_start = time.time()
        pos_array = self.planner.get_pos_array()
        pos_array = np.hstack((pos_array, self.des_pos_z * np.ones([len(pos_array), 1])))
        vel_array = np.linalg.norm(self.planner.get_vel_array(), axis=1)  # shape: (n,)
        des_path = self.visualizer.get_path(pos_array, vel_array)
        self.des_path_pub.publish(des_path)
        rospy.loginfo("Desired path published!")
        # time_end = time.time()
        # print("time cost of visualize_des_path: ", time_end - time_start)

    def plot_state_curve(self):
        # delete all existing plots
        plt.close('all')

        final_ts = self.planner.ts
        t_samples = np.arange(0, sum(final_ts), 0.1)
        t_cum_array = np.cumsum(final_ts)
        vel = self.planner.get_vel_array()
        acc = self.planner.get_acc_array()
        jer = self.planner.get_jer_array()

        # get the norm of vel, acc and jer
        vel_norm = np.linalg.norm(vel, axis=1)
        acc_norm = np.linalg.norm(acc, axis=1)
        jer_norm = np.linalg.norm(jer, axis=1)

        plt.figure("Vel, Acc, Jerk by axis")
        plt.plot(t_samples, vel[:, 0], label='Vel_x')
        plt.plot(t_samples, vel[:, 1], label='Vel_y')
        plt.plot(t_samples, acc[:, 0], label='Acc_x')
        plt.plot(t_samples, acc[:, 1], label='Acc_y')
        plt.plot(t_samples, jer[:, 0], label='Jerk_x')
        plt.plot(t_samples, jer[:, 1], label='Jerk_y')
        plt.xlabel('t/s')
        plt.legend()
        plt.grid()

        plt.figure("Vel, Acc, Jerk magnitude")
        plt.plot(t_samples, vel_norm, label='Vel')
        plt.plot(t_samples, acc_norm, label='Acc')
        plt.plot(t_samples, jer_norm, label='Jerk')
        plt.vlines(t_cum_array, 0, np.max(vel_norm))  # mark the wpts
        plt.xlabel('t/s')
        plt.legend()
        plt.grid()

        plt.show()
