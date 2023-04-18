'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2023-04-18 11:43:59
'''
import os
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_path)
import datetime
import pandas as pd
import tf
from mavros_msgs.srv import SetMode, SetModeRequest
from esdf import ESDF
from nav_msgs.msg import Path, OccupancyGrid
import time
from pyquaternion import Quaternion
from traj_planner import MinJerkPlanner
from mavros_msgs.srv import SetMode
from mavros_msgs.msg import State, PositionTarget
import numpy as np
import rospy
from visualization_msgs.msg import MarkerArray
from visualizer import Visualizer
from nav_msgs.msg import Odometry
from matplotlib import pyplot as plt
from std_msgs.msg import String, Float64
import pprint


class Config():
    def __init__(self):
        self.v_max = rospy.get_param("~v_max", 1.0)
        self.T_min = rospy.get_param("~T_min", 0.5)
        self.T_max = rospy.get_param("~T_max", 5.0)
        self.safe_dis = rospy.get_param("~safe_dis", 0.5)
        self.delta_t = rospy.get_param("~delta_t", 0.1)
        self.weights = rospy.get_param("~weights", [1, 1, 1, 10000])
        self.init_seg_len = rospy.get_param("~init_seg_len", 2.0)  # the initial length of each segment
        self.init_T = rospy.get_param("~init_T", 2.5)  # the initial T of each segment


class DroneState():
    def __init__(self):
        self.global_pos = np.zeros(3)
        self.global_vel = np.zeros(3)
        self.local_vel = np.zeros(3)
        self.attitude = np.zeros(4)


class TrajPlanner():
    def __init__(self, node_name="traj_planner"):
        # Node
        rospy.init_node(node_name, anonymous=False)

        # Members
        self.odom = None
        self.drone_state = DroneState()
        planner_config = Config()
        self.planner = MinJerkPlanner(planner_config)
        self.state_cmd = PositionTarget()
        self.state_cmd.coordinate_frame = 1
        self.ODOM_RECEIVED = False
        self.des_path = Path()
        self.real_path = Path()
        self.map = ESDF()
        self.visualizer = Visualizer()
        self.fsm_trigger = String()
        self.target_state = None
        self.target_reach_threshold = 0.2
        self.tracking_flag = False  # if the drone is tracking a target, this flag is True

        # Services
        rospy.wait_for_service("/mavros/set_mode")
        self.set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)

        # Subscribers
        self.flight_state_sub = rospy.Subscriber('/mavros/state', State, self.flight_state_cb)
        self.occupancy_map_sub = rospy.Subscriber('/projected_map', OccupancyGrid, self.map.occupancy_map_cb)
        self.odom_sub = rospy.Subscriber('/mavros/local_position/odom', Odometry, self.odom_cb)
        self.target_sub = rospy.Subscriber('/manager/local_target', PositionTarget, self.move, queue_size=1)  # when a new target is received, move

        # Publishers
        self.local_pos_cmd_pub = rospy.Publisher("/mavros/setpoint_raw/local", PositionTarget, queue_size=10)
        self.drone_snapshots_pub = rospy.Publisher('/robotMarker', MarkerArray, queue_size=10)
        self.des_wpts_pub = rospy.Publisher('/des_wpts', MarkerArray, queue_size=10)
        self.des_path_pub = rospy.Publisher('/des_path', MarkerArray, queue_size=10)
        self.fsm_trigger_pub = rospy.Publisher('/manager/trigger', String, queue_size=10)
        self.speed_plot_pub = rospy.Publisher('/drone_speed', Float64, queue_size=10)

        rospy.loginfo("Global planner initialized")

        # Initialize the csv file collecting training data
        self.table_header = ['id',
                             'drone_vel_x',
                             'drone_vel_y',
                             'drone_vel_z',
                             'drone_attitude_w',
                             'drone_attitude_x',
                             'drone_attitude_y',
                             'drone_attitude_z',
                             'target_pos_x',
                             'target_pos_y',
                             'target_pos_z',
                             'target_vel_x',
                             'target_vel_y',
                             'target_vel_z',
                             ]

        # create a blank csv file, with (1+3+4+3+3) columns
        df = pd.DataFrame(columns=self.table_header)
        df.to_csv('train.csv', index=False)

    def flight_state_cb(self, data):
        self.flight_state = data

    def odom_cb(self, data):
        '''
        1. store the drone's global status
        2. publish dynamic tf transform from map frame to camera frame
        (Currently, regard camera frame as drone body frame)
        '''
        self.ODOM_RECEIVED = True
        self.odom = data
        local_pos = np.array([data.pose.pose.position.x,
                              data.pose.pose.position.y,
                              data.pose.pose.position.z])
        global_pos = local_pos
        local_vel = np.array([data.twist.twist.linear.x,
                              data.twist.twist.linear.y,
                              data.twist.twist.linear.z])
        quat = Quaternion(data.pose.pose.orientation.w,
                          data.pose.pose.orientation.x,
                          data.pose.pose.orientation.y,
                          data.pose.pose.orientation.z)  # from local to global
        global_vel = quat.rotate(local_vel)
        self.drone_state.global_pos = global_pos
        self.drone_state.global_vel = global_vel
        self.drone_state.local_vel = local_vel
        self.drone_state.attitude = quat

        if (self.tracking_flag == True and np.linalg.norm(self.drone_state.global_pos[:2] - self.target_state[0]) < self.target_reach_threshold):
            self.tracking_cmd_timer.shutdown()
            self.tracking_flag = False
            rospy.loginfo("Trajectory execution finished!")
            self.fsm_trigger.data = "reach_target"
            self.fsm_trigger_pub.publish(self.fsm_trigger)

        self.speed_plot_pub.publish(np.linalg.norm(global_vel))

    def move(self, target):
        print(" ")
        rospy.loginfo("Target: x = %f, y = %f, vel_x = %f, vel_y = %f",
                      target.position.x, target.position.y, target.velocity.x, target.velocity.y)
        self.target_state = np.array([[target.position.x, target.position.y],
                                      [target.velocity.x, target.velocity.y]])

        while not self.ODOM_RECEIVED:
            time.sleep(0.01)

        self.traj_plan()
        self.traj_track()
        self.visualize_des_wpts()
        self.visualize_des_path()
        # self.plot_state_curve()

    def traj_plan(self):
        drone_state = self.drone_state
        drone_state_2d = np.array([drone_state.global_pos[:2],
                                   drone_state.global_vel[:2]])
        self.des_pos_z = drone_state.global_pos[2]  # use current height

        time_start = time.time()
        self.planner.plan(self.map, drone_state_2d, self.target_state)  # 2D planning, z is fixed
        time_end = time.time()
        self.planning_time = time_end - time_start
        rospy.loginfo("Planning finished! Time cost: %f", self.planning_time)

        # get planning results
        int_wpts = self.planner.int_wpts
        ts = self.planner.ts

        # convert int_wpts to body frame
        int_wpts_num = int_wpts.shape[1]
        int_wpts_local = np.zeros((3, int_wpts_num))
        for i in range(int_wpts_num):
            int_wpts_3d = np.array([int_wpts[0, i], int_wpts[1, i], self.des_pos_z])
            int_wpts_local[:, i] = drone_state.attitude.inverse.rotate(int_wpts_3d - drone_state.global_pos)

        print("int_wpts_local: ", int_wpts_local.T)
        print("ts: ", ts)

        self.record_train_input(drone_state)  # record training data

    def record_train_input(self, drone_state):
        '''
        Record training data and write to csv file
        '''
        now = datetime.datetime.now()
        timestamp = int(now.strftime("%Y%m%d%H%M%S"))

        # get training data
        drone_local_vel = drone_state.local_vel  # size: (3,)
        drone_quat = drone_state.attitude  # size: (4,)
        drone_attitude = np.array([drone_quat.w, drone_quat.x, drone_quat.y, drone_quat.z])  # size: (4,)
        target_state_3d = np.zeros((2, 3))
        target_state_3d[:, 0:2] = self.target_state
        target_state_3d[0, 2] = self.des_pos_z
        target_local_pos = drone_quat.inverse.rotate(target_state_3d[0] - drone_state.global_pos)  # size: (3,)
        target_local_vel = drone_quat.inverse.rotate(target_state_3d[1] - drone_state.global_vel)  # size: (3,)

        # write the data to local file
        train_data = np.concatenate((np.array([timestamp]), drone_local_vel, drone_attitude, target_local_pos, target_local_vel), axis=0)
        df = pd.read_csv('train.csv')
        df = pd.concat([df, pd.DataFrame(train_data.reshape(1, -1), columns=self.table_header)], ignore_index=True)
        df.to_csv('train.csv', index=False)
        rospy.loginfo("Training data (ID: %d) saved!", timestamp)

    def warm_up(self):
        # Send a few setpoints before switching to OFFBOARD mode
        self.state_cmd.position.x = self.drone_state.global_pos[0]
        self.state_cmd.position.y = self.drone_state.global_pos[1]
        self.state_cmd.position.z = self.drone_state.global_pos[2]
        rate = rospy.Rate(100)
        for _ in range(5):  # set 5 points
            if (rospy.is_shutdown()):
                break
            self.local_pos_cmd_pub.publish(self.state_cmd)
            rate.sleep()

    def traj_track(self):
        '''
        When triggered, start to publish full state cmd
        '''
        # if not in OFFBOARD mode, switch to OFFBOARD mode
        if self.flight_state.mode != "OFFBOARD":
            self.warm_up()
            set_offb_req = SetModeRequest()
            set_offb_req.custom_mode = 'OFFBOARD'
            if (self.set_mode_client.call(set_offb_req).mode_sent == True):
                rospy.loginfo("OFFBOARD enabled")

        self.tracking_flag = True
        self.fsm_trigger.data = "start_tracking"
        self.fsm_trigger_pub.publish(self.fsm_trigger)

        self.des_state, self.traj_time, hz = self.planner.get_full_state_cmd()

        # calculate the best index to start tracking by minimizing the distance between current state and desired state
        current_pos = self.drone_state.global_pos[:2]
        search_idx = np.arange(0, len(self.des_state), 30)
        pos_err = np.zeros(len(search_idx))
        for i in range(len(search_idx)):
            pos_err[i] = np.linalg.norm(current_pos - self.des_state[search_idx[i]][0])
        self.des_state_index = search_idx[np.argmin(pos_err)]
        rospy.loginfo("Trajectory duration: %f", len(self.des_state)/hz)
        rospy.loginfo("Start time offset: %f", self.des_state_index/hz)
        # print the first element of des vel
        rospy.loginfo("Init vel in traj: %f", np.linalg.norm(self.des_state[0][1]))
        rospy.loginfo("Terminal vel in traj: %f", np.linalg.norm(self.des_state[-1][1]))
        self.tracking_cmd_timer = rospy.Timer(rospy.Duration(1/hz), self.tracking_cmd_timer_cb)

    def tracking_cmd_timer_cb(self, event):
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

        self.state_cmd.yaw = np.arctan2(self.des_state[self.des_state_index][0][1] - self.des_state[self.des_state_index - 1][0][1],
                                        self.des_state[self.des_state_index][0][0] - self.des_state[self.des_state_index - 1][0][0])

        self.local_pos_cmd_pub.publish(self.state_cmd)

        if self.des_state_index < len(self.des_state)-1:
            self.des_state_index += 1

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
        pos_array = self.planner.int_wpts  # shape: (2,n)
        pos_array = np.vstack((pos_array, self.des_pos_z * np.ones([1, pos_array.shape[1]]))).T
        des_wpts = self.visualizer.get_marker_array(pos_array, 2, 0.4)
        self.des_wpts_pub.publish(des_wpts)

    def visualize_des_path(self):
        '''
        Visualize the desired path, where high-speed pieces and low-speed pieces are colored differently
        '''
        pos_array = self.planner.get_pos_array()
        pos_array = np.hstack((pos_array, self.des_pos_z * np.ones([len(pos_array), 1])))
        vel_array = np.linalg.norm(self.planner.get_vel_array(), axis=1)  # shape: (n,)
        des_path = self.visualizer.get_path(pos_array, vel_array)
        self.des_path_pub.publish(des_path)

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


if __name__ == "__main__":

    traj_planner = TrajPlanner()

    rospy.spin()
