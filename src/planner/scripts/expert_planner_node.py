'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2023-04-24 15:18:03
'''
import os
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_path)
from planner.msg import *
import actionlib
from nav_msgs.msg import Odometry, Path, OccupancyGrid
import datetime
import pandas as pd
from esdf import ESDF
import time
from pyquaternion import Quaternion
from traj_planner import MinJerkPlanner
from mavros_msgs.srv import SetMode, SetModeRequest
from mavros_msgs.msg import State, PositionTarget
import numpy as np
import rospy
from visualization_msgs.msg import MarkerArray
from visualizer import Visualizer
from matplotlib import pyplot as plt
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class PlannerConfig():
    def __init__(self):
        self.v_max = rospy.get_param("~v_max", 1.0)
        self.T_min = rospy.get_param("~T_min", 0.5)
        self.T_max = rospy.get_param("~T_max", 5.0)
        self.safe_dis = rospy.get_param("~safe_dis", 0.5)
        self.delta_t = rospy.get_param("~delta_t", 0.1)
        self.weights = rospy.get_param("~weights", [1, 1, 1, 10000])
        self.init_wpts_mode = rospy.get_param("~init_wpts_mode", 'fixed')  # 'fixed' or 'adaptive'
        self.init_seg_len = rospy.get_param("~init_seg_len", 2.0)  # the initial length of each segment
        self.init_wpts_num = rospy.get_param("~init_wpts_num", 2)  # the initial number of waypoints
        self.init_T = rospy.get_param("~init_T", 2.5)  # the initial T of each segment


class MissionConfig():
    def __init__(self):
        self.planning_mode = rospy.get_param("~planning_mode", 'online')  # 'online' or 'global (plan once)'
        self.planning_time_ahead = rospy.get_param("~planning_time_ahead", 1.0)  # the time ahead of the current time to plan the trajectory
        self.des_pos_z = rospy.get_param("~des_pos_z", 2.0)
        self.longitu_step_dis = rospy.get_param("~longitu_step_dis", 5.0)  # the distance forward in each replanning
        self.lateral_step_length = rospy.get_param("~lateral_step_length", 1.0)  # if local target pos in obstacle, take lateral step
        self.target_reach_threshold = rospy.get_param("~target_reach_threshold", 0.2)
        self.cmd_hz = rospy.get_param("~cmd_hz", 300)


class DroneState():
    def __init__(self):
        self.global_pos = np.zeros(3)
        self.global_vel = np.zeros(3)
        self.local_vel = np.zeros(3)
        self.attitude = np.zeros(4)


class TrajPlanner():
    def __init__(self, node_name="expert_planner"):
        # Node
        rospy.init_node(node_name, anonymous=False)

        # Members
        self.cv_bridge = CvBridge()
        self.des_path = Path()
        self.map = ESDF()
        self.visualizer = Visualizer()
        self.drone_state = DroneState()
        planner_config = PlannerConfig()
        mission_config = MissionConfig()
        self.planner = MinJerkPlanner(planner_config)
        self.state_cmd = PositionTarget()
        self.state_cmd.coordinate_frame = 1

        # Parameters
        self.planning_mode = mission_config.planning_mode
        self.planning_time_ahead = mission_config.planning_time_ahead
        self.des_pos_z = mission_config.des_pos_z
        self.longitu_step_dis = mission_config.longitu_step_dis
        self.lateral_step_length = mission_config.lateral_step_length
        self.target_reach_threshold = mission_config.target_reach_threshold
        self.cmd_hz = mission_config.cmd_hz
        self.move_vel = planner_config.v_max*0.8

        # Flags and counters
        self.mission_executing = False
        self.near_global_target = False
        self.reached_target = False
        self.odom_received = False
        self.has_traj = False
        self.des_state_index = 0
        self.future_index = 99999

        # Server
        self.plan_server = actionlib.SimpleActionServer('plan', PlanAction, self.execute_mission, False)
        self.plan_server.start()

        # Services
        rospy.wait_for_service("/mavros/set_mode")
        self.set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)

        # Subscribers
        self.flight_state_sub = rospy.Subscriber('/mavros/state', State, self.flight_state_cb)
        self.occupancy_map_sub = rospy.Subscriber('/projected_map', OccupancyGrid, self.map.occupancy_map_cb)
        self.odom_sub = rospy.Subscriber('/mavros/local_position/odom', Odometry, self.odom_cb)
        self.depth_img_sub = rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_img_cb, queue_size=1)

        # Publishers
        self.local_pos_cmd_pub = rospy.Publisher("/mavros/setpoint_raw/local", PositionTarget, queue_size=10)
        self.drone_snapshots_pub = rospy.Publisher('/robotMarker', MarkerArray, queue_size=10)
        self.des_wpts_pub = rospy.Publisher('/des_wpts', MarkerArray, queue_size=10)
        self.des_path_pub = rospy.Publisher('/des_path', MarkerArray, queue_size=10)

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
        self.table_filename = 'training_data/train.csv'
        if not os.path.isfile(self.table_filename):
            df = pd.DataFrame(columns=self.table_header)
            df.to_csv(self.table_filename, index=False)

    def flight_state_cb(self, data):
        self.flight_state = data

    def odom_cb(self, data):
        '''
        1. store the drone's global status
        2. publish dynamic tf transform from map frame to camera frame
        (Currently, regard camera frame as drone body frame)
        '''
        self.odom_received = True
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

        if self.mission_executing and np.linalg.norm(global_pos[:2] - self.global_target) < self.target_reach_threshold:
            self.end_mission()

    def end_mission(self):
        rospy.loginfo("Global target reached!")
        self.reached_target = True
        self.mission_executing = False
        self.near_global_target = False
        self.has_traj = False
        self.tracking_cmd_timer.shutdown()
        self.des_state_index = 0

    def depth_img_cb(self, img):
        self.depth_img = self.cv_bridge.imgmsg_to_cv2(img, desired_encoding="passthrough")

    def execute_mission(self, goal):
        self.mission_executing = True
        self.reached_target = False
        target = goal.target
        rospy.loginfo("Global target: x = %f, y = %f", target.pose.position.x, target.pose.position.y)
        self.global_target = np.array([target.pose.position.x, target.pose.position.y])

        if self.planning_mode == 'global':
            self.global_planning()
        else:
            self.online_planning()

        while not self.reached_target:
            time.sleep(0.01)

        result = PlanResult()
        result.success = True
        self.plan_server.set_succeeded(result)

    def global_planning(self):
        while not self.odom_received:
            time.sleep(0.01)

        self.target_state = np.zeros((2, 2))
        self.target_state[0] = self.global_target
        self.first_plan()
        self.start_tracking()
        self.visualize_des_wpts()
        self.visualize_des_path()

    def online_planning(self):
        while self.mission_executing and not self.near_global_target:
            while not self.odom_received:
                time.sleep(0.01)

            self.set_local_target()
            if not self.has_traj:
                self.first_plan()
                self.start_tracking()
            else:  # plan ahead 1s
                self.replan()

            self.visualize_des_wpts()
            self.visualize_des_path()

    def set_local_target(self):
        self.target_state = np.zeros((2, 2))
        current_pos = self.drone_state.global_pos[:2]
        global_target_pos = self.global_target

        # if current pos is close enough to global target, set local target as global target
        if np.linalg.norm(global_target_pos - current_pos) < self.longitu_step_dis:
            self.target_state[0] = global_target_pos
            self.near_global_target = True
            print(" ")
            rospy.loginfo("Last target: x = %f, y = %f", global_target_pos[0], global_target_pos[1])
            return

        longitu_dir = (global_target_pos - current_pos)/np.linalg.norm(global_target_pos - current_pos)
        lateral_dir = np.array([[longitu_dir[1], -longitu_dir[0]],
                                [-longitu_dir[1], longitu_dir[0]]])
        lateral_dir_flag = 0
        lateral_move_dis = self.lateral_step_length

        # get local target pos
        local_target_pos = current_pos + self.longitu_step_dis * longitu_dir
        while self.map.has_collision(local_target_pos):
            local_target_pos += lateral_move_dis * lateral_dir[lateral_dir_flag]
            lateral_dir_flag = 1 - lateral_dir_flag
            lateral_move_dis += self.lateral_step_length

        # get local target vel
        goal_dir = (global_target_pos - local_target_pos)/np.linalg.norm(global_target_pos - local_target_pos)
        local_target_vel = np.array([self.move_vel * goal_dir[0], self.move_vel * goal_dir[1]])

        local_target = np.array([local_target_pos,
                                 local_target_vel])

        self.target_state = local_target

        print(" ")
        rospy.loginfo("Local target: x = %f, y = %f, vel_x = %f, vel_y = %f",
                      local_target[0][0], local_target[0][1], local_target[1][0], local_target[1][1])

    def first_plan(self):
        self.traj_plan(self.drone_state)
        # First planning! Set the des_state_array as des_state
        self.des_state_array = self.des_state
        self.has_traj = True

    def get_drone_state_ahead(self):
        '''
        get the drone state after 1s from self.des_state
        '''
        self.future_index = int(self.planning_time_ahead * self.cmd_hz) + self.des_state_index
        drone_state_ahead = DroneState()
        drone_state_ahead.global_pos = np.append(self.des_state_array[self.future_index, 0, :], self.des_pos_z)
        drone_state_ahead.global_vel = np.append(self.des_state_array[self.future_index, 1, :], 0)

        return drone_state_ahead

    def replan(self):
        drone_state_ahead = self.get_drone_state_ahead()
        self.traj_plan(drone_state_ahead)
        # Concatenate the new trajectory to the old one, at index self.future_index
        self.des_state_array = np.concatenate((self.des_state_array[:self.future_index], self.des_state), axis=0)

    def traj_plan(self, drone_init_state):
        '''
        trajectory planning, store the full state cmd in self.des_state
        '''
        time_start = time.time()
        drone_state_2d = np.array([drone_init_state.global_pos[:2],
                                   drone_init_state.global_vel[:2]])
        self.des_pos_z = drone_init_state.global_pos[2]  # use current height
        self.planner.plan(self.map, drone_state_2d, self.target_state)  # 2D planning, z is fixed
        self.des_state, self.traj_time, self.cmd_hz = self.planner.get_full_state_cmd()
        time_end = time.time()
        planning_time = time_end - time_start
        rospy.loginfo("Planning finished in %f s", planning_time)

    def traj_plan_record(self):
        '''
        trajectory planning + store the training data
        '''
        depth_image = self.depth_img
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

        self.record_train_input(drone_state, depth_image)  # record training data

    def record_train_input(self, drone_state, depth_image):
        '''
        Record training data and write to csv file
        '''
        now = datetime.datetime.now()
        timestamp = int(now.strftime("%Y%m%d%H%M%S"))

        # scale and save depth_iamge
        rospy.loginfo("Range of depth image: %f, %f", np.min(depth_image), np.max(depth_image))
        depth_image = depth_image*255.0/np.max(depth_image)
        cv2.imwrite(f'training_data/depth_img/{timestamp}.png', depth_image)

        # save drone_local_vel, drone_attitude, target_local_pos, target_local_vel
        drone_local_vel = drone_state.local_vel  # size: (3,)
        drone_quat = drone_state.attitude  # size: (4,)
        drone_attitude = np.array([drone_quat.w, drone_quat.x, drone_quat.y, drone_quat.z])  # size: (4,)
        target_state_3d = np.zeros((2, 3))
        target_state_3d[:, 0:2] = self.target_state
        target_state_3d[0, 2] = self.des_pos_z
        target_local_pos = drone_quat.inverse.rotate(target_state_3d[0] - drone_state.global_pos)  # size: (3,)
        target_local_vel = drone_quat.inverse.rotate(target_state_3d[1] - drone_state.global_vel)  # size: (3,)

        train_data = np.concatenate((np.array([timestamp]), drone_local_vel, drone_attitude, target_local_pos, target_local_vel), axis=0)
        df = pd.read_csv(self.table_filename)
        df = pd.concat([df, pd.DataFrame(train_data.reshape(1, -1), columns=self.table_header)], ignore_index=True)
        df.to_csv(self.table_filename, index=False)
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

    def enter_offboard(self):
        '''
        if not in OFFBOARD mode, switch to OFFBOARD mode
        '''
        rospy.loginfo("Tracking started!")
        rospy.loginfo("Current drone position: %f, %f, %f",
                      self.drone_state.global_pos[0], self.drone_state.global_pos[1], self.drone_state.global_pos[2])

        if self.flight_state.mode != "OFFBOARD":
            self.warm_up()
            set_offb_req = SetModeRequest()
            set_offb_req.custom_mode = 'OFFBOARD'
            if (self.set_mode_client.call(set_offb_req).mode_sent == True):
                rospy.loginfo("OFFBOARD enabled")

    def start_tracking(self):
        '''
        When triggered, start to publish full state cmd
        '''
        self.enter_offboard()

        self.tracking_cmd_timer = rospy.Timer(rospy.Duration(1/self.cmd_hz), self.tracking_cmd_timer_cb)

    def tracking_cmd_timer_cb(self, event):
        '''
        Publish state cmd, height is fixed to current height
        '''
        self.state_cmd.position.x = self.des_state_array[self.des_state_index][0][0]
        self.state_cmd.position.y = self.des_state_array[self.des_state_index][0][1]
        self.state_cmd.position.z = self.des_pos_z

        self.state_cmd.velocity.x = self.des_state_array[self.des_state_index][1][0]
        self.state_cmd.velocity.y = self.des_state_array[self.des_state_index][1][1]
        self.state_cmd.velocity.z = 0

        self.state_cmd.acceleration_or_force.x = self.des_state_array[self.des_state_index][2][0]
        self.state_cmd.acceleration_or_force.y = self.des_state_array[self.des_state_index][2][1]
        self.state_cmd.acceleration_or_force.z = 0

        self.state_cmd.yaw = np.arctan2(self.des_state_array[self.des_state_index][0][1] - self.des_state_array[self.des_state_index - 1][0][1],
                                        self.des_state_array[self.des_state_index][0][0] - self.des_state_array[self.des_state_index - 1][0][0])

        self.local_pos_cmd_pub.publish(self.state_cmd)

        if self.des_state_index < self.future_index or self.near_global_target:
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
