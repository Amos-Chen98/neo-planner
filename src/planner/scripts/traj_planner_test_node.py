'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2023-08-15 13:15:43
'''
import os
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_path)
from tf.transformations import euler_from_quaternion
from enhanced_planner import EnhancedPlanner
from record_planner import RecordPlanner
from nn_planner import NNPlanner
from planner.msg import *
import actionlib
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from esdf import ESDF
import time
from pyquaternion import Quaternion
from expert_planner import MinJerkPlanner
from mavros_msgs.srv import SetMode, SetModeRequest
from mavros_msgs.msg import State, PositionTarget
import numpy as np
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from visualizer import Visualizer
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import copy



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
        self.init_wpts_num = int(rospy.get_param("~init_wpts_num", 2))  # the initial number of waypoints
        self.init_T = rospy.get_param("~init_T", 2.5)  # the initial T of each segment
        self.des_pos_z = rospy.get_param("~des_pos_z", 2.0)  # the desired z position of the drone
        self.collision_cost_tol = rospy.get_param("~collision_cost_tol", 10)  # the tolerance of collision cost


class DroneState():
    def __init__(self):
        self.global_pos = np.zeros(3)
        self.global_vel = np.zeros(3)
        self.local_vel = np.zeros(3)
        self.attitude = Quaternion()  # ref: http://kieranwynn.github.io/pyquaternion/
        self.yaw = 0.0


class TrajPlanner():
    def __init__(self, node_name="traj_planner"):
        # Node
        rospy.init_node(node_name, anonymous=False)

        # Members
        self.cv_bridge = CvBridge()
        self.des_path = Path()
        self.map = ESDF()
        self.visualizer = Visualizer()
        self.drone_state = DroneState()
        self.planner_config = PlannerConfig()
        self.state_cmd = PositionTarget()
        self.state_cmd.coordinate_frame = 1
        self.init_marker_arrays()

        # Parameters
        self.replan_mode = rospy.get_param("~replan_mode", 'online')  # 'online' or 'global (plan once)'
        self.planning_time_ahead = rospy.get_param("~planning_time_ahead", 1.0)  # the time ahead of the current time to plan the trajectory
        self.longitu_step_dis = rospy.get_param("~longitu_step_dis", 5.0)  # the distance forward in each replanning
        self.lateral_step_length = rospy.get_param("~lateral_step_length", 1.0)  # if local target pos in obstacle, take lateral step
        self.target_reach_threshold = rospy.get_param("~target_reach_threshold", 0.2)
        self.cmd_hz = rospy.get_param("~cmd_hz", 300)
        self.selected_planner = rospy.get_param("~selected_planner", 'basic')  # 'basic', 'batch', 'expert', 'record', 'nn', or 'enhanced'
        self.replan_period = rospy.get_param("~replan_period", 0.5)  # the interval between replanning， 0 means replan right after the previous plan
        self.move_vel = self.planner_config.v_max*0.8
        self.des_pos_z = self.planner_config.des_pos_z
        self.record_metric = rospy.get_param("~record_metric", False)
        self.yaw_shift_tol = rospy.get_param("~yaw_shift_tol", 0.17453)

        # Planner
        if self.selected_planner in ['basic', 'batch', 'warmstart']:
            self.planner = MinJerkPlanner(self.planner_config)
        elif self.selected_planner == 'record':
            self.planner = RecordPlanner(self.planner_config)
        elif self.selected_planner == 'nn':
            self.planner = NNPlanner(self.des_pos_z)
        elif self.selected_planner == 'enhanced':
            self.planner = EnhancedPlanner(self.planner_config)
        else:
            rospy.logerr("Invalid planner mode!")

        # Flags and counters
        self.target_received = False
        self.reached_target = False
        self.near_global_target = False
        self.odom_received = False
        self.des_state_index = 0
        self.future_index = 99999
        self.des_state_length = 99999  # this is used to check if the des_state_index is valid
        self.metric_eva_interval = 0.5

        # Server
        self.plan_server = actionlib.SimpleActionServer('plan', PlanAction, self.execute_mission, False)
        self.plan_server.start()

        # Services
        rospy.wait_for_service("mavros/set_mode")
        self.set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)

        # Subscribers
        self.flight_state_sub = rospy.Subscriber('mavros/state', State, self.flight_state_cb)
        self.occupancy_map_sub = rospy.Subscriber('projected_map', OccupancyGrid, self.map.occupancy_map_cb)
        self.odom_sub = rospy.Subscriber('mavros/local_position/odom', Odometry, self.odom_cb)
        self.depth_img_sub = rospy.Subscriber('camera/depth/image_raw', Image, self.depth_img_cb, queue_size=1)

        # Publishers
        self.local_pos_cmd_pub = rospy.Publisher("mavros/setpoint_raw/local", PositionTarget, queue_size=10)
        self.drone_snapshots_pub = rospy.Publisher('robotMarker', MarkerArray, queue_size=10)
        self.des_wpts_pub = rospy.Publisher('des_wpts', MarkerArray, queue_size=10)
        self.des_path_pub = rospy.Publisher('des_path', MarkerArray, queue_size=10)
        self.local_target_pub = rospy.Publisher('local_target', Marker, queue_size=10)

        rospy.loginfo(f"Global planner initialized! Selected planner: {self.selected_planner}")

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

        # get yaw from quaternion
        euler = euler_from_quaternion([data.pose.pose.orientation.x,
                                       data.pose.pose.orientation.y,
                                       data.pose.pose.orientation.z,
                                       data.pose.pose.orientation.w])

        self.drone_state.yaw = euler[2]

        if self.target_received and np.linalg.norm(global_pos[:2] - self.global_target) < self.target_reach_threshold:
            rospy.loginfo("Global target reached!\n")
            self.end_mission()

    def init_mission(self):
        self.target_received = True
        self.reached_target = False
        self.near_global_target = False
        self.des_state_index = 0
        if self.record_metric:
            self.init_metrics()

    def init_metrics(self):
        self.planner.iter_num = 0
        self.planner.opt_running_times = 0
        self.total_planning_duration = 0.0
        self.total_planning_times = 0
        # self.weighted_cost = 0.0
        self.drone_state_list = []
        self.metric_weights = np.array([1, 1, 1])  # distance, feasibility, collision
        self.metric_timer = rospy.Timer(rospy.Duration(self.metric_eva_interval), self.record_metric_cb)

    def record_metric_cb(self, event):
        self.drone_state_list.append(copy.copy(self.drone_state))  # if not using copy, the drone_state_list will be all the same

    def end_mission(self):
        # self.tracking_cmd_timer.shutdown()
        self.target_received = False
        self.reached_target = True
        self.near_global_target = False
        self.des_state_index = 0
        if self.replan_mode == 'periodic':
            self.replan_timer.shutdown()
        if self.record_metric:
            self.metric_timer.shutdown()

    def depth_img_cb(self, img):
        self.depth_img = self.cv_bridge.imgmsg_to_cv2(img, desired_encoding="passthrough")

    def execute_mission(self, goal):
        target = goal.target
        rospy.loginfo("Target received: x = %f, y = %f", target.pose.position.x, target.pose.position.y)
        self.global_target = np.array([target.pose.position.x, target.pose.position.y])
        self.init_mission()

        if self.replan_mode == 'global':
            rospy.loginfo("Mission mode: global")
            self.global_planning()
        elif self.replan_mode == 'online':
            rospy.loginfo("Mission mode: online")
            self.online_planning()
        elif self.replan_mode == 'periodic':
            rospy.loginfo("Mission mode: periodic")
            self.periodic_planning()
        else:
            rospy.logerr("Invalid mission mode!")

        self.report_planning_result()

    def report_planning_result(self):
        while not self.plan_server.is_preempt_requested() and not self.reached_target:
            time.sleep(0.01)

        if self.record_metric:
            self.report_metrics()

        if self.plan_server.is_preempt_requested():
            rospy.loginfo("Planning preempted!\n")
            self.end_mission()
            self.plan_server.set_preempted()
        else:  # this means the target is reached
            result = PlanResult()
            result.success = True
            self.plan_server.set_succeeded(result)

    def report_metrics(self):
        if self.selected_planner != 'nn':
            average_iter_num = self.planner.iter_num / self.planner.opt_running_times
            rospy.loginfo("Average iter num: %d", average_iter_num)

        if self.replan_mode != 'global':
            average_planning_duration = self.total_planning_duration / self.total_planning_times
            rospy.loginfo("Average planning duration: %f", average_planning_duration)

        rospy.loginfo("Total planning times: %d", self.total_planning_times)

        weighted_metric = self.get_weighted_metric(self.map, self.drone_state_list)
        rospy.loginfo("Weighted metric: %s\n", weighted_metric)

    def get_weighted_metric(self, map, drone_state_list):
        raw_cost = np.zeros(3)   # planning_time, distance, feasibility, collision

        for i in range(len(drone_state_list)):
            pos = drone_state_list[i].global_pos[:2]
            vel = drone_state_list[i].global_vel[:2]

            # distance
            if i > 0:
                pre_pos = drone_state_list[i-1].global_pos[:2]
                raw_cost[0] += np.linalg.norm(pos - pre_pos)

            # feasibility
            violate_vel = sum(vel**2) - self.planner_config.v_max**2
            if violate_vel > 0:
                raw_cost[1] += violate_vel**3

            # collision
            edt_dis = map.get_edt_dis(pos)
            violate_dis = self.planner_config.safe_dis - edt_dis

            if violate_dis > 0.0:
                raw_cost[2] += violate_dis**3

        weighted_metric = np.dot(raw_cost, self.metric_weights)

        return weighted_metric

    def global_planning(self):
        while not self.odom_received:
            time.sleep(0.01)

        self.target_state = np.zeros((2, 2))
        self.target_state[0] = self.global_target
        self.first_plan()
        # self.start_tracking()
        self.visualize_des_wpts()
        self.visualize_des_path()

    def try_local_planning(self):
        seed = 0
        self.set_local_target(seed)
        self.visualize_local_target()
        while True:
            try:
                self.replan()
                break
            except Exception as ex:
                rospy.logwarn("Local planning failed: %s", ex)
                seed += 1
                self.set_local_target(seed)
                if seed > 10:
                    rospy.logerr("Entire planning failed!\n")
                    self.end_mission()
                    self.plan_server.set_aborted()
                    return

        self.visualize_des_wpts()
        self.visualize_des_path()

    def online_planning(self):
        while not self.odom_received:
            time.sleep(0.01)

        self.set_local_target()
        self.first_plan()
        # self.start_tracking()
        self.visualize_des_wpts()
        self.visualize_des_path()

        while (
            not self.reached_target
            and not self.near_global_target
            and not self.plan_server.is_preempt_requested()
        ):
            self.try_local_planning()

    def periodic_planning(self):
        while not self.odom_received:
            time.sleep(0.01)

        self.set_local_target()
        self.first_plan()
        # self.start_tracking()
        self.visualize_des_wpts()
        self.visualize_des_path()

        # after the first plan, replan periodically
        self.replan_timer = rospy.Timer(rospy.Duration(self.replan_period), self.replan_cb)

    def replan_cb(self, event):
        if (
            not self.reached_target
            and not self.near_global_target
            and not self.plan_server.is_preempt_requested()
        ):
            self.try_local_planning()

    def set_local_target(self, seed=0):
        self.target_state = np.zeros((2, 2))
        current_pos = self.drone_state.global_pos[:2]
        global_target_pos = self.global_target

        # if current pos is close enough to global target, set local target as global target
        if np.linalg.norm(global_target_pos - current_pos) < self.longitu_step_dis:
            self.target_state[0] = global_target_pos
            self.near_global_target = True
            return

        longitu_dir = (global_target_pos - current_pos)/np.linalg.norm(global_target_pos - current_pos)
        lateral_dir = np.array([[longitu_dir[1], -longitu_dir[0]],
                                [-longitu_dir[1], longitu_dir[0]]])
        lateral_dir_flag = 0
        lateral_move_dis = self.lateral_step_length

        # get local target pos
        if seed > 1e-3:
            local_target_pos = current_pos + self.longitu_step_dis * longitu_dir + np.random.normal(0, 1, 2)  # 0 for mean, 1 for std
        else:
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

    def first_plan(self):
        drone_state = self.drone_state
        time_start = time.time()
        if self.selected_planner == 'basic' or self.selected_planner == 'warmstart':
            self.basic_traj_plan(self.map, self.drone_state, self.target_state)
        elif self.selected_planner == 'batch':
            self.batch_traj_plan(self.map, self.drone_state, self.target_state)
        elif self.selected_planner == 'record':
            self.record_traj_plan(self.map, self.depth_img, self.drone_state, self.drone_state, self.target_state)
        elif self.selected_planner == 'nn':
            self.nn_traj_plan(self.depth_img, self.drone_state, self.drone_state, self.target_state)
        elif self.selected_planner == 'enhanced':
            self.enhanced_traj_plan(self.map, self.depth_img, self.drone_state, self.drone_state, self.target_state)
        else:
            rospy.logerr("Invalid planner mode!")

        time_end = time.time()
        rospy.loginfo("Planning time: {}".format(time_end - time_start))

        # collect metrics
        if self.record_metric:
            self.total_planning_duration += time_end - time_start
            self.total_planning_times += 1
        # self.weighted_cost += self.planner.final_cost

        # calculate the int_wpts regarding drone_state_ahead, for warmstart planning only
        self.int_wpts_local = self.get_int_wpts_local(drone_state, self.planner.int_wpts)

        # First planning! Retrieve planned trajectory
        self.des_state = self.planner.get_full_state_cmd(self.cmd_hz)

        # Set the des_state_array as des_state
        self.des_state_array = self.des_state
        self.des_state_length = self.des_state_array.shape[0]

    def get_drone_state_ahead(self):
        '''
        get the drone state after 1s from self.des_state
        '''
        self.future_index = min(int(self.planning_time_ahead * self.cmd_hz) + self.des_state_index,
                                self.des_state_length - 1)
        drone_state_ahead = DroneState()
        drone_state_ahead.global_pos = np.append(self.des_state_array[self.future_index, 0, :], self.des_pos_z)
        drone_state_ahead.global_vel = np.append(self.des_state_array[self.future_index, 1, :], 0)

        return drone_state_ahead

    def replan(self):
        drone_state_ahead = self.get_drone_state_ahead()

        time_start = time.time()

        if self.selected_planner == 'basic':
            self.basic_traj_plan(self.map, drone_state_ahead, self.target_state)
        elif self.selected_planner == 'batch':
            self.batch_traj_plan(self.map, drone_state_ahead, self.target_state)
        elif self.selected_planner == 'record':
            self.record_traj_plan(self.map, self.depth_img, self.drone_state, drone_state_ahead, self.target_state)
        elif self.selected_planner == 'nn':
            self.nn_traj_plan(self.depth_img, self.drone_state, drone_state_ahead, self.target_state)
        elif self.selected_planner == 'enhanced':
            self.enhanced_traj_plan(self.map, self.depth_img, self.drone_state, drone_state_ahead, self.target_state)
        elif self.selected_planner == 'warmstart':
            self.warmstart_traj_plan(self.map, drone_state_ahead, self.target_state, self.int_wpts_local, self.planner.ts)
        else:
            rospy.logerr("Invalid planner mode!")

        time_end = time.time()
        rospy.loginfo("Planning time: {}".format(time_end - time_start))

        # collect metrics
        if self.record_metric:
            self.total_planning_duration += time_end - time_start
            self.total_planning_times += 1
            # self.weighted_cost += self.planner.final_cost

        # calculate the int_wpts regarding drone_state_ahead, for warmstart planning only
        self.int_wpts_local = self.get_int_wpts_local(drone_state_ahead, self.planner.int_wpts)

        # retrieve planned trajectory
        self.des_state = self.planner.get_full_state_cmd(self.cmd_hz)

        # Concatenate the new trajectory to the old one, at index self.future_index
        self.des_state_array = np.concatenate((self.des_state_array[:self.future_index], self.des_state), axis=0)
        self.des_state_length = self.des_state_array.shape[0]

    def get_int_wpts_local(self, drone_state, int_wpts):
        int_wpts = int_wpts.T
        ref_pos = np.array([drone_state.global_pos[0], drone_state.global_pos[1]])
        int_wpts_local = np.zeros((int_wpts.shape[0], 2))
        for i in range(int_wpts.shape[0]):
            int_wpts_local[i] = int_wpts[i] - ref_pos

        return int_wpts_local  # row major

    def basic_traj_plan(self, map, plan_init_state, target_state):
        '''
        trajectory planning, store the full state cmd in self.des_state
        '''
        drone_state_2d = np.array([plan_init_state.global_pos[:2],
                                   plan_init_state.global_vel[:2]])
        self.planner.plan(map, drone_state_2d, target_state)  # 2D planning, z is fixed

    def warmstart_traj_plan(self, map, plan_init_state, target_state, int_wpts_local, ts):
        # get int_wpts refered to plan_init_state
        int_wpts = np.zeros((int_wpts_local.shape[0], 2))
        for i in range(int_wpts_local.shape[0]):
            int_wpts[i] = int_wpts_local[i] + plan_init_state.global_pos[:2]

        int_wpts = int_wpts.T  # col major

        drone_state_2d = np.array([plan_init_state.global_pos[:2],
                                   plan_init_state.global_vel[:2]])

        rospy.loginfo("Init int_wpts: {}".format(int_wpts))
        rospy.loginfo("Init ts: {}".format(ts))

        self.planner.warm_start_plan(map, drone_state_2d, target_state, int_wpts, ts)

        rospy.loginfo("Result int_wpts: {}".format(self.planner.int_wpts))
        rospy.loginfo("Result ts: {}".format(self.planner.ts))

    def batch_traj_plan(self, map, plan_init_state, target_state):
        '''
        trajectory planning, store the full state cmd in self.des_state
        '''
        drone_state_2d = np.array([plan_init_state.global_pos[:2],
                                   plan_init_state.global_vel[:2]])
        self.planner.batch_plan(map, drone_state_2d, target_state)  # 2D planning, z is fixed

    def record_traj_plan(self, map, depth_img, drone_state, plan_init_state, target_state):
        self.planner.record_traj_plan(map, depth_img, drone_state, plan_init_state, target_state)

    def nn_traj_plan(self, depth_img, drone_state, plan_init_state, target_state):
        self.planner.nn_traj_plan(depth_img, drone_state, plan_init_state, target_state)

    def enhanced_traj_plan(self, map, depth_img, drone_state, plan_init_state, target_state):
        self.planner.enhanced_traj_plan(map, depth_img, drone_state, plan_init_state, target_state)

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

        # self.state_cmd.yaw = 0.0

        # self.state_cmd.yaw = np.arctan2(self.des_state_array[self.des_state_index][0][1] - self.des_state_array[self.des_state_index - 1][0][1],
        #                                 self.des_state_array[self.des_state_index][0][0] - self.des_state_array[self.des_state_index - 1][0][0])

        des_yaw = np.arctan2(self.des_state_array[self.des_state_index][0][1] - self.des_state_array[self.des_state_index - 1][0][1],
                             self.des_state_array[self.des_state_index][0][0] - self.des_state_array[self.des_state_index - 1][0][0])

        yaw_shift = des_yaw - self.drone_state.yaw
        if abs(yaw_shift) < self.yaw_shift_tol:
            self.state_cmd.yaw = des_yaw
        else:
            self.state_cmd.yaw = self.drone_state.yaw + np.sign(yaw_shift)*self.yaw_shift_tol

        # print("current yaw: {}".format(self.drone_state.yaw*180/np.pi))
        # print("Des yaw: {}".format(des_yaw*180/np.pi))
        # print("State cmd yaw: {}".format(self.state_cmd.yaw*180/np.pi))

        self.state_cmd.header.stamp = rospy.Time.now()

        self.local_pos_cmd_pub.publish(self.state_cmd)

        if self.des_state_index < self.des_state_length - 1:
            self.des_state_index += 1

    def init_marker_arrays(self):
        # local target
        self.local_target_marker = Marker()
        self.local_target_marker.header.frame_id = "map"
        self.local_target_marker.type = Marker.SPHERE
        self.local_target_marker.scale.x = 0.4
        self.local_target_marker.scale.y = 0.4
        self.local_target_marker.scale.z = 0.4
        self.local_target_marker.color.a = 1
        self.local_target_marker.color.r = 1
        self.local_target_marker.color.g = 1
        self.local_target_marker.color.b = 0
        self.local_target_marker.pose.orientation.w = 1.0

        # des wpts
        self.wpts_markerarray = MarkerArray()
        max_wpts_length = 20
        for i in range(max_wpts_length):
            marker = Marker()
            marker.id = i
            marker.header.frame_id = "map"
            marker.type = Marker.SPHERE
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.4
            marker.scale.y = 0.4
            marker.scale.z = 0.4

            self.wpts_markerarray.markers.append(marker)

        # des path
        self.path_markerarray = MarkerArray()
        max_path_length = 1000
        for i in range(max_path_length):
            marker = Marker()
            marker.id = i
            marker.header.frame_id = "map"
            marker.type = Marker.LINE_STRIP
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.1

            self.path_markerarray.markers.append(marker)

    def visualize_local_target(self):
        self.local_target_marker.header.stamp = rospy.Time.now()
        self.local_target_marker.pose.position.x = self.target_state[0][0]
        self.local_target_marker.pose.position.y = self.target_state[0][1]
        self.local_target_marker.pose.position.z = self.des_pos_z

        self.local_target_pub.publish(self.local_target_marker)

    def visualize_des_wpts(self):
        '''
        Visualize the desired waypoints as markers
        '''
        pos_array = self.planner.int_wpts  # shape: (2,n)
        pos_array = np.vstack((pos_array, self.des_pos_z * np.ones([1, pos_array.shape[1]]))).T
        self.wpts_markerarray = self.visualizer.modify_wpts_markerarray(
            self.wpts_markerarray, pos_array)  # the id of self.wpts_markerarray will be the same
        self.des_wpts_pub.publish(self.wpts_markerarray)

    def visualize_des_path(self):
        '''
        Visualize the desired path, where high-speed pieces and low-speed pieces are colored differently
        '''
        pos_array = self.planner.get_pos_array()
        pos_array = np.hstack((pos_array, self.des_pos_z * np.ones([len(pos_array), 1])))
        vel_array = np.linalg.norm(self.planner.get_vel_array(), axis=1)  # shape: (n,)
        self.path_markerarray = self.visualizer.modify_path_markerarray(self.path_markerarray, pos_array, vel_array)
        self.des_path_pub.publish(self.path_markerarray)

    def visualize_drone_snapshots(self):
        '''
        publish snapshots of drone model (mesh) along the trajectory
        this is not used
        '''
        pos_array = self.planner.get_pos_array()
        pos_array = np.hstack((pos_array, self.des_pos_z * np.ones([len(pos_array), 1])))
        drone_snapshots = self.visualizer.get_marker_array(pos_array, 10, 2)
        self.drone_snapshots_pub.publish(drone_snapshots)
        rospy.loginfo("Drone_snapshots published!")


if __name__ == "__main__":

    traj_planner = TrajPlanner()

    rospy.spin()
