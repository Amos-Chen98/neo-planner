'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2023-05-23 16:08:21
'''
import os
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_path)
from pyquaternion import Quaternion
from planner.msg import *
import actionlib
from visualization_msgs.msg import Marker
from mavros_msgs.srv import SetMode, SetModeRequest
from mavros_msgs.srv import SetMode
from nav_msgs.msg import Odometry
from mavros_msgs.msg import State, PositionTarget
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest, CommandTOL, CommandTOLRequest
import rospy
import numpy as np
from transitions import Machine
from transitions.extensions import GraphMachine
from geometry_msgs.msg import PoseStamped
from esdf import ESDF
import pandas as pd
import datetime


class DroneState():
    def __init__(self):
        self.global_pos = np.zeros(3)
        self.global_vel = np.zeros(3)
        self.local_vel = np.zeros(3)
        self.attitude = Quaternion()


class DesState():
    def __init__(self):
        self.global_pos = np.zeros(3)
        self.global_vel = np.zeros(3)
        self.global_acc = np.zeros(3)
        self.timestamp = 0


class FullDataMat():
    def __init__(self):
        self.timestamp_list = []
        self.drone_pos_list = []
        self.drone_vel_list = []
        self.des_pos_list = []
        self.des_vel_list = []
        self.des_acc_list = []


class Manager():
    def __init__(self, node_name="manager"):
        # Node
        rospy.init_node(node_name, anonymous=False)

        # Members
        self.flight_state = State()
        self.map = ESDF()
        self.offb_req = SetModeRequest()
        self.arm_req = CommandBoolRequest()
        self.pos_cmd = PositionTarget()
        self.drone_state = DroneState()
        self.des_state = DesState()  # class for recording desired state
        self.full_data_mat = FullDataMat()  # class for recording data

        # Parameters
        self.offb_req.custom_mode = 'OFFBOARD'
        self.arm_req.value = True
        self.hover_height = 2.0
        self.pos_cmd.coordinate_frame = 1
        self.pos_cmd.position.z = self.hover_height
        self.global_target = None

        # Flags and counters
        self.odom_received = False
        self.has_goal = False
        self.recording_data = True

        # Client / Service init
        self.arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)
        self.set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)
        self.takeoff_client = rospy.ServiceProxy("/mavros/cmd/takeoff", CommandTOL)

        # Action client
        self.plan_client = actionlib.SimpleActionClient('plan', PlanAction)

        try:
            rospy.wait_for_service('/mavros/cmd/arming')
            rospy.wait_for_service('/mavros/set_mode')
            rospy.wait_for_service("/mavros/cmd/takeoff")
            self.plan_client.wait_for_server()
        except rospy.ROSException:
            exit('Wait for service timeout')

        # Subscribers
        self.flight_state_sub = rospy.Subscriber('/mavros/state', State, self.flight_state_cb)
        self.odom_sub = rospy.Subscriber('/mavros/local_position/odom', Odometry, self.odom_cb)
        self.target_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.trigger_plan)
        self.local_pos_cmd_sub = rospy.Subscriber("/mavros/setpoint_raw/local", PositionTarget, self.pos_cmd_cb)

        # Publishers
        self.local_pos_cmd_pub = rospy.Publisher("/mavros/setpoint_raw/local", PositionTarget, queue_size=10)
        self.target_vis_pub = rospy.Publisher('/global_target', Marker, queue_size=10)

        # FSM
        self.fsm = GraphMachine(model=self, states=['INIT', 'TAKINGOFF', 'HOVER', 'MISSION'], initial='INIT')
        self.fsm.add_transition(trigger='launch', source='INIT', dest='TAKINGOFF', before="get_odom", after=['takeoff', 'print_current_state'])
        self.fsm.add_transition(trigger='reach_height', source='TAKINGOFF', dest='HOVER', after=['print_current_state'])
        self.fsm.add_transition(trigger='set_goal', source='HOVER', dest='MISSION', after=['print_current_state'])
        self.fsm.add_transition(trigger='set_goal', source='MISSION', dest='MISSION', after=['print_current_state'])
        self.fsm.add_transition(trigger='reach_goal', source='MISSION', dest='HOVER', after=['print_current_state'])

        # Initialize the csv file collecting training data
        self.table_header = ['time',
                             'drone_pos_x',
                             'drone_pos_y',
                             'drone_pos_z',
                             'drone_vel_x',
                             'drone_vel_y',
                             'drone_vel_z',
                             'des_pos_x',
                             'des_pos_y',
                             'des_pos_z',
                             'des_vel_x',
                             'des_vel_y',
                             'des_vel_z',
                             'des_acc_x',
                             'des_acc_y',
                             'des_acc_z'
                             ]

    def print_current_state(self):
        rospy.loginfo("Current state: %s", self.state)

    def trigger_plan(self, target):
        rospy.loginfo("Global target: x = %f, y = %f", target.pose.position.x, target.pose.position.y)
        self.global_target = np.array([target.pose.position.x,
                                       target.pose.position.y,
                                       target.pose.position.z])
        self.vis_target()
        self.set_goal()
        goal_msg = PlanGoal()
        goal_msg.target = target
        if self.has_goal == True:
            self.plan_client.cancel_goal()
        else:
            self.has_goal = True

        self.plan_client.send_goal(goal_msg, done_cb=self.finish_planning_cb)

        if self.recording_data:
            self.start_recording()

    def finish_planning_cb(self, state, result):
        rospy.loginfo("Reached goal!")
        self.reach_goal()
        if self.recording_data:
            self.end_recording()

    def vis_target(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.pose.position.x = self.global_target[0]
        marker.pose.position.y = self.global_target[1]
        marker.pose.position.z = self.hover_height
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.4
        marker.scale.y = 0.4
        marker.scale.z = 0.4
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        self.target_vis_pub.publish(marker)

    def hover_cmd(self, event):
        if not self.flight_state.armed:
            self.arming_client.call(self.arm_req)

        if self.flight_state.mode != "OFFBOARD":
            self.set_mode_client.call(self.offb_req)

        self.pos_cmd.position.x = self.drone_state.global_pos[0]
        self.pos_cmd.position.y = self.drone_state.global_pos[1]
        self.pos_cmd.position.z = self.drone_state.global_pos[2]

        self.global_target_pub.publish(self.pos_cmd)

    def flight_state_cb(self, data):
        self.flight_state = data

    def get_odom(self):
        while not self.odom_received or not self.flight_state.connected:
            rospy.sleep(0.01)

    def odom_cb(self, data):
        '''
        1. store the drone's global status
        2. publish dynamic tf transform from map frame to camera frame
        (Currently, regard camera frame as drone body frame)
        '''
        self.odom_received = True
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

    def pos_cmd_cb(self, data):
        '''
        This is for data recording
        '''
        self.des_state.timestamp = data.header.stamp
        self.des_state.global_pos = np.array([data.position.x, data.position.y, data.position.z])
        self.des_state.global_vel = np.array([data.velocity.x, data.velocity.y, data.velocity.z])
        self.des_state.global_acc = np.array([data.acceleration_or_force.x, data.acceleration_or_force.y, data.acceleration_or_force.z])

    def start_recording(self):
        self.init_recording()
        self.recording_timer = rospy.Timer(rospy.Duration(0.1), self.recording_timer_cb)

    def init_recording(self):
        self.full_data_mat.timestamp_list = []
        self.full_data_mat.drone_pos_list = []
        self.full_data_mat.drone_vel_list = []
        self.full_data_mat.des_pos_list = []
        self.full_data_mat.des_vel_list = []
        self.full_data_mat.des_acc_list = []

    def recording_timer_cb(self, event):
        time = rospy.Time.now()

        self.full_data_mat.timestamp_list.append(time)
        self.full_data_mat.drone_pos_list.append(self.drone_state.global_pos)
        self.full_data_mat.drone_vel_list.append(self.drone_state.global_vel)
        self.full_data_mat.des_pos_list.append(self.des_state.global_pos)
        self.full_data_mat.des_vel_list.append(self.des_state.global_vel)
        self.full_data_mat.des_acc_list.append(self.des_state.global_acc)

    def end_recording(self):
        self.recording_timer.shutdown()
        self.save_recording()

    def save_recording(self):
        length = len(self.full_data_mat.timestamp_list)

        # remove time offset in self.full_data_mat.timestamp_list
        mission_start_time = int(self.full_data_mat.timestamp_list[0].to_sec())
        for i in range(length):
            self.full_data_mat.timestamp_list[i] = self.full_data_mat.timestamp_list[i].to_sec() - mission_start_time
        self.full_data_mat.timestamp_list[0] = 0

        result = np.concatenate((np.array(self.full_data_mat.timestamp_list).reshape(length, 1),
                                 np.array(self.full_data_mat.drone_pos_list).reshape(length, 3),
                                 np.array(self.full_data_mat.drone_vel_list).reshape(length, 3),
                                 np.array(self.full_data_mat.des_pos_list).reshape(length, 3),
                                 np.array(self.full_data_mat.des_vel_list).reshape(length, 3),
                                 np.array(self.full_data_mat.des_acc_list).reshape(length, 3)), axis=1)

        # create a blank csv file
        now = datetime.datetime.now()
        current_time = int(now.strftime("%Y%m%d%H%M%S"))
        table_filename = f'planning_result/{current_time}.csv'
        df = pd.DataFrame(result, columns=self.table_header)
        df.to_csv(table_filename, index=False)
        rospy.loginfo("Planning result (ID: %d) saved!", current_time)

    def takeoff(self):
        self.takeoff_cmd_timer = rospy.Timer(rospy.Duration(0.1), self.takeoff_cmd_cb)

        if not self.flight_state.armed and self.arming_client.call(self.arm_req).success == True:
            rospy.loginfo("Vehicle armed")

        if self.flight_state.mode != "OFFBOARD" and self.set_mode_client.call(self.offb_req).mode_sent == True:
            rospy.loginfo("OFFBOARD enabled")

    def takeoff_by_service(self):
        if not self.flight_state.armed and self.arming_client.call(self.arm_req).success == True:
            rospy.loginfo("Vehicle armed")

        takeoff_cmd = CommandTOLRequest()
        takeoff_cmd.altitude = 3.0

        try:
            res = self.takeoff_client.call(takeoff_cmd)
            if not res.success:
                rospy.logerr('Failed to take off')
        except rospy.ServiceException as e:
            rospy.logerr(e)

    def takeoff_cmd_cb(self, event):
        if not self.flight_state.armed:
            self.arming_client.call(self.arm_req)

        if self.flight_state.mode != "OFFBOARD":
            self.set_mode_client.call(self.offb_req)

        self.pos_cmd.position.x = self.drone_state.global_pos[0]
        self.pos_cmd.position.y = self.drone_state.global_pos[1]
        self.local_pos_cmd_pub.publish(self.pos_cmd)

        if self.drone_state.global_pos[2] >= self.hover_height - 0.05:
            self.takeoff_cmd_timer.shutdown()
            self.reach_height()

    def draw_fsm_graph(self):
        self.get_graph().draw('fsm.pdf', prog='dot')


if __name__ == "__main__":

    manager = Manager()

    manager.launch()

    manager.draw_fsm_graph()

    rospy.spin()
