'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2023-08-11 13:55:05
'''
import os
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_path)
import datetime
import rosbag
from esdf import ESDF
from geometry_msgs.msg import PoseStamped
from transitions.extensions import GraphMachine
from transitions import Machine
import numpy as np
import rospy
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest, CommandTOL, CommandTOLRequest
from mavros_msgs.msg import State, PositionTarget
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
import actionlib
from planner.msg import *
from pyquaternion import Quaternion


class DroneState():
    def __init__(self):
        self.global_pos = np.zeros(3)
        self.global_vel = np.zeros(3)
        self.local_vel = np.zeros(3)
        self.attitude = Quaternion()


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

        # customized parameters
        self.recording_data = rospy.get_param("~recording_data", False)
        self.mission_mode = rospy.get_param("~mission_mode", 'manual')
        self.predefined_goal = rospy.get_param("~predefined_goal", [[0.0, 0.0]])
        self.hover_height = rospy.get_param("~hover_height", 2.0)

        # Parameters
        self.offb_req.custom_mode = 'OFFBOARD'
        self.arm_req.value = True
        self.pos_cmd.coordinate_frame = 1
        self.pos_cmd.position.z = self.hover_height
        self.global_target = None
        self.takeoff_pos_x = 0.0
        self.takeoff_pos_y = 0.0

        # Flags and counters
        self.odom_received = False
        self.has_goal = False
        self.rosbag_is_on = False
        self.goal_index = 0
        self.max_goal_index = len(self.predefined_goal) - 1

        # Client / Service init
        self.arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)
        self.set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)
        self.takeoff_client = rospy.ServiceProxy("mavros/cmd/takeoff", CommandTOL)

        # Action client
        self.plan_client = actionlib.SimpleActionClient('plan', PlanAction)

        try:
            rospy.wait_for_service('mavros/cmd/arming')
            rospy.wait_for_service('mavros/set_mode')
            rospy.wait_for_service("mavros/cmd/takeoff")
            self.plan_client.wait_for_server()
        except rospy.ROSException:
            exit('Wait for service timeout')

        # Subscribers
        self.flight_state_sub = rospy.Subscriber('mavros/state', State, self.flight_state_cb)
        self.odom_sub = rospy.Subscriber('mavros/local_position/odom', Odometry, self.odom_cb)
        self.target_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.trigger_plan)

        # Publishers
        self.local_pos_cmd_pub = rospy.Publisher("mavros/setpoint_raw/local", PositionTarget, queue_size=10)
        self.target_vis_pub = rospy.Publisher('global_target', Marker, queue_size=10)
        self.next_goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)

        # FSM
        self.fsm = GraphMachine(model=self, states=['INIT', 'HOVER', 'MISSION'], initial='INIT')
        self.fsm.add_transition(trigger='launch', source='INIT', dest='HOVER', before="get_odom", after=['print_current_state'])
        self.fsm.add_transition(trigger='set_goal', source='HOVER', dest='MISSION', after=['print_current_state', 'open_rosbag'])
        self.fsm.add_transition(trigger='set_goal', source='MISSION', dest='MISSION', after=['print_current_state', 'open_rosbag'])
        self.fsm.add_transition(trigger='reach_goal', source='MISSION', dest='HOVER', after=['print_current_state', 'close_rosbag'])

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
        print("")
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

    def open_rosbag(self):
        if self.recording_data:
            now = datetime.datetime.now()
            timestamp = now.strftime("%m%d%H%M%S")
            self.bag = rosbag.Bag(f'{current_path[:-8]}/rosbag/{timestamp}.bag', 'w')
            self.rosbag_is_on = True
            rospy.loginfo("rosbag opened!")

    def close_rosbag(self):
        if self.recording_data:
            self.rosbag_is_on = False
            self.bag.close()
            rospy.loginfo("rosbag closed!")

    def finish_planning_cb(self, state, result):
        rospy.loginfo("Reached goal!")
        self.reach_goal()

        if self.mission_mode == "random":
            self.set_random_goal()
        elif self.mission_mode == "predefined" and self.goal_index <= self.max_goal_index:
            self.set_predefined_goal()
            self.goal_index += 1

        # if mission_mode is 'maunal', do nothing and wait for the next target

    def set_predefined_goal(self):
        '''
        set a predefined goal
        '''
        target = PoseStamped()
        target.header.frame_id = "map"
        target.pose.position.x = self.predefined_goal[self.goal_index][0]
        target.pose.position.y = self.predefined_goal[self.goal_index][1]
        target.pose.position.z = self.hover_height
        self.next_goal_pub.publish(target)

    def set_random_goal(self):
        '''
        randomly generate a goal
        '''
        x_bounds = [-2, 28]
        y_bounds = [-8, 8]
        # randomly generate a goal
        x = np.random.uniform(x_bounds[0], x_bounds[1])
        y = np.random.uniform(y_bounds[0], y_bounds[1])

        # if target is in obstale-rich aera, regenerate
        while x > 0 and x < 26 and y > -6 and y < 6:
            x = np.random.uniform(x_bounds[0], x_bounds[1])
            y = np.random.uniform(y_bounds[0], y_bounds[1])

        target = PoseStamped()
        target.header.frame_id = "map"
        target.pose.position.x = x
        target.pose.position.y = y
        target.pose.position.z = self.hover_height

        self.next_goal_pub.publish(target)

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

        if self.recording_data and self.rosbag_is_on:
            self.bag.write('mavros/local_position/odom', data)

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

    def takeoff(self):
        self.takeoff_pos_x = self.drone_state.global_pos[0]
        self.takeoff_pos_y = self.drone_state.global_pos[1]

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

        self.pos_cmd.position.x = self.takeoff_pos_x
        self.pos_cmd.position.y = self.takeoff_pos_y
        self.local_pos_cmd_pub.publish(self.pos_cmd)

        if self.drone_state.global_pos[2] >= self.hover_height - 0.05:
            self.takeoff_cmd_timer.shutdown()
            self.reach_height()

    def draw_fsm_graph(self):
        self.get_graph().draw(f'{current_path[:-8]}/fsm.pdf', prog='dot')


if __name__ == "__main__":

    manager = Manager()

    manager.launch()

    rospy.spin()
