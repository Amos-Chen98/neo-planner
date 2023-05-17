'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2023-05-17 14:45:20
'''
import os
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_path)
from esdf import ESDF
from geometry_msgs.msg import PoseStamped
from transitions.extensions import GraphMachine
from transitions import Machine
import numpy as np
import rospy
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest, CommandTOL, CommandTOLRequest
from mavros_msgs.msg import State, PositionTarget
from nav_msgs.msg import Odometry
from mavros_msgs.srv import SetMode
from mavros_msgs.srv import SetMode, SetModeRequest
from visualization_msgs.msg import Marker
import actionlib
from planner.msg import *


class Manager():
    def __init__(self, node_name="manager"):
        # Node
        rospy.init_node(node_name, anonymous=False)
        self.flight_state = State()
        self.map = ESDF()
        self.odom_received = False
        self.hover_height = 2.0
        self.offb_req = SetModeRequest()
        self.offb_req.custom_mode = 'OFFBOARD'
        self.arm_req = CommandBoolRequest()
        self.arm_req.value = True
        self.pos_cmd = PositionTarget()
        self.pos_cmd.coordinate_frame = 1
        self.pos_cmd.position.z = self.hover_height
        self.drone_state = np.zeros((3, 3))  # p,v,a in map frame
        self.global_target = None
        self.has_goal = False

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

    def finish_planning_cb(self, state, result):
        rospy.loginfo("Reached goal!")
        self.reach_goal()

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

        self.pos_cmd.position.x = self.drone_state[0, 0]
        self.pos_cmd.position.y = self.drone_state[0, 1]
        self.pos_cmd.position.z = self.drone_state[0, 2]

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
                              data.twist.twist.linear.z,
                              ])
        global_vel = local_vel

        self.drone_state[0] = global_pos
        self.drone_state[1] = global_vel

    def takeoff(self):
        self.takeoff_cmd_timer = rospy.Timer(rospy.Duration(0.1), self.takeoff_cmd)

        if not self.flight_state.armed and self.arming_client.call(self.arm_req).success == True:
            rospy.loginfo("Vehicle armed")

        if self.flight_state.mode != "OFFBOARD" and self.set_mode_client.call(self.offb_req).mode_sent == True:
            rospy.loginfo("OFFBOARD enabled")

    def takeoff_by_service(self):
        if not self.flight_state.armed and self.arming_client.call(self.arm_req).success == True:
            rospy.loginfo("Vehicle armed")

        takeoff_cmd = CommandTOLRequest()
        takeoff_cmd.altitude = 5.0
        try:
            res = self.takeoff_client(takeoff_cmd)
            if not res.success:
                rospy.logerr('Failed to take off')
        except rospy.ServiceException as e:
            rospy.logerr(e)

    def takeoff_cmd(self, event):
        if not self.flight_state.armed:
            self.arming_client.call(self.arm_req)

        if self.flight_state.mode != "OFFBOARD":
            self.set_mode_client.call(self.offb_req)

        self.local_pos_cmd_pub.publish(self.pos_cmd)

        if self.drone_state[0, 2] >= self.hover_height - 0.05:
            self.takeoff_cmd_timer.shutdown()
            self.reach_height()

    def draw_fsm_graph(self):
        self.get_graph().draw('fsm.pdf', prog='dot')


if __name__ == "__main__":

    manager = Manager()

    manager.launch()

    manager.draw_fsm_graph()

    rospy.spin()
