'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2023-03-09 22:26:56
'''
import os
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_path)
from visualization_msgs.msg import Marker
from std_msgs.msg import String
from nav_msgs.msg import OccupancyGrid
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
        self.replan_time = 2.0  # the time interval between two replanning
        self.longitu_step_dis = 5.0  # the distance forward in each replanning
        self.lateral_step_length = 1.0  # if local target pos in obstacle, take lateral step
        self.move_vel = 1.0
        self.global_target = None
        # self.planning_mode = 'global'
        self.planning_mode = 'online'

        # Client / Service init
        try:
            rospy.wait_for_service('/mavros/cmd/arming')
            rospy.wait_for_service('/mavros/set_mode')
            rospy.wait_for_service("/mavros/cmd/takeoff")
        except rospy.ROSException:
            exit('Failed to connect to MAVROS services')
        self.arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)
        self.set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)
        self.takeoff_client = rospy.ServiceProxy("/mavros/cmd/takeoff", CommandTOL)

        # Subscribers
        self.flight_state_sub = rospy.Subscriber('/mavros/state', State, self.flight_state_cb)
        self.odom_sub = rospy.Subscriber('/mavros/local_position/odom', Odometry, self.odom_cb)
        self.target_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.trigger_plan)
        self.fsm_trigger_sub = rospy.Subscriber('/manager/trigger', String, self.trigger_fsm)
        self.occupancy_map_sub = rospy.Subscriber('/projected_map', OccupancyGrid, self.map.occupancy_map_cb)

        # Publishers
        self.local_pos_cmd_pub = rospy.Publisher("/mavros/setpoint_raw/local", PositionTarget, queue_size=10)
        self.local_target_pub = rospy.Publisher("/manager/local_target", PositionTarget, queue_size=1)
        self.target_vis_pub = rospy.Publisher('/global_target', Marker, queue_size=10)

        # FSM
        self.fsm = GraphMachine(model=self, states=['INIT', 'TRACKING', 'HOVER', 'PLANNING'], initial='INIT')
        self.fsm.add_transition(trigger='launch', source='INIT', dest='TRACKING', before="get_odom", after=['takeoff', 'print_current_state'])
        self.fsm.add_transition(trigger='reach_target', source='TRACKING', dest='HOVER', after=['print_current_state'])
        self.fsm.add_transition(trigger='init_planning', source='HOVER', dest='PLANNING', after=['start_planning', 'print_current_state'])
        self.fsm.add_transition(trigger='init_planning', source='TRACKING', dest='PLANNING', after=['start_planning', 'print_current_state'])
        self.fsm.add_transition(trigger='start_tracking', source='PLANNING', dest='TRACKING', after=['print_current_state'])
        # if triggered planning in state TRACKING, and the target is reached during planning, stay in PLANNING
        self.fsm.add_transition(trigger='reach_target', source='PLANNING', dest='PLANNING', after='print_current_state')
        self.fsm.add_transition(trigger='replan_timeout', source='TRACKING', dest='PLANNING', after=['publish_local_target', 'print_current_state'])
        self.fsm.add_transition(trigger='replan_timeout', source='HOVER', dest='PLANNING', after=['publish_local_target', 'print_current_state'])

    def print_current_state(self):
        rospy.loginfo("Current state: %s", self.state)

    def trigger_plan(self, target):
        rospy.loginfo("Global target: x = %f, y = %f", target.pose.position.x, target.pose.position.y)
        self.global_target = np.array([target.pose.position.x,
                                       target.pose.position.y,
                                       target.pose.position.z])
        self.vis_target()
        self.init_planning()

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

        self.local_target_pub.publish(self.pos_cmd)

    def trigger_fsm(self, trigger):
        if trigger.data == "reach_target":
            self.reach_target()
        elif trigger.data == "start_tracking":
            self.start_tracking()

    def start_planning(self):
        if self.planning_mode == 'global':
            global_target = PositionTarget()
            global_target.position.x = self.global_target[0]
            global_target.position.y = self.global_target[1]
            self.local_target_pub.publish(global_target)
        else:
            self.publish_local_target()
            self.replan_timer = rospy.Timer(rospy.Duration(self.replan_time), self.replan_timer_cb)

    def replan_timer_cb(self, event):
        self.replan_timeout()

    def publish_local_target(self):
        local_target = PositionTarget()
        current_pos = self.drone_state[0, :2]  # np.array
        global_target_pos = self.global_target[:2]

        # if current pos is close enough to global target, set local target to global target
        if np.linalg.norm(global_target_pos - current_pos) < self.longitu_step_dis:
            local_target.position.x = self.global_target[0]
            local_target.position.y = self.global_target[1]
            local_target.position.z = self.hover_height
            self.local_target_pub.publish(local_target)
            self.replan_timer.shutdown()
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

        local_target.position.x = local_target_pos[0]
        local_target.position.y = local_target_pos[1]

        # get local target vel
        goal_dir = (global_target_pos - local_target_pos)/np.linalg.norm(global_target_pos - local_target_pos)
        local_target.velocity.x = self.move_vel * goal_dir[0]
        local_target.velocity.y = self.move_vel * goal_dir[1]

        self.local_target_pub.publish(local_target)

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
            self.reach_target()

    def draw_fsm_graph(self):
        self.get_graph().draw('fsm.pdf', prog='dot')


if __name__ == "__main__":

    manager = Manager()

    manager.launch()

    manager.draw_fsm_graph()

    rospy.spin()
