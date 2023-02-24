#! /usr/bin/env python
from mavros_msgs.srv import SetMode, SetModeRequest
from mavros_msgs.srv import SetMode
from nav_msgs.msg import Odometry
from mavros_msgs.srv import SetMode
from mavros_msgs.msg import State, PositionTarget
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest, CommandTOL, CommandTOLRequest
import rospy
import numpy as np
from transitions import Machine
from transitions.extensions import GraphMachine
from geometry_msgs.msg import PoseStamped, TwistStamped, Point, Vector3
from std_msgs.msg import String


class Manager():
    def __init__(self, node_name="manager"):
        # Node
        rospy.init_node(node_name, anonymous=False)
        self.flight_state = State()
        self.exec_state = "INIT"
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

        # Publishers
        self.local_pos_cmd_pub = rospy.Publisher("/mavros/setpoint_raw/local", PositionTarget, queue_size=10)
        self.local_target_pub = rospy.Publisher("/manager/local_target", PoseStamped, queue_size=10)

        # FSM
        self.fsm = Machine(model=self, states=['INIT', 'TRACKING', 'HOVER', 'PLANNING'], initial='INIT')
        self.fsm.add_transition(trigger='launch', source='INIT', dest='TRACKING', before="get_odom", after='takeoff')
        self.fsm.add_transition(trigger='reach_target', source='TRACKING', dest='HOVER')
        self.fsm.add_transition(trigger='start_planning', source='HOVER', dest='PLANNING')
        self.fsm.add_transition(trigger='start_tracking', source='PLANNING', dest='TRACKING')
        self.fsm.add_transition(trigger='start_planning', source='TRACKING', dest='PLANNING')
        self.fsm.add_transition(trigger='start_planning', source='*', dest='PLANNING')

    def trigger_fsm(self, data):
        if data.data == "reach_target":
            self.reach_target()
            rospy.loginfo("Current state: %s", self.state)
        elif data.data == "start_tracking":
            self.start_tracking()
            rospy.loginfo("Current state: %s", self.state)

    def trigger_plan(self, data):
        self.local_target_pub.publish(data)
        self.start_planning()
        rospy.loginfo("Current state: %s", self.state)

    def flight_state_cb(self, data):
        self.flight_state = data

    def get_odom(self):
        rospy.loginfo("Current state: %s", self.state)
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
        rospy.loginfo("Current state: %s", self.state)
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
            rospy.loginfo("Current state: %s", self.state)


if __name__ == "__main__":

    manager = Manager()

    manager.launch()

    rospy.spin()
