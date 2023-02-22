#! /usr/bin/env python
from mavros_msgs.srv import SetMode, SetModeRequest
from mavros_msgs.srv import SetMode
from nav_msgs.msg import Odometry
from mavros_msgs.srv import SetMode
from mavros_msgs.msg import State, PositionTarget
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest, CommandTOL, CommandTOLRequest
import rospy


class Manager():
    def __init__(self, node_name="mission_manager"):
        # Node
        rospy.init_node(node_name, anonymous=True)
        self.odom_received = False
        self.taking_off = False
        self.hover_height = 2.0
        self.flight_state = State()
        self.exec_state = "INIT"

        self.offb_req = SetModeRequest()
        self.offb_req.custom_mode = 'OFFBOARD'
        self.arm_req = CommandBoolRequest()
        self.arm_req.value = True
        self.pos_cmd = PositionTarget()
        self.pos_cmd.coordinate_frame = 1
        self.pos_cmd.position.z = self.hover_height

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

        # Publishers
        self.local_pos_cmd_pub = rospy.Publisher("/mavros/setpoint_raw/local", PositionTarget, queue_size=10)

        # FSM
        self.fsm_timer = rospy.Timer(rospy.Duration(0.01), self.fsm)

    def fsm(self, event):
        if self.exec_state == "INIT":
            if self.odom_received and self.flight_state.connected:
                rospy.loginfo("Connected to vehicle")
                self.exec_state = "TAKEOFF"
        elif self.exec_state == "TAKEOFF":
            if self.taking_off == False:
                self.takeoff()
                # self.takeoff_by_service() # This is the alternative way to takeoff, but it is not working

    def flight_state_cb(self, data):
        self.flight_state = data

    def odom_cb(self, data):
        self.odom_received = True

    def takeoff(self):
        self.taking_off = True
        self.takeoff_cmd_timer = rospy.Timer(rospy.Duration(0.1), self.takeoff_cmd)

        if not self.flight_state.armed and self.arming_client.call(self.arm_req).success == True:
            rospy.loginfo("Vehicle armed")

        if self.flight_state.mode != "OFFBOARD" and self.set_mode_client.call(self.offb_req).mode_sent == True:
            rospy.loginfo("OFFBOARD enabled")

    def takeoff_by_service(self):
        self.taking_off = True

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
        if not self.flight_state.armed and self.arming_client.call(self.arm_req).success == True:
            rospy.loginfo("Vehicle re-armed")

        if self.flight_state.mode != "OFFBOARD" and self.set_mode_client.call(self.offb_req).mode_sent == True:
            rospy.loginfo("OFFBOARD re-enabled")

        self.local_pos_cmd_pub.publish(self.pos_cmd)


if __name__ == "__main__":

    mission_manager = Manager()

    rospy.spin()
