#! /usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest, CommandTOL, CommandTOLRequest

current_state = State()


def state_cb(msg):
    global current_state
    current_state = msg


if __name__ == "__main__":
    rospy.init_node("my_offb_node_py")

    state_sub = rospy.Subscriber("mavros/state", State, callback=state_cb)

    rospy.wait_for_service("/mavros/cmd/arming")
    arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)

    rospy.wait_for_service("/mavros/cmd/takeoff")
    takeoff_client = rospy.ServiceProxy("/mavros/cmd/takeoff", CommandTOL)

    # Setpoint publishing MUST be faster than 2Hz
    rate = rospy.Rate(20)

    # Wait for Flight Controller connection
    while (not rospy.is_shutdown() and not current_state.connected):
        rate.sleep()

    offb_set_mode = SetModeRequest()
    offb_set_mode.custom_mode = 'OFFBOARD'

    arm_cmd = CommandBoolRequest()
    arm_cmd.value = True

    takeoff_cmd = CommandTOLRequest()
    takeoff_cmd.altitude = 2.5

    last_req = rospy.Time.now()

    # arm
    if (arming_client.call(arm_cmd).success == True):
        rospy.loginfo("Vehicle armed")

    # takeoff
    if (takeoff_client.call(takeoff_cmd).success == True):
        rospy.loginfo("Take off")
