#! /usr/bin/env python
import os
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_path)
from mavros_msgs.srv import SetMode, SetModeRequest
import numpy as np
import rospy
from global_planner import GlobalPlanner


if __name__ == "__main__":
    rospy.wait_for_service("/mavros/set_mode")
    set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)
    offb_set_mode = SetModeRequest()
    offb_set_mode.custom_mode = 'OFFBOARD'

    global_planner = GlobalPlanner()

    tail_state = np.array([[50, 0, 10],
                           [0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]])

    # int_wpts = np.array([[10, 0, 10],
    #                     [10, 30, 15],
    #                     [5, 35, 10]])

    # int_wpts = get_int_wpts(head_state, tail_state)

    global_planner.plan(tail_state)

    global_planner.warm_up()

    if (set_mode_client.call(offb_set_mode).mode_sent == True):
        rospy.loginfo("OFFBOARD enabled")

    global_planner.publish_state_cmd()
    
    global_planner.publish_des_path()

    rospy.spin()
