#! /usr/bin/env python
import os
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_path)
from mavros_msgs.srv import SetMode, SetModeRequest
import numpy as np
import rospy
from global_planner import GlobalPlanner
import time


if __name__ == "__main__":
    rospy.wait_for_service("/mavros/set_mode")
    set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)
    offb_set_mode = SetModeRequest()
    offb_set_mode.custom_mode = 'OFFBOARD'

    global_planner = GlobalPlanner()

    tail_state = np.array([[11, 0],
                           [0, 0],
                           [0, 0],
                           [0, 0]]) # p,v,a, jer in map frame

    time.sleep(1)

    global_planner.plan(tail_state)

    global_planner.warm_up()

    if (set_mode_client.call(offb_set_mode).mode_sent == True):
        rospy.loginfo("OFFBOARD enabled")

    global_planner.publish_state_cmd()
    
    global_planner.visualize_des_wpts()

    global_planner.visualize_des_path()

    rospy.spin()
