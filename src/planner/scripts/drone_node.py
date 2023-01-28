#! /usr/bin/env python
import os
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_path)
import rospy
from drone import Drone

drone = Drone()

drone.arm()

drone.takeoff()

rospy.spin()
