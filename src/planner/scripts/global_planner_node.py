#! /usr/bin/env python
import os
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_path)
import rospy
from global_planner import GlobalPlanner


if __name__ == "__main__":
    
    global_planner = GlobalPlanner()

    rospy.spin()
