'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2023-01-16 20:38:07
'''

import os
import sys

current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_path)

import numpy as np
import rospy
from traj_planner import MinJerkPlanner
from visualization_msgs.msg import MarkerArray
from visualizer import get_marker_array


class Config():
    def __init__(self):
        self.v_max = rospy.get_param("~v_max")
        self.T_min = rospy.get_param("~T_min")
        self.T_max = rospy.get_param("~T_max")
        self.kappa = rospy.get_param("~kappa")
        self.weights = rospy.get_param("~weights")


rospy.init_node("global_planning")
rospy.loginfo("global_planning initialized!!!")
rospy.sleep(1)

markerPub = rospy.Publisher('/robotMarker', MarkerArray, queue_size=10)

# head_state = np.array([[0.0, 0, 5],
#                     [0, 0, 0],
#                     [0, 0, 0],
#                     [0, 0, 0]])
# tail_state = np.array([[100.0, 100, 5],
#                     [0, 0, 0],
#                     [0, 0, 0],
#                     [0, 0, 0]])

# wpts = np.array([[40.0, 0, 10],
#                 [100, 30, 10],
#                 [100, 50, 10],
#                 [20, 60, 10],
#                 [70, 90, 10],
#                 [10, 120, 10]])

head_state = np.array([[0.0, 0, 5],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]])
tail_state = np.array([[10.0, 10, 5],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]])

wpts = np.array([[10, 0, 10],
                [10, 30, 15],
                [5,35,10]])

ts = 10 * np.ones((len(wpts)+1,))

planner_config = Config()

planner = MinJerkPlanner(head_state, tail_state, wpts, ts, planner_config)

planner.optimize()

pos = planner.get_pos_array()

pos_z = np.array([5.0*np.ones(len(pos))]).T

pos_array = np.concatenate((pos, pos_z), axis=1)

markerArray = get_marker_array(pos_array)

markerPub.publish(markerArray)

rospy.loginfo("markerArray published!!!")

rospy.spin()
