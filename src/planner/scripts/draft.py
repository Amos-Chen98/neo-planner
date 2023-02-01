#! /usr/bin/env python
from nav_msgs.msg import Odometry
from visualizer import get_marker_array
from visualization_msgs.msg import MarkerArray
import rospy
import math
import numpy as np
from geometry_msgs.msg import PoseStamped, TwistStamped, Point, Vector3
from mavros_msgs.msg import Altitude, ExtendedState, HomePosition, State, WaypointList, PositionTarget
from mavros_msgs.srv import CommandBool, ParamGet, SetMode, WaypointClear, WaypointPush
from sensor_msgs.msg import NavSatFix, Imu
from nav_msgs.msg import Path, OccupancyGrid
from octomap_msgs.msg import Octomap
from traj_planner import MinJerkPlanner
from pyquaternion import Quaternion
import time
import octomap


class OctomapNode():
    def __init__(self, node_name="octomap_node"):
        # Node
        rospy.init_node(node_name, anonymous=False)

        # Member
        self.octomap = Octomap()
        self.octree = octomap.OcTree(0.1)
        self.generateEDT = True
        self.updateOctomap = True
        
        # Subscriber
        # self.octomap_sub = rospy.Subscriber('/octomap_binary', Octomap, self.octomap_cb)
        self.occupancy_map_sub = rospy.Subscriber('/projected_map', OccupancyGrid, self.occupancy_map_cb)

    def octomap_cb(self, octomap_in):
        rospy.loginfo('Octomap updated !')
        time0 = time.time()
        self.octomap = octomap_in

        octree_header = f'# Octomap OcTree binary file\nid {self.octomap.id}\n'
        octree_header += f'size 252291\nres {self.octomap.resolution}\ndata\n'
        octree_file = str.encode(octree_header)  # type: bytes

        # Read the octomap binary data and load it in the octomap wrapper class
        octree_data = np.array(self.octomap.data, dtype=np.int8).tobytes()  # type: bytes
        octree_file += octree_data
        # data_size = np.array(self.octomap.data, dtype=np.int8).shape
        # print(data_size)

        # An error is triggered because a wrong tree size has been specified in the
        # header. We did not find a way to extract the tree size from the octomap_in msg
        tree = octomap.OcTree(self.octomap.resolution)
        tree.readBinary(octree_file)
        self.octree = tree

        time1 = time.time()
        print("Time of building tree: %f" %(time1 - time0))

        test_point = np.array([0, 2, 1])
        node = tree.search(test_point)

        try:
            occupancy = tree.isNodeOccupied(node)
        except(octomap.NullPointerException):
            occupancy = False

        print(occupancy)

        time2 = time.time()
        print("Time of collision check: %f" %(time2 - time1))

        # Euclidean Distance Transform generation
        if self.generateEDT:
            print('Generating EDT...')
            bbmin = self.octree.getMetricMin()
            bbmax = self.octree.getMetricMax()
            # print(bbmin)
            # print(bbmax)
            maxdist = 50
            self.octree.dynamicEDT_generate(maxdist, bbmin, bbmax)
            # # The update computes distances in real unit (with sqrt)
            # # This step can be faster if we use squared distances instead
            self.octree.dynamicEDT_update(True)

        time3 = time.time()
        print("Time of building EDT: %f" %(time3 - time2))

    def occupancy_map_cb(self, data):
        rospy.loginfo('Got occupancy map!')
        self.occupancy_map = data
        

if __name__ == "__main__":
    octomap_node = OctomapNode()

    rospy.spin()
