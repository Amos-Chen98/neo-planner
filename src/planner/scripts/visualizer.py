'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2023-01-16 20:42:30
'''

import rospy
from matplotlib import cm
from numpy import linalg as LA
from pyquaternion import Quaternion
from std_msgs.msg import ColorRGBA
from tf.transformations import quaternion_from_euler
from visualization_msgs.msg import Marker, MarkerArray


def get_marker_array(pos_array):
    markerArray = MarkerArray()

    for i in range(0,len(pos_array),2):
        robotMarker = Marker()
        robotMarker.header.frame_id = "world"
        robotMarker.header.seq = i
        robotMarker.header.stamp = rospy.get_rostime()
        robotMarker.ns = "robot"
        robotMarker.id = i

        robotMarker.type = robotMarker.MESH_RESOURCE
        # robotMarker.type = Marker.CUBE
        robotMarker.action = Marker.ADD
        robotMarker.pose.position.x = pos_array[i][0]
        robotMarker.pose.position.y = pos_array[i][1]
        robotMarker.pose.position.z = pos_array[i][2]

        # to be changed
        robotMarker.pose.orientation.x = 0.0
        robotMarker.pose.orientation.y = 0.0
        robotMarker.pose.orientation.z = 0.0
        robotMarker.pose.orientation.w = 1.0

        robotMarker.scale.x = 1.0
        robotMarker.scale.y = 1.0
        robotMarker.scale.z = 1.0

        # robotMarker.lifetime = rospy.Duration()
        # robotMarker.lifetime = 10

        color = cm.jet(int(((i*1.0)/len(pos_array)*256)))
        robotMarker.color = ColorRGBA(color[0], color[1], color[2], color[3])
        robotMarker.mesh_resource = "package://planner/models/quad.stl"

        markerArray.markers.append(robotMarker)

    return markerArray
