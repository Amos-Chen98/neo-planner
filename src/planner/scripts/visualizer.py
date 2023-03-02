'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2023-03-02 11:15:28
'''

import rospy
import numpy as np
from matplotlib import cm
from pyquaternion import Quaternion
from std_msgs.msg import ColorRGBA
from tf.transformations import quaternion_from_euler
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


class Visualizer():

    def get_path(self, pos_array, vel_array):
        path = MarkerArray()
        vel_max = np.max(vel_array)
        color_codes = (vel_array/vel_max*256).astype(int)

        for i in range(len(pos_array)-1):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.seq = i
            marker.header.stamp = rospy.get_rostime()
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            line_head = [pos_array[i][0], pos_array[i][1], pos_array[i][2]]
            line_tail = [pos_array[i+1][0], pos_array[i+1][1], pos_array[i+1][2]]
            marker.points = [Point(line_head[0], line_head[1], line_head[2]), Point(line_tail[0], line_tail[1], line_tail[2])]
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.1
            color = cm.jet(color_codes[i])
            marker.color = ColorRGBA(color[0], color[1], color[2], color[3])

            path.markers.append(marker)

        return path

    def get_marker_array(self, pos_array, marker_typeID, scale=1, step_length=1):
        '''
        Get marker array from pos array
        input: 
        pos_array: np.ndarray of (n,3)
        marker_typeID: int
        step_length: int
        output: markerArray: visualization_msgs.msg.MarkerArray
        '''
        markerArray = MarkerArray()

        for i in range(0, len(pos_array), step_length):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.seq = i
            marker.header.stamp = rospy.get_rostime()
            marker.id = i

            marker.type = marker_typeID
            marker.action = Marker.ADD
            marker.pose.position.x = pos_array[i][0]
            marker.pose.position.y = pos_array[i][1]
            marker.pose.position.z = pos_array[i][2]

            # to be changed
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0

            marker.scale.x = scale
            marker.scale.y = scale
            marker.scale.z = scale

            color = cm.jet(int(i*1.0/len(pos_array)*256))
            marker.color = ColorRGBA(color[0], color[1], color[2], color[3])

            if marker_typeID == 10:
                marker.mesh_resource = "package://simulator/models/meshes/iris.stl"
                # marker.mesh_resource = "package://simulator/models/meshes/quad.stl"

            markerArray.markers.append(marker)

        return markerArray
