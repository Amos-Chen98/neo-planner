import rospy
import numpy as np
from matplotlib import cm
from pyquaternion import Quaternion
from std_msgs.msg import ColorRGBA
from tf.transformations import quaternion_from_euler
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


class Visualizer():
    def modify_wpts_markerarray(self, wpts_markerarray, pos_array):
        for i in range(len(pos_array)):
            wpts_markerarray.markers[i].header.stamp = rospy.get_rostime()
            wpts_markerarray.markers[i].action = Marker.ADD
            wpts_markerarray.markers[i].pose.position.x = pos_array[i][0]
            wpts_markerarray.markers[i].pose.position.y = pos_array[i][1]
            wpts_markerarray.markers[i].pose.position.z = pos_array[i][2]
            color = cm.jet(int(i*1.0/len(pos_array)*256))
            wpts_markerarray.markers[i].color = ColorRGBA(color[0], color[1], color[2], color[3])

        for i in range(len(pos_array), len(wpts_markerarray.markers)):
            wpts_markerarray.markers[i].action = Marker.DELETE

        return wpts_markerarray

    def modify_path_markerarray(self, path_markerarray, pos_array, vel_array):
        vel_max = np.max(vel_array)
        color_codes = (vel_array/vel_max*256).astype(int)

        for i in range(len(pos_array)-1):
            path_markerarray.markers[i].header.stamp = rospy.get_rostime()
            path_markerarray.markers[i].action = Marker.ADD
            line_head = [pos_array[i][0], pos_array[i][1], pos_array[i][2]]
            line_tail = [pos_array[i+1][0], pos_array[i+1][1], pos_array[i+1][2]]
            path_markerarray.markers[i].points = [Point(line_head[0], line_head[1], line_head[2]),
                                                  Point(line_tail[0], line_tail[1], line_tail[2])]
            color = cm.jet(color_codes[i])
            path_markerarray.markers[i].color = ColorRGBA(color[0], color[1], color[2], color[3])

        for i in range(len(pos_array)-1, len(path_markerarray.markers)):
            path_markerarray.markers[i].action = Marker.DELETE

        return path_markerarray

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
                marker.mesh_resource = "package://planner/models/meshes/iris.stl"
                # marker.mesh_resource = "package://planner/models/meshes/quad.stl"

            markerArray.markers.append(marker)

        return markerArray
