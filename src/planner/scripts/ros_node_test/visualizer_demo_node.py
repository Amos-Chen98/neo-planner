'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2023-08-29 20:35:20
'''
from nav_msgs.msg import Path
import rospy
from visualization_msgs.msg import MarkerArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
import numpy as np
from pyquaternion import Quaternion


class DroneState():
    def __init__(self):
        self.global_pos = np.zeros(3)
        self.global_vel = np.zeros(3)
        self.local_vel = np.zeros(3)
        self.attitude = Quaternion()  # ref: http://kieranwynn.github.io/pyquaternion/
        self.yaw = 0.0


class Visualizer():
    def __init__(self, node_name="visualizer"):
        # Node
        rospy.init_node(node_name, anonymous=False)
        self.real_path = Path()
        self.real_path.header.frame_id = "map"
        self.odom_received = False
        self.ready_to_pub = False

        self.drone_marker = Marker()
        self.drone_marker.header.frame_id = "map"
        self.drone_marker.type = self.drone_marker.MESH_RESOURCE
        self.drone_marker.mesh_resource = "package://planner/models/meshes/iris.stl"
        self.drone_marker.color.r = 81/256
        self.drone_marker.color.g = 196/256
        self.drone_marker.color.b = 211/256
        self.drone_marker.color.a = 1
        self.drone_marker.scale.x = 0.6
        self.drone_marker.scale.y = 0.6
        self.drone_marker.scale.z = 0.6

        self.drone_marker_array = MarkerArray()
        self.marker_id = 0

        # Subscribers
        self.odom_sub = rospy.Subscriber('/mavros/local_position/odom', Odometry, self.odom_cb)
        self.trigger_sub = rospy.Subscriber('/des_wpts', MarkerArray, self.trigger_cb)

        # Publishers
        self.real_path_pub = rospy.Publisher('/real_path', Path, queue_size=10)
        self.drone_marker_array_pub = rospy.Publisher('/drone_marker_array', MarkerArray, queue_size=10)

    def odom_cb(self, data):
        '''
        1. store the drone's global status
        2. publish dynamic tf transform from map frame to camera frame
        (Currently, regard camera frame as drone body frame)
        '''
        self.odom_received = True
        self.odom = data

    def trigger_cb(self, data):
        self.ready_to_pub = True

    def pub_drone_snapshots(self):
        # wait for odom
        while not self.odom_received or not self.ready_to_pub:
            rospy.sleep(0.01)
        rospy.loginfo("Let's go!")

        self.drone_vis_timer = rospy.Timer(rospy.Duration(0.5), self.add_drone_snapshots)

    def add_drone_snapshots(self, event):
        self.drone_marker.id = self.marker_id
        self.drone_marker.header.stamp = rospy.Time.now()
        self.drone_marker.pose = self.odom.pose.pose

        self.drone_marker_array.markers.append(self.drone_marker)

        self.drone_marker_array_pub.publish(self.drone_marker_array)
        self.marker_id += 1


if __name__ == "__main__":
    visualizer = Visualizer()

    visualizer.pub_drone_snapshots()

    rospy.spin()
