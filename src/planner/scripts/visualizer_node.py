'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2023-01-29 16:40:14
'''

from nav_msgs.msg import Path
import rospy
from visualization_msgs.msg import MarkerArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker


class Visualizer():
    def __init__(self, node_name="visualizer"):
        # Node
        rospy.init_node(node_name, anonymous=False)
        self.real_path = Path()
        self.real_path.header.frame_id = "map"

        # Subscribers
        self.odom_sub = rospy.Subscriber('/mavros/local_position/odom', Odometry, self.odom_cb)

        # Publishers
        self.marker_pub = rospy.Publisher('/robotMarker', MarkerArray, queue_size=10)
        self.real_path_pub = rospy.Publisher('/real_path', Path, queue_size=10)
        # self.drone_marker_pub = rospy.Publisher('/drone_marker', Marker, queue_size=10)

    def odom_cb(self, data):
        pose_stamped = PoseStamped()
        pose_stamped.pose = data.pose.pose
        self.real_path.poses.append(pose_stamped)
        self.real_path_pub.publish(self.real_path)

        # drone_marker = Marker()
        # drone_marker.header.frame_id = "map"
        # drone_marker.pose = data.pose.pose
        # drone_marker.type = drone_marker.MESH_RESOURCE
        # drone_marker.mesh_resource = "package://simulator/models/meshes/iris.stl"
        # drone_marker.color.r = 81/256
        # drone_marker.color.g = 196/256
        # drone_marker.color.b = 211/256
        # drone_marker.color.a = 1
        # drone_marker.scale.x = 1
        # drone_marker.scale.y = 1
        # drone_marker.scale.z = 1
        # self.drone_marker_pub.publish(drone_marker)


if __name__ == "__main__":
    visualizer = Visualizer()

    rospy.spin()
