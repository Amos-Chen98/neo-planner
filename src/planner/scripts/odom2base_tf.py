'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2023-07-08 20:59:34
'''

import rospy
import tf2_ros
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped


class TFServer():
    def __init__(self, node_name="dynamic_tf"):
        # Node
        rospy.init_node(node_name, anonymous=False)

        self.tfs = TransformStamped()
        self.tfs.header.frame_id = "odom"
        self.tfs.child_frame_id = "base_link"

        # Subscribers
        self.odom_sub = rospy.Subscriber('mavros/local_position/odom', Odometry, self.odom_cb)

        # Publishers
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

    def odom_cb(self, odom):
        self.tfs.header.stamp = rospy.Time.now()

        self.tfs.transform.translation.x = odom.pose.pose.position.x
        self.tfs.transform.translation.y = odom.pose.pose.position.y
        self.tfs.transform.translation.z = odom.pose.pose.position.z

        self.tfs.transform.rotation = odom.pose.pose.orientation

        self.tf_broadcaster.sendTransform(self.tfs)


if __name__ == "__main__":
    tf_server = TFServer()

    rospy.spin()
