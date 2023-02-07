#! /usr/bin/env python
import rospy
import tf2_ros
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped


class TFServer():
    def __init__(self, node_name="dynamic_tf"):
        # Node
        rospy.init_node(node_name, anonymous=False)

        # Subscribers
        self.odom_sub = rospy.Subscriber('/mavros/local_position/odom', Odometry, self.odom_cb)

        # Publishers
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

    def odom_cb(self, odom):
        tfs = TransformStamped()
        tfs.header.stamp = rospy.Time.now()
        tfs.header.frame_id = "odom"
        tfs.child_frame_id = "base_link"

        tfs.transform.translation.x = odom.pose.pose.position.x
        tfs.transform.translation.y = odom.pose.pose.position.y
        tfs.transform.translation.z = odom.pose.pose.position.z

        tfs.transform.rotation = odom.pose.pose.orientation
        self.tf_broadcaster.sendTransform(tfs)


if __name__ == "__main__":
    tf_server = TFServer()

    rospy.spin()
