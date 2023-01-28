'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2023-01-28 22:26:09
'''
import os
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_path)
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
from nav_msgs.msg import Path
from octomap_msgs.msg import Octomap
from traj_planner import MinJerkPlanner
from pyquaternion import Quaternion



class Config():
    def __init__(self):
        self.v_max = rospy.get_param("~v_max", 5.0)
        self.T_min = rospy.get_param("~T_min", 2.0)
        self.T_max = rospy.get_param("~T_max", 20)
        self.kappa = rospy.get_param("~kappa", 50)
        self.weights = rospy.get_param("~weights", [1.0, 1.0, 0.001])


class GlobalPlanner():
    def __init__(self, node_name="global_planner"):
        # Node
        rospy.init_node(node_name, anonymous=False)

        # Members
        self.octomap = None
        self.odom = None
        self.drone_state = np.zeros((3, 3))  # p,v,a in inertia frame
        planner_config = Config()
        self.planner = MinJerkPlanner(planner_config)
        self.state_cmd = PositionTarget()
        self.state_cmd.coordinate_frame = 1

        # Subscribers
        self.octomap_sub = rospy.Subscriber('/octomap_binary', Octomap, self.octomap_cb)
        self.odom_sub = rospy.Subscriber('/mavros/local_position/odom', Odometry, self.odom_cb)

        # Publishers
        self.local_pos_cmd_pub = rospy.Publisher("/mavros/setpoint_raw/local", PositionTarget, queue_size=10)
        self.marker_pub = rospy.Publisher('/robotMarker', MarkerArray, queue_size=10)

    def octomap_cb(self, data):
        rospy.loginfo('Octomap updated !')
        self.octomap = data

    def odom_cb(self, data):
        '''
        Local velocity conversion to global frame
        '''
        self.odom = data
        local_pos = np.array([data.pose.pose.position.x,
                              data.pose.pose.position.y,
                              data.pose.pose.position.z])
        global_pos = local_pos
        quat = Quaternion(data.pose.pose.orientation.w,
                          data.pose.pose.orientation.x,
                          data.pose.pose.orientation.y,
                          data.pose.pose.orientation.z)
        local_vel = np.array([data.twist.twist.linear.x,
                              data.twist.twist.linear.y,
                              data.twist.twist.linear.z,
                              ])
        global_vel = quat.inverse.rotate(local_vel)
        self.drone_state[0] = global_pos
        self.drone_state[1] = global_vel

    def plan(self, head_state, tail_state, int_wpts, ts):
        '''
        Input: 
        current drone state: np.ndarray of (3,3)
        target state: np.ndarray of (3,3)
        octomap: tuple, for obstacle avoidance

        Store traj in self.planner
        '''
        self.planner.plan(head_state, tail_state, int_wpts, ts)
        rospy.loginfo("Trajectory planning finished!")

    def warm_up(self):
        # Send a few setpoints before switching to OFFBOARD mode
        self.state_cmd.position.x = self.drone_state[0][0]
        self.state_cmd.position.y = self.drone_state[0][1]
        self.state_cmd.position.z = self.drone_state[0][2]
        rate = rospy.Rate(100)
        for _ in range(5):
            if (rospy.is_shutdown()):
                break
            self.local_pos_cmd_pub.publish(self.state_cmd)
            rate.sleep()        

    def publish_state_cmd(self):
        '''
        When triggered, start to publish full state cmd
        '''
        self.des_state, self.traj_time, hz = self.planner.get_full_state_cmd()
        self.des_state_index = 0
        self.start_time = rospy.get_time()
        self.timer = rospy.Timer(rospy.Duration(1/hz), self.timer_cb)

    def timer_cb(self, event):
        self.state_cmd.position.x = self.des_state[self.des_state_index][0][0]
        self.state_cmd.position.y = self.des_state[self.des_state_index][0][1]
        self.state_cmd.position.z = self.des_state[self.des_state_index][0][2]

        self.state_cmd.velocity.x = self.des_state[self.des_state_index][1][0]
        self.state_cmd.velocity.y = self.des_state[self.des_state_index][1][1]
        self.state_cmd.velocity.z = self.des_state[self.des_state_index][1][2]

        self.state_cmd.acceleration_or_force.x = self.des_state[self.des_state_index][2][0]
        self.state_cmd.acceleration_or_force.y = self.des_state[self.des_state_index][2][1]
        self.state_cmd.acceleration_or_force.z = self.des_state[self.des_state_index][2][2]

        self.state_cmd.yaw = 0

        self.local_pos_cmd_pub.publish(self.state_cmd)

        self.des_state_index += 1

        if event.current_real.to_sec() - self.start_time > self.traj_time:
            self.des_state_index -= 1  # keep publishing the last des_state
            # self.timer.shutdown()

    def visualize_traj(self):
        pos = self.planner.get_pos_array()
        pos_z = np.array([5.0*np.ones(len(pos))]).T
        pos_array = np.concatenate((pos, pos_z), axis=1)
        markerArray = get_marker_array(pos_array)
        self.marker_pub.publish(markerArray)
        rospy.loginfo("markerArray published!!!")
