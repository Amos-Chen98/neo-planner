'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2023-01-28 15:48:26
'''
import rospy
import math
import numpy as np
from geometry_msgs.msg import PoseStamped, TwistStamped, Point, Vector3
from mavros_msgs.msg import Altitude, ExtendedState, HomePosition, State, WaypointList, PositionTarget
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, WaypointClear, WaypointPush, CommandTOL, CommandTOLRequest
from sensor_msgs.msg import NavSatFix, Imu
from nav_msgs.msg import Path




class Drone():
    '''
    The node for a single drone. Holding the basic state information of the drone
    '''

    def __init__(self, node_name="drone"):
        self.node_name = node_name
        self.state = State()
        self.local_position = None
        self.local_state_cmd = PositionTarget()

        # Node
        rospy.init_node(self.node_name, anonymous=False)

        # Client / Service init
        try:
            rospy.wait_for_service('/mavros/cmd/arming')
            rospy.wait_for_service('/mavros/set_mode')
            rospy.wait_for_service("/mavros/cmd/takeoff")
            rospy.loginfo('All MAVROS services are up !')
        except rospy.ROSException:
            exit('Failed to connect to MAVROS services')

        self.set_arming_srv = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
        self.set_mode_srv = rospy.ServiceProxy('/mavros/set_mode', SetMode)
        self.takeoff_client = rospy.ServiceProxy("/mavros/cmd/takeoff", CommandTOL)

        # Subscribers
        self.state_sub = rospy.Subscriber("mavros/state", State, callback=self.state_cb)
        self.local_pos_sub = rospy.Subscriber('mavros/local_position/pose', PoseStamped, self.local_position_cb)

        # Publishers
        self.local_pos_cmd_pub = rospy.Publisher("mavros/setpoint_position/local", PoseStamped, queue_size=10)

    def state_cb(self, data):
        self.state = data

    def local_position_cb(self, data):
        self.local_position = data
        # self.pos = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
        # self.pos += self.start_pos

    def set_mode(self, mode, timeout=5, loop_freq=1):
        """
        mode: PX4 mode string, timeout(int): seconds, loop_freq(int): seconds
        """
        rospy.loginfo("Setting FCU mode: {0}".format(mode))
        rate = rospy.Rate(loop_freq)
        mode_set = False

        for i in range(timeout * loop_freq):
            if self.state.mode == mode:
                mode_set = True
                rospy.loginfo("Mode set successfully in {} sec".format(i / loop_freq))
                break
            else:
                try:
                    res = self.set_mode_srv(0, mode)  # 0 is custom mode
                    if not res.mode_sent:
                        rospy.logerr("failed to send mode command")
                except rospy.ServiceException as e:
                    rospy.logerr(e)
            try:
                rate.sleep()
            except rospy.ROSException as e:
                self.fail(e)

        if not mode_set:
            exit('Timeout: failed to set the mode {} !'.format(mode))

    def arm(self, freq=0.2):
        arm_cmd = CommandBoolRequest()
        arm_cmd.value = True

        if self.state.armed:
            return
        if not hasattr(self, 'last_arm_request'):
            self.last_arm_request = rospy.get_time()

        now = rospy.get_time()
        if self.last_arm_request + 1./freq > now:
            self.last_arm_request = now
            # rospy.loginfo('Try to set arming...')
            try:
                res = self.set_arming_srv(arm_cmd)
                if not res.success:
                    rospy.logerr('Failed to send arm command')
                else:
                    rospy.loginfo("Armed!")
            except rospy.ServiceException as e:
                rospy.logerr(e)

    def takeoff(self):
        takeoff_cmd = CommandTOLRequest()
        takeoff_cmd.altitude = 5.0
        try:
            res = self.takeoff_client(takeoff_cmd)
            if not res.success:
                rospy.logerr('Failed to take off')
        except rospy.ServiceException as e:
            rospy.logerr(e)

    def pub_state(self):
        rospy.loginfo("State: blank")
