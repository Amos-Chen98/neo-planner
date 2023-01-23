#! /usr/bin/env python
import rospy
from math import sqrt
from math import sin
from math import cos
from mavros_msgs.msg import PositionTarget
from mavros_msgs.srv import SetMode, SetModeRequest


class State():
    def __init__(self):
        self.p_x = 0.0
        self.p_y = 0.0
        self.p_z = 10.0

        self.v_x = 0.0
        self.v_y = 0.0
        self.v_z = 0.0

        self.a_x = 0.0
        self.a_y = 0.0
        self.a_z = 0.0

        self.yaw = 0.0

    def get_state(self, t):
        '''
        using Lemniscate_of_Bernoulli, ref: https://zhuanlan.zhihu.com/p/448214022
        '''
        self.p_x = (60*sqrt(2)*cos(0.1*t) + 20*cos(0.2*t) + 60) / \
            (2*sqrt(2)*cos(0.1*t) + 3)

        self.p_y = (20*sqrt(2)*sin(0.1*t) + 20*sin(0.2*t)) / \
            (2*sqrt(2)*cos(0.1*t) + 3)

        self.p_z = 10.0

        self.v_x = (-6.0*sqrt(2)*sin(0.1*t) - 4.0*sin(0.2*t))/(2*sqrt(2)*cos(0.1*t) + 3) + 0.2*sqrt(
            2)*(60*sqrt(2)*cos(0.1*t) + 20*cos(0.2*t) + 60)*sin(0.1*t)/(2*sqrt(2)*cos(0.1*t) + 3)**2

        self.v_y = 0.2*sqrt(2)*(20*sqrt(2)*sin(0.1*t) + 20*sin(0.2*t))*sin(0.1*t)/(2*sqrt(2)*cos(
            0.1*t) + 3)**2 + (2.0*sqrt(2)*cos(0.1*t) + 4.0*cos(0.2*t))/(2*sqrt(2)*cos(0.1*t) + 3)

        self.v_z = 0.0

        self.a_x = 0.4*sqrt(2)*(-6.0*sqrt(2)*sin(0.1*t) - 4.0*sin(0.2*t))*sin(0.1*t)/(2*sqrt(2)*cos(0.1*t) + 3)**2 + (-0.6*sqrt(2)*cos(0.1*t) - 0.8*cos(0.2*t))/(2*sqrt(2)*cos(0.1*t) + 3) + 0.02*sqrt(
            2)*(60*sqrt(2)*cos(0.1*t) + 20*cos(0.2*t) + 60)*cos(0.1*t)/(2*sqrt(2)*cos(0.1*t) + 3)**2 + 0.16*(60*sqrt(2)*cos(0.1*t) + 20*cos(0.2*t) + 60)*sin(0.1*t)**2/(2*sqrt(2)*cos(0.1*t) + 3)**3

        self.a_y = (-0.2*sqrt(2)*sin(0.1*t) - 0.8*sin(0.2*t))/(2*sqrt(2)*cos(0.1*t) + 3) + 0.02*sqrt(2)*(20*sqrt(2)*sin(0.1*t) + 20*sin(0.2*t))*cos(0.1*t)/(2*sqrt(2)*cos(0.1*t) + 3)**2 + 0.16 * \
            (20*sqrt(2)*sin(0.1*t) + 20*sin(0.2*t))*sin(0.1*t)**2/(2*sqrt(2)*cos(0.1*t) + 3)**3 + 0.4 * \
            sqrt(2)*(2.0*sqrt(2)*cos(0.1*t) + 4.0*cos(0.2*t)) * \
            sin(0.1*t)/(2*sqrt(2)*cos(0.1*t) + 3)**2

        self.a_z = 0.0


if __name__ == "__main__":
    rospy.init_node("traj_tracking")

    cmd_publisher = rospy.Publisher(
        "/mavros/setpoint_raw/local", PositionTarget, queue_size=10)

    rospy.wait_for_service("/mavros/set_mode")
    set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)

    offb_set_mode = SetModeRequest()
    offb_set_mode.custom_mode = 'OFFBOARD'

    state = State()
    des_state = PositionTarget()
    des_state.coordinate_frame = 1

    rate = rospy.Rate(300)

    last_req = rospy.Time.now().to_sec()

    # Send a few setpoints before starting
    for _ in range(300):
        if (rospy.is_shutdown()):
            break

        t = rospy.Time.now().to_sec() - last_req
        state.get_state(t)
        des_state.position.x = state.p_x
        des_state.position.y = state.p_y
        des_state.position.z = state.p_z

        des_state.velocity.x = state.v_x
        des_state.velocity.y = state.v_y
        des_state.velocity.z = state.v_z

        des_state.acceleration_or_force.x = state.a_x
        des_state.acceleration_or_force.y = state.a_y
        des_state.acceleration_or_force.z = state.a_z

        des_state.yaw = state.yaw

        cmd_publisher.publish(des_state)

        rate.sleep()

    # OFFBOARD mode
    set_mode_client.call(offb_set_mode)

    last_req = rospy.Time.now().to_sec()

    while (not rospy.is_shutdown()):
        t = rospy.Time.now().to_sec() - last_req
        state.get_state(t)
        des_state.position.x = state.p_x
        des_state.position.y = state.p_y
        des_state.position.z = state.p_z

        des_state.velocity.x = state.v_x
        des_state.velocity.y = state.v_y
        des_state.velocity.z = state.v_z

        des_state.acceleration_or_force.x = state.a_x
        des_state.acceleration_or_force.y = state.a_y
        des_state.acceleration_or_force.z = state.a_z

        des_state.yaw = state.yaw

        cmd_publisher.publish(des_state)

        rate.sleep()
