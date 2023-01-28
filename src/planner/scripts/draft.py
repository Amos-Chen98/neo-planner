#! /usr/bin/env python
import numpy as np
import rospy
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
import time

# class Puber():
#     def __init__(self):
#         rospy.init_node("tes_node")
#         self.timer = None

#     def my_callback(self, event):
#         print('Timer called at ' + str(event.current_real.to_sec()))
#         if rospy.get_time() - self.start_time > 5:
#             print("time to STOP!!")
#             self.timer.shutdown()

#     def pub(self):
#         self.start_time = rospy.get_time()
#         self.timer = rospy.Timer(rospy.Duration(1), self.my_callback)


# puber = Puber()
# puber.pub()
# rospy.spin()

# q = [-0.71934025092983234, 1.876085535681999e-06, 3.274841213980097e-08, 0.69465790385533299]
# r = Rotation.from_quat(q)
# Rm = r.as_matrix()
# print(r)
# print(Rm)

t_start = time.time()

# using pyquaternion
# q = Quaternion(1, 1, 4, 0)
# q_inv = q.inverse
# v1 = [1, 0, 0]
# v2 = q.rotate(v1)
# v3 = q_inv.rotate(v2)
# print(v2)
# print(np.linalg.norm(v2))

q1 = Quaternion(0.35, 0.2, 0.3, 0.1)
q2 = Quaternion(-0.5, 0.4, -0.1, 0.2)
t1 = np.array([0.3, 0.1, 0.1])
t2 = np.array([-0.1, 0.5, 0.3])

p1 = np.array([0.5, 0, 0.2])
p1_w = q1.inverse.rotate(p1 - t1)

p1_2 = q2.rotate(p1_w) + t2
print(p1_2)


# using scipy.spatial.transform
# q = [1, 1, 4, 0]
# Rt = Rotation.from_quat(q).as_matrix()
# print(Rt)
# v1 = np.array([1, 0, 0])
# v1 = [1, 0, 0]
# v2 = np.dot(Rt, v1)
# print(v2)
# print(np.linalg.norm(v2))

# q1 = np.array([0.35, 0.2, 0.3, 0.1])
# q2 = np.array([-0.5, 0.4, -0.1, 0.2])
# R1 = Rotation.from_quat(q1).as_matrix()
# R2 = Rotation.from_quat(q2).as_matrix()
# t1 = np.array([[0.3, 0.1, 0.1]])
# t2 = np.array([[0.3, 0.1, 0.1]])

t_end = time.time()
print("time cost: %f" % (t_end - t_start))
