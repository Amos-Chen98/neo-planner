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


# using pyquaternion
# q1 = Quaternion(0.35, 0.2, 0.3, 0.1)
# q2 = Quaternion(-0.5, 0.4, -0.1, 0.2)
# t1 = np.array([0.3, 0.1, 0.1])
# t2 = np.array([-0.1, 0.5, 0.3])

# p1 = np.array([0.5, 0, 0.2])
# p1_w = q1.inverse.rotate(p1 - t1)

# p1_2 = q2.rotate(p1_w) + t2
# print(p1_2)


# t_end = time.time()
# print("time cost: %f" % (t_end - t_start))
start_pos = np.array([0, 10, 0])
target_pos = np.array([8, 8, 8])
num = 3
dim = len(start_pos)
int_wpts = np.zeros((num, dim))
for i in range(dim):
    step_length = (target_pos[i] - start_pos[i])/(num + 1)
    int_wpts[:, i] = np.linspace(start_pos[i] + step_length, target_pos[i], num, endpoint=False)

print(int_wpts)
