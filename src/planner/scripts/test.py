'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2023-04-30 20:38:55
'''
from pyquaternion import Quaternion

q = Quaternion(1, 2, 3, 4)  # Example quaternion

R = q.rotation_matrix  # Convert quaternion to rotation matrix

a = R.reshape(-1)
print(a)
print(a.shape)
