'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2023-01-08 17:11:39
'''

import numpy as np

pos = np.array([[40.0, 0],
                [100, 30],
                [100, 50],
                [20, 60],
                [70, 90],
                [10, 120]])

pos_z = np.array([5*np.ones(len(pos))]).T

print(pos.shape)
print(pos_z.shape)

pos_array = np.concatenate((pos, pos_z), axis=1)

print(pos_array)
print(pos_array.shape)
