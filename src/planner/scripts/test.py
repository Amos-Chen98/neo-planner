'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2023-02-28 16:47:19
'''
import numpy as np
import time


def get_int_wpts(head_state, tail_state):
    start_pos = head_state[0]
    target_pos = tail_state[0]
    straight_length = np.linalg.norm(target_pos - start_pos)
    int_wpts_num = max(int(straight_length/2 - 1), 1)  # 2m for each intermediate waypoint
    step_length = (tail_state[0] - head_state[0]) / (int_wpts_num + 1)
    int_wpts = np.linspace(start_pos + step_length, target_pos, int_wpts_num, endpoint=False)
    return int_wpts


def get_int_wpts2(head_state, tail_state):
    start_pos = head_state[0]
    target_pos = tail_state[0]
    straight_length = np.linalg.norm(target_pos - start_pos)
    int_wpts_num = max(int(straight_length / 2), 1)  # 2m for each intermediate waypoint
    dim = len(start_pos)
    int_wpts = np.zeros((int_wpts_num, dim))
    for i in range(dim):
        step_length = (target_pos[i] - start_pos[i])/(int_wpts_num + 1)
        int_wpts[:, i] = np.linspace(start_pos[i] + step_length, target_pos[i], int_wpts_num, endpoint=False)

    return int_wpts


head_state = np.array([[0.0, 0, 5],
                       [0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]])
tail_state = np.array([[0.0, 10, 5],
                       [0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]])

time_start = time.time()
int_wpts = get_int_wpts(head_state, tail_state)
time_end = time.time()
print("time cost: %f" % (time_end - time_start))
print(int_wpts)

print("")
time_start = time.time()
int_wpts2 = get_int_wpts2(head_state, tail_state)
time_end = time.time()
print("time cost: %f" % (time_end - time_start))
print(int_wpts2)
