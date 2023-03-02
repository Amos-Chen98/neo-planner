'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2023-03-01 11:03:51
'''
import numpy as np
import time


def get_int_wpts(head_state, tail_state, seed):
    start_pos = head_state[0]
    target_pos = tail_state[0]
    straight_length = np.linalg.norm(target_pos - start_pos)
    int_wpts_num = max(int(straight_length/2 - 1), 1)  # 2m for each intermediate waypoint
    step_length = (tail_state[0] - head_state[0]) / (int_wpts_num + 1)
    int_wpts = np.linspace(start_pos + step_length, target_pos, int_wpts_num, endpoint=False)
    if seed == 0:
        return int_wpts
    else:
        extra = np.random.normal(0, 0.5, int_wpts.shape)
        print(extra)
        int_wpts += extra
        return int_wpts
    
head_state = np.array([[0.0, 0, 5],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0]])
tail_state = np.array([[0, 10, 5],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0]])

int_wpts = get_int_wpts(head_state, tail_state, 1)
print(int_wpts)