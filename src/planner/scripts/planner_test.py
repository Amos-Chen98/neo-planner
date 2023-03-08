'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2023-03-08 10:20:52
'''

import matplotlib.pyplot as plt
import numpy as np
from traj_planner import MinJerkPlanner
import cProfile


class Config():
    def __init__(self):
        self.v_max = 5.0
        self.T_min = 1.0  # The minimum T of each piece
        self.T_max = 20.0
        self.safe_dis = 0.5  # the safe distance to the obstacle
        self.delta_t = 0.1  # the time interval of sampling
        self.weights = [1.0, 1.0, 1.0, 1.0]  # the weights of different costs: [energy, time, feasibility, collision]
        self.init_seg_len = 2.0  # the initial length of each segment
        self.init_T = 2.0  # the initial T of each segment


class FakeMap():
    def is_occuiped(self, pos):
        return False

    def get_edt_dis(self, pos):
        return 10000

    def get_edt_grad(self, pos):
        return np.array([0, 0])


if __name__ == "__main__":

    config = Config()
    map = FakeMap()
    head_state = np.array([[0.0, 0, 5],
                           [0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]])
    tail_state = np.array([[10.0, 10, 5],
                           [0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]])
    wpts = np.array([[10, 0, 10],
                    [10, 30, 15],
                    [5, 35, 10]])

    planner = MinJerkPlanner(config)
    cProfile.run('planner.plan(map, head_state, tail_state)', 'result.prof')

    # Evaluate results
    # final_wpts = planner.int_wpts.T
    # final_ts = planner.ts

    # cost = planner.get_cost(np.concatenate((np.reshape(planner.int_wpts, (planner.D*(planner.M - 1),)), planner.tau), axis=0))
    # print("Trajectory cost: %f" % cost)

    # t_samples = np.arange(0, sum(final_ts), 0.1)
    # pos = planner.get_pos_array()
    # vel = planner.get_vel_array()
    # acc = planner.get_acc_array()
    # jer = planner.get_jer_array()

    # # Calculate pos, vel, acc and jer by value
    # pos_value = np.zeros(pos.shape[0])
    # vel_value = np.zeros(vel.shape[0])
    # acc_value = np.zeros(acc.shape[0])
    # jer_value = np.zeros(jer.shape[0])
    # for i in range(pos.shape[0]):
    #     pos_value[i] = np.linalg.norm(pos[i])
    #     vel_value[i] = np.linalg.norm(vel[i])
    #     acc_value[i] = np.linalg.norm(acc[i])
    #     jer_value[i] = np.linalg.norm(jer[i])

    # # Plot results
    # plt.figure()
    # plt.plot(pos[:, 0], pos[:, 1])
    # plt.scatter(final_wpts[:, 0], final_wpts[:, 1])
    # plt.xlabel('x-position/m')
    # plt.ylabel('y-position/m')
    # plt.grid()
    # plt.axis('equal')

    # plt.figure()
    # plt.plot(t_samples, vel[:, 0], label='Vel_x')
    # plt.plot(t_samples, vel[:, 1], label='Vel_y')
    # plt.plot(t_samples, acc[:, 0], label='Acc_x')
    # plt.plot(t_samples, acc[:, 1], label='Acc_y')
    # plt.plot(t_samples, jer[:, 0], label='Jerk_x')
    # plt.plot(t_samples, jer[:, 1], label='Jerk_y')
    # plt.xlabel('t/s')
    # plt.legend()
    # plt.grid()

    # plt.figure()
    # plt.plot(t_samples, vel_value, label='Vel')
    # plt.plot(t_samples, acc_value, label='Acc')
    # plt.plot(t_samples, jer_value, label='Jerk')
    # plt.xlabel('t/s')
    # plt.legend()
    # plt.grid()

    # plt.show()
