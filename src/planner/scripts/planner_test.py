'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2023-01-16 20:38:47
'''

import matplotlib.pyplot as plt
import numpy as np
import yaml
from traj_planner import MinJerkPlanner


class Config():
    def __init__(self):
        f = open('src/my_planner/config/planner_params.yaml', 'r')
        config = yaml.load(f, Loader=yaml.FullLoader)
        self.v_max = config['v_max']
        self.T_min = config['T_min']
        self.T_max = config['T_max']
        self.kappa = config['kappa']
        self.weights = config['weights']


if __name__ == "__main__":
    config = Config()

    # head_state = np.array([[0.0, 0],
    #                        [0, 0],
    #                        [0, 0],
    #                        [0, 0]])
    # tail_state = np.array([[100.0, 100],
    #                        [0, 0],
    #                        [0, 0],
    #                        [0, 0]])

    # wpts = np.array([[40.0, 0],
    #                  [100, 30],
    #                  [100, 50],
    #                  [20, 60],
    #                  [70, 90],
    #                  [10, 120]])

    head_state = np.array([[0.0, 0, 5],
                           [0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]])
    tail_state = np.array([[100.0, 100, 5],
                           [0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]])

    wpts = np.array([[40.0, 0, 10],
                     [100, 30, 10],
                     [100, 50, 10],
                     [20, 60, 10],
                     [70, 90, 10],
                     [10, 120, 10]])

    ts = 10 * np.ones((len(wpts)+1,))

    planner = MinJerkPlanner(head_state, tail_state, wpts, ts, config)

    planner.optimize()

    print("Times of getting cost: %d  Times of getting grad: %d" %
          (planner.get_cost_times, planner.get_grad_times))

    final_wpts = planner.int_wpts.T
    final_ts = planner.ts

    cost = planner.get_cost(np.concatenate(
        (np.reshape(planner.int_wpts, (planner.D*(planner.M - 1),)), planner.tau), axis=0))

    print("Trajectory cost: %f" % cost)

    t_samples = np.arange(0, sum(final_ts), 0.1)
    pos = planner.get_pos_array()
    vel = planner.get_vel_array()
    acc = planner.get_acc_array()
    jer = planner.get_jer_array()

    # Calculate pos, vel, acc and jer by value
    pos_value = np.zeros(pos.shape[0])
    vel_value = np.zeros(vel.shape[0])
    acc_value = np.zeros(acc.shape[0])
    jer_value = np.zeros(jer.shape[0])
    for i in range(pos.shape[0]):
        pos_value[i] = np.linalg.norm(pos[i])
        vel_value[i] = np.linalg.norm(vel[i])
        acc_value[i] = np.linalg.norm(acc[i])
        jer_value[i] = np.linalg.norm(jer[i])

    # Plot results
    plt.figure()
    plt.plot(pos[:, 0], pos[:, 1])
    plt.scatter(final_wpts[:, 0], final_wpts[:, 1])
    plt.xlabel('x-position/m')
    plt.ylabel('y-position/m')
    plt.grid()
    plt.axis('equal')

    plt.figure()
    plt.plot(t_samples, vel[:, 0], label='Vel_x')
    plt.plot(t_samples, vel[:, 1], label='Vel_y')
    plt.plot(t_samples, acc[:, 0], label='Acc_x')
    plt.plot(t_samples, acc[:, 1], label='Acc_y')
    plt.plot(t_samples, jer[:, 0], label='Jerk_x')
    plt.plot(t_samples, jer[:, 1], label='Jerk_y')
    plt.xlabel('t/s')
    plt.legend()
    plt.grid()

    plt.figure()
    plt.plot(t_samples, vel_value, label='Vel')
    plt.plot(t_samples, acc_value, label='Acc')
    plt.plot(t_samples, jer_value, label='Jerk')
    plt.xlabel('t/s')
    plt.legend()
    plt.grid()

    plt.show()
