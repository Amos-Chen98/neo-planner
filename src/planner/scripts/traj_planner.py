'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2023-01-29 16:58:50
'''
import math
import pprint
import time
import numpy as np
import scipy


class MinAccPlanner():
    def __init__(self, s, head_state, tail_state, int_wpts, ts):
        self.s = s
        self.D = head_state.shape[1]
        self.head_state = np.zeros((s, self.D))
        self.tail_state = np.zeros((s, self.D))
        for i in range(s):
            self.head_state[i] = head_state[i]
            self.tail_state[i] = tail_state[i]

        self.int_wpts = int_wpts.T  # 'int' for 'intermediate'
        self.ts = ts
        self.M = ts.shape[0]

        print("s = %d" % self.s)
        print("D = %d" % self.D)
        print("M = %d" % self.M)

        self.coeffs = []

    def get_coeffs(self, int_wpts, ts):
        '''
        Calculate coeffs according to q and T
        input: q(D,M-1) and T(M,)
        '''
        int_wpts = int_wpts.T
        T1 = ts
        T2 = ts**2
        T3 = ts**3
        A = np.zeros((2 * self.M * self.s, 2 * self.M * self.s))
        b = np.zeros((2 * self.M * self.s, self.D))
        b[0:self.s, :] = self.head_state
        b[-self.s:, :] = self.tail_state

        A[0, 0] = 1.0
        A[1, 1] = 1.0

        for i in range(self.M - 1):
            A[4 * i + 2, 4 * i] = 1.0
            A[4 * i + 2, 4 * i + 1] = T1[i]
            A[4 * i + 2, 4 * i + 2] = T2[i]
            A[4 * i + 2, 4 * i + 3] = T3[i]
            A[4 * i + 3, 4 * i] = 1.0
            A[4 * i + 3, 4 * i + 1] = T1[i]
            A[4 * i + 3, 4 * i + 2] = T2[i]
            A[4 * i + 3, 4 * i + 3] = T3[i]
            A[4 * i + 3, 4 * i + 4] = -1.0
            A[4 * i + 4, 4 * i + 1] = 1.0
            A[4 * i + 4, 4 * i + 2] = 2.0 * T1[i]
            A[4 * i + 4, 4 * i + 3] = 3.0 * T2[i]
            A[4 * i + 4, 4 * i + 5] = -1.0
            A[4 * i + 5, 4 * i + 2] = 2.0
            A[4 * i + 5, 4 * i + 3] = 6.0 * T1[i]
            A[4 * i + 5, 4 * i + 6] = -2.0

            b[4 * i + 2] = int_wpts[i]

        A[2 * self.M * self.s - 2, 2 * self.M * self.s - 4] = 1.0
        A[2 * self.M * self.s - 2, 2 * self.M * self.s - 3] = T1[-1]
        A[2 * self.M * self.s - 2, 2 * self.M * self.s - 2] = T2[-1]
        A[2 * self.M * self.s - 2, 2 * self.M * self.s - 1] = T3[-1]
        A[2 * self.M * self.s - 1, 2 * self.M * self.s - 3] = 1.0
        A[2 * self.M * self.s - 1, 2 * self.M * self.s - 2] = 2 * T1[-1]
        A[2 * self.M * self.s - 1, 2 * self.M * self.s - 1] = 3 * T2[-1]

        self.A = A
        self.coeffs = np.linalg.solve(A, b)

    def get_pos(self, t):
        '''
        get position at time t
        return a (1,D) array
        '''
        if t > sum(self.ts):
            return self.get_pos(sum(self.ts))

        if self.coeffs == []:
            self.get_coeffs(self.int_wpts, self.ts)

        # Locate piece index
        piece_idx = 0
        while sum(self.ts[:piece_idx+1]) < t:
            piece_idx += 1

        T = t - sum(self.ts[:piece_idx])

        c_block = self.coeffs[2*self.s*piece_idx:2*self.s*(piece_idx+1), :]

        beta = np.array([1, T, T**2, T**3])

        return np.dot(c_block.T, np.array([beta]).T).T

    def get_vel(self, t):
        '''
        get velocity at time t
        return a (1,D) array
        '''
        if t > sum(self.ts):
            return self.get_vel(sum(self.ts))

        if self.coeffs == []:
            self.get_coeffs(self.int_wpts, self.ts)

        # Locate piece index
        piece_idx = 0
        while sum(self.ts[:piece_idx+1]) < t:
            piece_idx += 1

        T = t - sum(self.ts[:piece_idx])

        c_block = self.coeffs[2*self.s*piece_idx:2*self.s*(piece_idx+1), :]

        beta = np.array([0, 1, 2*T, 3*T**2])

        return np.dot(c_block.T, np.array([beta]).T).T

    def get_acc(self, t):
        '''
        get acceleration at time t
        return a (1,D) array
        '''
        if t > sum(self.ts):
            return self.get_acc(sum(self.ts))

        if self.coeffs == []:
            self.get_coeffs(self.int_wpts, self.ts)

        # Locate piece index
        piece_idx = 0
        while sum(self.ts[:piece_idx+1]) < t:
            piece_idx += 1

        T = t - sum(self.ts[:piece_idx])

        c_block = self.coeffs[2*self.s*piece_idx:2*self.s*(piece_idx+1), :]

        beta = np.array([0, 0, 2, 6*T])

        return np.dot(c_block.T, np.array([beta]).T).T

    def get_jerk(self, t):
        '''
        get jerk at time t
        return a (1,D) array
        '''
        if t > sum(self.ts):
            return self.get_jerk(sum(self.ts))

        if self.coeffs == []:
            self.get_coeffs(self.int_wpts, self.ts)

        # Locate piece index
        piece_idx = 0
        while sum(self.ts[:piece_idx+1]) < t:
            piece_idx += 1

        T = t - sum(self.ts[:piece_idx])

        c_block = self.coeffs[2*self.s*piece_idx:2*self.s*(piece_idx+1), :]

        beta = np.array([0, 0, 0, 6])

        return np.dot(c_block.T, np.array([beta]).T).T


class MinJerkPlanner():
    '''
    Trajectory planner using MINCO class

    Input:
    s: the degree of trajectory to be optimized, e.g. s=4: minimum snap
    head_state: (s,D) array, where s=2(acc)/3(jerk)/4(snap), D is the dimension
    tail_state: the same as head_state
    initial waypoints: (M-1,D) array, where M is the piece num of trajectory
    initial time allocation: (M,) array

    Output:
    D-dimensional polynomial trajectory of (2s-1)-degree

    Note:
    If you want a trajecory of minimum p^(s), only the 0~(s-1) degree
    bounded contitions and valid.
    e.g. for minimum jerk(s=3) trajectory, we can specify
    pos(s=0), vel(s=1), and acc(s=2) in head_state and tail_state both.
    If you don't fill up the bounded conditions, the default will be zero.
    '''

    def __init__(self, config):
        # Mission conditions
        self.s = 3
        self.get_cost_times = 0
        self.get_grad_times = 0

        # Dynamic constraints
        self.v_max = config.v_max
        self.T_min = config.T_min
        self.T_max = config.T_max

        # Hyper params in cost func
        self.kappa = config.kappa
        self.weights = np.array(config.weights)

        # self.get_coeffs(self.int_wpts, self.ts)

    def get_coeffs(self, int_wpts, ts):
        '''
        Calculate coeffs according to int_wpts and ts
        input: int_wpts(D,M-1) and ts(M,)
        stores self.A and self.coeffs
        '''
        int_wpts = int_wpts.T
        T1 = ts
        T2 = ts**2
        T3 = ts**3
        T4 = ts**4
        T5 = ts**5

        A = np.zeros((2 * self.M * self.s, 2 * self.M * self.s))
        b = np.zeros((2 * self.M * self.s, self.D))
        b[0:self.s, :] = self.head_state
        b[-self.s:, :] = self.tail_state

        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 2.0

        for i in range(self.M - 1):
            A[6 * i + 3, 6 * i] = 1.0
            A[6 * i + 3, 6 * i + 1] = T1[i]
            A[6 * i + 3, 6 * i + 2] = T2[i]
            A[6 * i + 3, 6 * i + 3] = T3[i]
            A[6 * i + 3, 6 * i + 4] = T4[i]
            A[6 * i + 3, 6 * i + 5] = T5[i]
            A[6 * i + 4, 6 * i] = 1.0
            A[6 * i + 4, 6 * i + 1] = T1[i]
            A[6 * i + 4, 6 * i + 2] = T2[i]
            A[6 * i + 4, 6 * i + 3] = T3[i]
            A[6 * i + 4, 6 * i + 4] = T4[i]
            A[6 * i + 4, 6 * i + 5] = T5[i]
            A[6 * i + 4, 6 * i + 6] = -1.0
            A[6 * i + 5, 6 * i + 1] = 1.0
            A[6 * i + 5, 6 * i + 2] = 2 * T1[i]
            A[6 * i + 5, 6 * i + 3] = 3 * T2[i]
            A[6 * i + 5, 6 * i + 4] = 4 * T3[i]
            A[6 * i + 5, 6 * i + 5] = 5 * T4[i]
            A[6 * i + 5, 6 * i + 7] = -1.0
            A[6 * i + 6, 6 * i + 2] = 2.0
            A[6 * i + 6, 6 * i + 3] = 6 * T1[i]
            A[6 * i + 6, 6 * i + 4] = 12 * T2[i]
            A[6 * i + 6, 6 * i + 5] = 20 * T3[i]
            A[6 * i + 6, 6 * i + 8] = -2.0
            A[6 * i + 7, 6 * i + 3] = 6.0
            A[6 * i + 7, 6 * i + 4] = 24.0 * T1[i]
            A[6 * i + 7, 6 * i + 5] = 60.0 * T2[i]
            A[6 * i + 7, 6 * i + 9] = -6.0
            A[6 * i + 8, 6 * i + 4] = 24.0
            A[6 * i + 8, 6 * i + 5] = 120.0 * T1[i]
            A[6 * i + 8, 6 * i + 10] = -24.0

            b[6 * i + 3] = int_wpts[i]

        A[-3, -6] = 1.0
        A[-3, -5] = T1[-1]
        A[-3, -4] = T2[-1]
        A[-3, -3] = T3[-1]
        A[-3, -2] = T4[-1]
        A[-3, -1] = T5[-1]
        A[-2, -5] = 1.0
        A[-2, -4] = 2 * T1[-1]
        A[-2, -3] = 3 * T2[-1]
        A[-2, -2] = 4 * T3[-1]
        A[-2, -1] = 5 * T4[-1]
        A[-1, -4] = 2
        A[-1, -3] = 6 * T1[-1]
        A[-1, -2] = 12 * T2[-1]
        A[-1, -1] = 20 * T3[-1]

        self.A = A

        self.coeffs = np.linalg.solve(A, b)

    def get_pos(self, t):
        '''
        get position at time t
        return a (1,D) array
        '''
        if t > sum(self.ts):
            return self.get_pos(sum(self.ts))

        if self.coeffs == []:
            self.get_coeffs(self.int_wpts, self.ts)

        # Locate piece index
        piece_idx = 0
        while sum(self.ts[:piece_idx+1]) < t:
            piece_idx += 1

        T = t - sum(self.ts[:piece_idx])

        c_block = self.coeffs[2*self.s*piece_idx:2*self.s*(piece_idx+1), :]

        beta = np.array([1, T, T**2, T**3, T**4, T**5])

        return np.dot(c_block.T, np.array([beta]).T).T

    def get_vel(self, t):
        '''
        get velocity at time t
        return a (1,D) array
        '''
        if t > sum(self.ts):
            return self.get_vel(sum(self.ts))

        if self.coeffs == []:
            self.get_coeffs(self.int_wpts, self.ts)

        # Locate piece index
        piece_idx = 0
        while sum(self.ts[:piece_idx+1]) < t:
            piece_idx += 1

        T = t - sum(self.ts[:piece_idx])

        c_block = self.coeffs[2*self.s*piece_idx:2*self.s*(piece_idx+1), :]

        beta = np.array([0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4])

        return np.dot(c_block.T, np.array([beta]).T).T

    def get_acc(self, t):
        '''
        get acceleration at time t
        return a (1,D) array
        '''
        if t > sum(self.ts):
            return self.get_acc(sum(self.ts))

        if self.coeffs == []:
            self.get_coeffs(self.int_wpts, self.ts)

        # Locate piece index
        piece_idx = 0
        while sum(self.ts[:piece_idx+1]) < t:
            piece_idx += 1

        T = t - sum(self.ts[:piece_idx])

        c_block = self.coeffs[2*self.s*piece_idx:2*self.s*(piece_idx+1), :]

        beta = np.array([0, 0, 2, 6*T, 12*T**2, 20*T**3])

        return np.dot(c_block.T, np.array([beta]).T).T

    def get_jerk(self, t):
        '''
        get jerk at time t
        return a (1,D) array
        '''
        if t > sum(self.ts):
            return self.get_jerk(sum(self.ts))

        if self.coeffs == []:
            self.get_coeffs(self.int_wpts, self.ts)

        # Locate piece index
        piece_idx = 0
        while sum(self.ts[:piece_idx+1]) < t:
            piece_idx += 1

        T = t - sum(self.ts[:piece_idx])

        c_block = self.coeffs[2*self.s*piece_idx:2*self.s*(piece_idx+1), :]

        beta = np.array([0, 0, 0, 6, 24*T, 60*T**2])

        return np.dot(c_block.T, np.array([beta]).T).T

    def get_full_state_cmd(self, hz=300):
        if self.coeffs == []:
            self.get_coeffs(self.int_wpts, self.ts)

        total_time = sum(self.ts)
        t_samples = np.arange(0, total_time, 1/hz)
        sample_num = t_samples.shape[0]
        state_cmd = np.zeros((sample_num, 3, self.D))

        for i in range(sample_num):
            t = t_samples[i]
            state_cmd[i][0] = self.get_pos(t)
            state_cmd[i][1] = self.get_vel(t)
            state_cmd[i][2] = self.get_acc(t)

        return state_cmd, total_time, hz

    def get_pos_array(self):
        '''
        return the full pos array
        '''
        if self.coeffs == []:
            self.get_coeffs(self.int_wpts, self.ts)

        t_samples = np.arange(0, sum(self.ts), 0.1)
        pos_array = np.zeros((t_samples.shape[0], self.D))
        for i in range(t_samples.shape[0]):
            pos_array[i] = self.get_pos(t_samples[i])

        return pos_array

    def get_vel_array(self):
        '''
        return the full vel array
        '''
        if self.coeffs == []:
            self.get_coeffs(self.int_wpts, self.ts)

        t_samples = np.arange(0, sum(self.ts), 0.1)
        vel_array = np.zeros((t_samples.shape[0], self.D))
        for i in range(t_samples.shape[0]):
            vel_array[i] = self.get_vel(t_samples[i])

        return vel_array

    def get_acc_array(self):
        '''
        return the full acc array
        '''
        if self.coeffs == []:
            self.get_coeffs(self.int_wpts, self.ts)

        t_samples = np.arange(0, sum(self.ts), 0.1)
        acc_array = np.zeros((t_samples.shape[0], self.D))
        for i in range(t_samples.shape[0]):
            acc_array[i] = self.get_acc(t_samples[i])

        return acc_array

    def get_jer_array(self):
        '''
        return the full jer array
        '''
        if self.coeffs == []:
            self.get_coeffs(self.int_wpts, self.ts)

        t_samples = np.arange(0, sum(self.ts), 0.1)
        jer_array = np.zeros((t_samples.shape[0], self.D))
        for i in range(t_samples.shape[0]):
            jer_array[i] = self.get_jerk(t_samples[i])

        return jer_array

    def reset_cost(self):
        self.costs = np.zeros(3)

    def reset_grad_CT(self):
        self.grad_C = np.zeros((2 * self.M * self.s, self.D))
        self.grad_T = np.zeros(self.M)

    def add_energy_cost(self):
        '''
        get energy cost according to self.coeffs and self.ts
        refer to eq (14) in distributed...
        '''
        for i in range(self.M):
            c = self.coeffs[2*self.s*i:2*self.s*(i+1), :]
            T = self.ts[i]
            beta3_mat = np.array([[0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 36*T, 72*T**2, 120*T**3],
                                  [0, 0, 0, 72*T**2, 192*T**3, 360*T**4],
                                  [0, 0, 0, 120*T**3, 360*T**4, 720*T**5]])
            self.costs[0] += np.trace(c.T @ beta3_mat @ c)

    def add_energy_grad_CT(self):
        '''
        get energy grad according to self.coeffs and self.ts
        stores self.grad_C and self.grad_T
        refer to eq (13,14) in Decentralized...
        '''
        for i in range(self.M):
            c = self.coeffs[2*self.s*i:2*self.s*(i+1), :]
            T = self.ts[i]
            beta3 = np.array([[0, 0, 0, 6, 24*T, 60*T**2]]).T  # shape:(6,1)
            beta3_mat = np.array([[0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 36*T, 72*T**2, 120*T**3],
                                  [0, 0, 0, 72*T**2, 192*T**3, 360*T**4],
                                  [0, 0, 0, 120*T**3, 360*T**4, 720*T**5]])
            # self.costs[0] += self.weights[0] * np.trace(c.T @ beta3_mat @ c)

            self.grad_C[2*self.s*i: 2*self.s *
                        (i+1), :] += self.weights[0] * 2 * beta3_mat @ c

            for j in range(self.D):
                c_current_dim = c[:, j]  # (6,) vector
                self.grad_T[i] += self.weights[0] * \
                    (np.dot(c_current_dim, beta3).item())**2
                # the above 3 lines are equal to this following line
                # self.grad_T[i] = c.T @ beta3 @ beta3.T @ c

    def add_time_cost(self):
        self.costs[1] += np.sum(self.ts)

    def add_time_grad_CT(self):
        self.grad_T += self.weights[1] * np.ones(self.M)

    def add_feasibility_cost(self):
        '''
        get dynamic feasibility cost according to self.coeffs and self.ts
        refer to eq (6) in Decentralized...
        '''

        for i in range(self.M):  # for every piece
            t = 0.0
            step = self.ts[i] / self.kappa
            c = self.coeffs[2*self.s*i:2*self.s*(i+1), :]

            for j in range(self.kappa):  # for every time slot within the i-th piece
                beta = self.get_beta_s3(t)
                pos = np.dot(c.T, beta[0])  # return a (2,) array
                vel = np.dot(c.T, beta[1])
                acc = np.dot(c.T, beta[2])
                jer = np.dot(c.T, beta[3])
                omg = 0.5 if j in [0, self.kappa-1] else 1

                violate_vel = sum(vel**2) - self.v_max**2

                if violate_vel > 0.0:
                    self.costs[2] += omg*step*violate_vel**3

                t += step

    def add_feasibility_grad_CT(self):
        '''
        get dynamic feasibility grad according to self.coeffs and self.ts
        stores self.grad_C and self.grad_T
        refer to eq (6-16) in Decentralized...
        '''
        for i in range(self.M):  # for every piece
            t = 0.0
            step = self.ts[i] / self.kappa
            c = self.coeffs[2*self.s*i:2*self.s*(i+1), :]

            for j in range(self.kappa):  # for every time slot within the i-th piece
                beta = self.get_beta_s3(t)
                pos = np.dot(c.T, beta[0])  # return a (2,) array
                vel = np.dot(c.T, beta[1])
                acc = np.dot(c.T, beta[2])
                jer = np.dot(c.T, beta[3])
                omg = 0.5 if j in [0, self.kappa-1] else 1

                violate_vel = sum(vel**2) - self.v_max**2
                # print("violate_vel:%f" % violate_vel)

                if violate_vel > 0.0:
                    # self.costs[2] += self.weights[2] * omg*step*violate_vel**3

                    grad_v2c = 2 * \
                        np.dot(np.array([beta[1]]).T,
                               np.array([vel]))  # col * row
                    grad_v2t = 2*(np.array([beta[2]])
                                  @ c @ np.array([vel]).T).item()
                    grad_K2v = 3 * step * omg * violate_vel**2

                    self.grad_C[2*self.s*i: 2*self.s *
                                (i+1), :] += self.weights[2] * grad_K2v * grad_v2c

                    self.grad_T[i] += self.weights[2] * (omg*violate_vel**3/self.kappa +
                                                         grad_K2v * grad_v2t * j/self.kappa)

                t += step

    def get_beta_s3(self, t):
        '''
        used in func 'add_feasibility_grad_CT'
        '''
        beta = np.array([[1, t, t**2, t**3, t**4, t**5],
                         [0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4],
                         [0, 0, 2, 6*t, 12*t**2, 20*t**3],
                         [0, 0, 0, 6, 24*t, 60*t**2]])
        return beta

    def map_T2tau(self, ts):
        '''
        inverse sigmoid func
        '''
        tau = np.zeros(self.M)
        for i in range(self.M):
            tau[i] = -math.log((self.T_max - self.T_min) /
                               (ts[i]-self.T_min) - 1)
        return tau

    def map_tau2T(self, tau):
        ts = np.zeros(self.M)
        for i in range(self.M):
            ts[i] = (self.T_max - self.T_min) / \
                (1 + math.exp(-tau[i])) + self.T_min
        return ts

    def get_grad_T2tau(self, grad_T):
        grad_tau = np.zeros(self.M)
        for i in range(self.M):
            grad_tau[i] = grad_T[i] * (self.T_max - self.T_min) * \
                math.exp(-self.tau[i])/(1+math.exp(-self.tau[i]))**2
        return grad_tau

    def propagate_grad_q_tau(self):
        '''
        get grad W2q (grad_q) and W2tau (grad_tau) from grad_C and grad_T
        return
        grad_q: matrix of (self.D, self.M - 1)
        grad_tau: vector of (self.M,)
        '''
        grad_q = np.zeros((self.D, self.M - 1))
        grad_T = np.zeros(self.M)
        G = np.linalg.solve(self.A.T, self.grad_C)

        # Calculate grad_q, refer to eq(3-89) in Wang's thesis
        for i in range(self.M - 1):
            # Indices of G should be consistent with b's row configuration
            grad_q[:, i] = G[6*i + 3, :].T

        # Calculate grad_tau, refer to eq(3-93) in Wang's thesis
        for i in range(self.M - 1):
            '''
            Note that the i in Ti starts from 1 and ends at M
            so here T[0] actually means T1
            '''
            T = self.ts[i]
            Gi_T = G[2*i*self.s+self.s: 2*(i+1)*self.s+self.s, :].T
            grad_E2T = np.array([[0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],
                                 [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],
                                 [0, 0, 2, 6*T, 12*T**2, 20*T**3],
                                 [0, 0, 0, 6, 24*T, 60*T**2],
                                 [0, 0, 0, 0, 24, 120*T],
                                 [0, 0, 0, 0, 0, 120]])
            ci = self.coeffs[2*i*self.s: 2*(i+1)*self.s, :]
            grad_T[i] = self.grad_T[i] - np.trace(Gi_T @ grad_E2T @ ci)

        # for E_M specially
        Gi_T = G[-self.s:, :].T
        grad_E2T = np.array([[0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],
                             [0, 0, 2, 6*T, 12*T**2, 20*T**3],
                             [0, 0, 0, 6, 24*T, 60*T**2]])
        ci = self.coeffs[-2*self.s:, :]
        grad_T[-1] = self.grad_T[-1] - np.trace(Gi_T @ grad_E2T @ ci)

        grad_tau = self.get_grad_T2tau(grad_T)  # for T = exp(tau)

        return grad_q, grad_tau

    def get_cost(self, x):
        self.get_cost_times += 1
        self.int_wpts = np.reshape(
            x[:self.D*(self.M - 1)], (self.D, self.M - 1))
        self.tau = x[self.D*(self.M - 1):]
        self.ts = self.map_tau2T(self.tau)

        # print("-----------------Current ts-----------------")
        # print(self.ts)

        self.get_coeffs(self.int_wpts, self.ts)  # get A and coeffs by input
        self.reset_cost()
        self.add_energy_cost()
        self.add_time_cost()
        self.add_feasibility_cost()

        # print("-----------Current weighted cost------------")
        # print("Energy cost: %f, Time cost: %f, Feasibility cost: %f" %
        #       (self.costs[0], self.costs[1], self.costs[2]))
        # print("\n")
        return np.dot(self.costs, self.weights)

    def get_grad(self, x):
        '''
        get the gradient of cost func to x
        1. Retrieve q and tau from x
        2. Get grad_q and grad_tau, size: grad_q(D,M-1), grad_tau(M,)
        3. Flatten and concatenate grad_q and grad_tau to return a vector        
        '''
        self.get_grad_times += 1
        self.int_wpts = np.reshape(
            x[:self.D*(self.M - 1)], (self.D, self.M - 1))
        self.tau = x[self.D*(self.M - 1):]
        self.ts = self.map_tau2T(self.tau)

        self.get_coeffs(self.int_wpts, self.ts)  # get A and coeffs by input
        self.reset_grad_CT()  # init self.cost, self.grad_c, and self.grad_T as zero
        self.add_energy_grad_CT()
        self.add_time_grad_CT()
        self.add_feasibility_grad_CT()
        grad_q, grad_tau = self.propagate_grad_q_tau()  # prop grad to q and tau

        grad = np.concatenate(
            (np.reshape(grad_q, (self.D * (self.M - 1),)), grad_tau), axis=0)

        # print("----------------Current grad----------------")
        # print(grad)
        # print("\n")

        return grad

    def get_int_wpts(self, head_state, tail_state, int_wpts_num=3):
        start_pos = head_state[0]
        target_pos = tail_state[0]
        dim = len(start_pos)
        int_wpts = np.zeros((int_wpts_num, dim))
        for i in range(dim):
            step_length = (target_pos[i] - start_pos[i])/(int_wpts_num + 1)
            int_wpts[:, i] = np.linspace(start_pos[i] + step_length, target_pos[i], int_wpts_num, endpoint=False)

        return int_wpts

    def plan(self, head_state, tail_state):
        int_wpts = self.get_int_wpts(head_state, tail_state)
        ts = 10 * np.ones((len(int_wpts)+1,))

        self.D = head_state.shape[1]
        self.M = ts.shape[0]
        self.head_state = np.zeros((self.s, self.D))
        self.tail_state = np.zeros((self.s, self.D))
        for i in range(self.s):
            self.head_state[i] = head_state[i]
            self.tail_state[i] = tail_state[i]
        self.int_wpts = int_wpts.T  # 'int' for 'intermediate'
        self.ts = ts
        self.tau = self.map_T2tau(ts)  # agent for ts

        x0 = np.concatenate(
            (np.reshape(self.int_wpts, (self.D*(self.M - 1),)), self.tau), axis=0)

        time_start = time.time()
        res = scipy.optimize.minimize(self.get_cost,
                                      x0,
                                      method='L-BFGS-B',
                                      jac=self.get_grad,
                                      bounds=None,
                                      tol=1e-8,
                                      callback=None,
                                      options={'disp': 0,
                                               'maxcor': 10,
                                               'maxfun': 15000,
                                               'maxiter': 15000,
                                               'iprint': 1,
                                               'maxls': 20})
        time_end = time.time()

        self.int_wpts = np.reshape(res.x[:self.D*(self.M - 1)], (self.D, self.M - 1))
        self.tau = res.x[self.D*(self.M - 1):]
        self.ts = self.map_tau2T(self.tau)

        self.get_coeffs(self.int_wpts, self.ts)

        print("-----------------------Final intermediate waypoints-----------------------")
        pprint.pprint(self.int_wpts.T)
        print("-----------------------Final T--------------------------------------------")
        pprint.pprint(self.ts)

        print("Otimization running time: %f" % (time_end - time_start))


class MinSnapPlanner():

    def __init__(self, s, head_state, tail_state, int_wpts, ts):
        self.s = s
        self.D = head_state.shape[1]
        self.head_state = np.zeros((s, self.D))
        self.tail_state = np.zeros((s, self.D))
        for i in range(s):
            self.head_state[i] = head_state[i]
            self.tail_state[i] = tail_state[i]

        self.int_wpts = int_wpts.T  # 'int' for 'intermediate'
        self.ts = ts
        self.M = ts.shape[0]

        print("s = %d" % self.s)
        print("D = %d" % self.D)
        print("M = %d" % self.M)

        self.coeffs = []

    def get_coeffs(self, int_wpts, ts):
        '''
        Calculate coeffs according to q and T
        input: q(D,M-1) and T(M,)
        '''
        int_wpts = int_wpts.T
        T1 = ts
        T2 = ts**2
        T3 = ts**3
        T4 = ts**4
        T5 = ts**5
        T6 = ts**6
        T7 = ts**7
        A = np.zeros((2 * self.M * self.s, 2 * self.M * self.s))
        b = np.zeros((2 * self.M * self.s, self.D))
        b[0:self.s, :] = self.head_state
        b[-self.s:, :] = self.tail_state

        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 2.0
        A[3, 3] = 6.0

        for i in range(self.M - 1):
            A[8 * i + 4, 8 * i + 4] = 24.0
            A[8 * i + 4, 8 * i + 5] = 120.0 * T1[i]
            A[8 * i + 4, 8 * i + 6] = 360.0 * T2[i]
            A[8 * i + 4, 8 * i + 7] = 840.0 * T3[i]
            A[8 * i + 4, 8 * i + 12] = -24.0
            A[8 * i + 5, 8 * i + 5] = 120.0
            A[8 * i + 5, 8 * i + 6] = 720.0 * T1[i]
            A[8 * i + 5, 8 * i + 7] = 2520.0 * T2[i]
            A[8 * i + 5, 8 * i + 13] = -120.0
            A[8 * i + 6, 8 * i + 6] = 720.0
            A[8 * i + 6, 8 * i + 7] = 5040.0 * T1[i]
            A[8 * i + 6, 8 * i + 14] = -720.0
            A[8 * i + 7, 8 * i] = 1.0
            A[8 * i + 7, 8 * i + 1] = T1[i]
            A[8 * i + 7, 8 * i + 2] = T2[i]
            A[8 * i + 7, 8 * i + 3] = T3[i]
            A[8 * i + 7, 8 * i + 4] = T4[i]
            A[8 * i + 7, 8 * i + 5] = T5[i]
            A[8 * i + 7, 8 * i + 6] = T6[i]
            A[8 * i + 7, 8 * i + 7] = T7[i]
            A[8 * i + 8, 8 * i] = 1.0
            A[8 * i + 8, 8 * i + 1] = T1[i]
            A[8 * i + 8, 8 * i + 2] = T2[i]
            A[8 * i + 8, 8 * i + 3] = T3[i]
            A[8 * i + 8, 8 * i + 4] = T4[i]
            A[8 * i + 8, 8 * i + 5] = T5[i]
            A[8 * i + 8, 8 * i + 6] = T6[i]
            A[8 * i + 8, 8 * i + 7] = T7[i]
            A[8 * i + 8, 8 * i + 8] = -1.0
            A[8 * i + 9, 8 * i + 1] = 1.0
            A[8 * i + 9, 8 * i + 2] = 2.0 * T1[i]
            A[8 * i + 9, 8 * i + 3] = 3.0 * T2[i]
            A[8 * i + 9, 8 * i + 4] = 4.0 * T3[i]
            A[8 * i + 9, 8 * i + 5] = 5.0 * T4[i]
            A[8 * i + 9, 8 * i + 6] = 6.0 * T5[i]
            A[8 * i + 9, 8 * i + 7] = 7.0 * T6[i]
            A[8 * i + 9, 8 * i + 9] = -1.0
            A[8 * i + 10, 8 * i + 2] = 2.0
            A[8 * i + 10, 8 * i + 3] = 6.0 * T1[i]
            A[8 * i + 10, 8 * i + 4] = 12.0 * T2[i]
            A[8 * i + 10, 8 * i + 5] = 20.0 * T3[i]
            A[8 * i + 10, 8 * i + 6] = 30.0 * T4[i]
            A[8 * i + 10, 8 * i + 7] = 42.0 * T5[i]
            A[8 * i + 10, 8 * i + 10] = -2.0
            A[8 * i + 11, 8 * i + 3] = 6.0
            A[8 * i + 11, 8 * i + 4] = 24.0 * T1[i]
            A[8 * i + 11, 8 * i + 5] = 60.0 * T2[i]
            A[8 * i + 11, 8 * i + 6] = 120.0 * T3[i]
            A[8 * i + 11, 8 * i + 7] = 210.0 * T4[i]
            A[8 * i + 11, 8 * i + 11] = -6.0

            b[8 * i + 7] = int_wpts[i]

        A[8 * self.M - 4, 8 * self.M - 8] = 1.0
        A[8 * self.M - 4, 8 * self.M - 7] = T1[-1]
        A[8 * self.M - 4, 8 * self.M - 6] = T2[-1]
        A[8 * self.M - 4, 8 * self.M - 5] = T3[-1]
        A[8 * self.M - 4, 8 * self.M - 4] = T4[-1]
        A[8 * self.M - 4, 8 * self.M - 3] = T5[-1]
        A[8 * self.M - 4, 8 * self.M - 2] = T6[-1]
        A[8 * self.M - 4, 8 * self.M - 1] = T7[-1]
        A[8 * self.M - 3, 8 * self.M - 7] = 1.0
        A[8 * self.M - 3, 8 * self.M - 6] = 2.0 * T1[-1]
        A[8 * self.M - 3, 8 * self.M - 5] = 3.0 * T2[-1]
        A[8 * self.M - 3, 8 * self.M - 4] = 4.0 * T3[-1]
        A[8 * self.M - 3, 8 * self.M - 3] = 5.0 * T4[-1]
        A[8 * self.M - 3, 8 * self.M - 2] = 6.0 * T5[-1]
        A[8 * self.M - 3, 8 * self.M - 1] = 7.0 * T6[-1]
        A[8 * self.M - 2, 8 * self.M - 6] = 2.0
        A[8 * self.M - 2, 8 * self.M - 5] = 6.0 * T1[-1]
        A[8 * self.M - 2, 8 * self.M - 4] = 12.0 * T2[-1]
        A[8 * self.M - 2, 8 * self.M - 3] = 20.0 * T3[-1]
        A[8 * self.M - 2, 8 * self.M - 2] = 30.0 * T4[-1]
        A[8 * self.M - 2, 8 * self.M - 1] = 42.0 * T5[-1]
        A[8 * self.M - 1, 8 * self.M - 5] = 6.0
        A[8 * self.M - 1, 8 * self.M - 4] = 24.0 * T1[-1]
        A[8 * self.M - 1, 8 * self.M - 3] = 60.0 * T2[-1]
        A[8 * self.M - 1, 8 * self.M - 2] = 120.0 * T3[-1]
        A[8 * self.M - 1, 8 * self.M - 1] = 210.0 * T4[-1]

        self.A = A
        self.coeffs = np.linalg.solve(A, b)

    def get_pos(self, t):
        '''
        get position at time t
        return a (1,D) array
        '''
        if t > sum(self.ts):
            return self.get_pos(sum(self.ts))

        if self.coeffs == []:
            self.get_coeffs(self.int_wpts, self.ts)

        # Locate piece index
        piece_idx = 0
        while sum(self.ts[:piece_idx+1]) < t:
            piece_idx += 1

        T = t - sum(self.ts[:piece_idx])

        c_block = self.coeffs[2*self.s*piece_idx:2*self.s*(piece_idx+1), :]

        beta = np.array([1, T, T**2, T**3, T**4, T**5, T**6, T**7])

        return np.dot(c_block.T, np.array([beta]).T).T

    def get_vel(self, t):
        '''
        get velocity at time t
        return a (1,D) array
        '''
        if t > sum(self.ts):
            return self.get_vel(sum(self.ts))

        if self.coeffs == []:
            self.get_coeffs(self.int_wpts, self.ts)

        # Locate piece index
        piece_idx = 0
        while sum(self.ts[:piece_idx+1]) < t:
            piece_idx += 1

        T = t - sum(self.ts[:piece_idx])

        c_block = self.coeffs[2*self.s*piece_idx:2*self.s*(piece_idx+1), :]

        beta = np.array(
            [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4, 6*T**5, 7*T**6])

        return np.dot(c_block.T, np.array([beta]).T).T

    def get_acc(self, t):
        '''
        get acceleration at time t
        return a (1,D) array
        '''
        if t > sum(self.ts):
            return self.get_acc(sum(self.ts))

        if self.coeffs == []:
            self.get_coeffs(self.int_wpts, self.ts)

        # Locate piece index
        piece_idx = 0
        while sum(self.ts[:piece_idx+1]) < t:
            piece_idx += 1

        T = t - sum(self.ts[:piece_idx])

        c_block = self.coeffs[2*self.s*piece_idx:2*self.s*(piece_idx+1), :]

        beta = np.array(
            [0, 0, 2, 6*T, 12*T**2, 20*T**3, 30*T**4, 42*T**5])

        return np.dot(c_block.T, np.array([beta]).T).T

    def get_jerk(self, t):
        '''
        get jerk at time t
        return a (1,D) array
        '''
        if t > sum(self.ts):
            return self.get_jerk(sum(self.ts))

        if self.coeffs == []:
            self.get_coeffs(self.int_wpts, self.ts)

        # Locate piece index
        piece_idx = 0
        while sum(self.ts[:piece_idx+1]) < t:
            piece_idx += 1

        T = t - sum(self.ts[:piece_idx])

        c_block = self.coeffs[2*self.s*piece_idx:2*self.s*(piece_idx+1), :]

        beta = np.array(
            [0, 0, 0, 6, 24*T, 60*T**2, 120*T**3, 210*T**4])

        return np.dot(c_block.T, np.array([beta]).T).T
