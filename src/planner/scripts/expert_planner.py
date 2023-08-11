'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2023-07-30 21:36:55
'''
import math
import pprint
import numpy as np
import scipy.optimize as opt
import math
from traj_utils import TrajUtils


class DefaultConfig():
    def __init__(self):
        self.v_max = 10.0
        self.T_min = 2.0  # The minimum T of each piece
        self.T_max = 20.0
        self.safe_dis = 0.5  # the safe distance to the obstacle
        self.delta_t = 0.1  # the time interval of sampling
        self.weights = [1.0, 1.0, 0.001, 10000]  # the weights of different costs: [energy, time, feasibility, collision]
        self.init_wpts_mode = 'fixed'  # options: 'fixed', 'adaptive'
        self.init_seg_len = 2.0  # (valid when init_wpts_mode = 'adaptive') the initial length of each segment
        self.init_wpts_num = 2  # (valid when init_wpts_mode = 'fixed') the number of intermediate waypoints
        self.init_T = 2.0  # the initial T of each segment
        self.collision_cost_tol = 10  # the tolerance of collision cost


class MinJerkPlanner(TrajUtils):
    '''
    Minimum-jerk trajectory planner using MINCO class
    '''

    def __init__(self, config=DefaultConfig()):
        super().__init__()
        # Mission conditions
        self.s = 3

        # Dynamic constraints
        self.v_max = config.v_max
        self.T_min = config.T_min
        self.T_max = config.T_max
        self.safe_dis = config.safe_dis
        self.collision_cost_tol = config.collision_cost_tol

        # Hyper params in cost func
        self.weights = np.array(config.weights)
        self.delta_t = config.delta_t
        self.get_beta_full()

        # Initial conditions
        self.init_wpts_mode = config.init_wpts_mode
        self.init_seg_len = config.init_seg_len
        self.init_wpts_num = int(config.init_wpts_num)
        self.init_T = config.init_T
        self.batch_num = 3  # this is used in batch_generate_init_variables()

        # counters
        self.iter_num = 0
        self.opt_running_times = 0

    def plan(self, map, head_state, tail_state):
        '''
        Input:
        map: map object
        head_state: (s,D) array, where s=2(acc)/3(jerk)/4(snap), D is the dimension
        tail_state: the same as head_state
            Note:
            If you want a trajecory of minimum s, only the 0~(s-1) degree bounded contitions and valid.
            e.g. for minimum jerk(s=3) trajectory, we can specify
            pos(s=0), vel(s=1), and acc(s=2) in head_state and tail_state both.
            If you don't fill up the bounded conditions, the default will be zero.

        Stores the results in self.int_wpts, self.ts, self.tau, self.coeffs
        '''
        # generate init variables
        int_wpts, ts = self.generate_init_variables(head_state, tail_state)

        # plan from init variables
        self.warm_start_plan(map, head_state, tail_state, int_wpts, ts)

    def generate_init_variables(self, head_state, tail_state, seed=0):
        # generate init waypoints
        start_pos = head_state[0]
        target_pos = tail_state[0]
        straight_length = np.linalg.norm(target_pos - start_pos)
        if self.init_wpts_mode == 'adaptive':
            int_wpts_num = max(math.ceil(straight_length/self.init_seg_len - 1), 1)  # 2m for each intermediate waypoint
        elif self.init_wpts_mode == 'fixed':
            int_wpts_num = self.init_wpts_num
        step_length = (target_pos - start_pos) / (int_wpts_num + 1)
        int_wpts = np.linspace(start_pos + step_length, target_pos, int_wpts_num, endpoint=False)
        if seed != 0:
            int_wpts += np.random.normal(0, 0.5, int_wpts.shape)

        # generate init ts
        ts = self.init_T * np.ones((int_wpts_num+1,))  # allocate time for each piece
        ts[0] *= 1.5
        ts[-1] *= 1.5

        return int_wpts.T, ts  # make int_wpts (D,M-1) array

    def batch_generate_init_variables(self, head_state, tail_state):
        '''
        sample 'self.batch_num' batches of init wpts
        this is only valid when init_wpts_mode = 'fixed'
        '''
        if self.init_wpts_mode != 'fixed':
            print("Error! init_wpts_mode must be 'fixed'")

        start_pos = head_state[0]
        target_pos = tail_state[0]
        longitu_dir = (target_pos - start_pos)/np.linalg.norm(target_pos - start_pos)
        lateral_dir = np.array([[longitu_dir[1], -longitu_dir[0]],
                                [-longitu_dir[1], longitu_dir[0]]])

        batch_init_wpts = np.zeros((self.batch_num,  self.init_wpts_num, head_state.shape[1]))

        # first candidate - straight line
        step_length = (target_pos - start_pos) / (self.init_wpts_num + 1)
        int_wpts = np.linspace(start_pos + step_length, target_pos, self.init_wpts_num, endpoint=False)
        batch_init_wpts[0] = int_wpts  # the first candidate is the straight line

        lateral_dir_flag = 0
        lateral_move_dis = 0.6

        # other candidates
        for i in range(1, self.batch_num):
            batch_init_wpts[i] = batch_init_wpts[0] + lateral_move_dis * lateral_dir[lateral_dir_flag]
            lateral_dir_flag = 1 - lateral_dir_flag

        # generate init ts
        ts = self.init_T * np.ones((self.init_wpts_num+1,))  # allocate time for each piece
        ts[0] *= 1.5
        ts[-1] *= 1.5

        # revert the last two dimensions of batch_init_wpts, to make every sample a (D,M-1) array
        batch_init_wpts = np.transpose(batch_init_wpts, (0, 2, 1))

        return batch_init_wpts, ts

    def batch_plan(self, map, head_state, tail_state):
        batch_init_wpts, ts = self.batch_generate_init_variables(head_state, tail_state)
        batch_opt_wpts = np.zeros(batch_init_wpts.shape)  # col major
        batch_opt_ts = np.zeros((self.batch_num, len(ts)))
        batch_cost = np.zeros(self.batch_num)

        for i in range(self.batch_num):
            try:
                self.read_planning_conditions(map, head_state, tail_state, batch_init_wpts[i], ts)
                self.plan_once()
                batch_opt_wpts[i] = self.int_wpts
                batch_opt_ts[i] = self.ts
                batch_cost[i] = self.weighted_cost.sum()
                print(f"batch_cost[{i}] = {batch_cost[i]}")
            except Exception as ex:
                print(f"The {i}th attempt is deprecated for {ex}")
                batch_cost[i] = np.inf

            # check if there is one batch that is feasible
            if np.min(batch_cost) < np.inf:
                best_idx = np.argmin(batch_cost)
                self.int_wpts = batch_opt_wpts[best_idx]
                self.ts = batch_opt_ts[best_idx]
                self.final_cost = batch_cost[best_idx]
            else:
                print("All attempts are infeasible! Start re-planning from scratch...")
                self.warm_start_plan(map, head_state, tail_state, batch_init_wpts[0], ts)

    def read_planning_conditions(self, map, head_state, tail_state, int_wpts, ts):
        self.map = map
        self.D = head_state.shape[1]
        self.M = ts.shape[0]

        self.head_state = np.zeros((self.s, self.D))
        self.tail_state = np.zeros((self.s, self.D))

        for i in range(min(self.s, head_state.shape[0])):
            self.head_state[i] = head_state[i]
        for i in range(min(self.s, tail_state.shape[0])):
            self.tail_state[i] = tail_state[i]

        self.int_wpts = int_wpts
        self.ts = ts

    def warm_start_plan(self, map, head_state, tail_state, int_wpts, ts):
        # this function is only executed once

        self.read_planning_conditions(map, head_state, tail_state, int_wpts, ts)

        seed = 0

        while seed < 5:
            try:
                self.plan_once()
                return
            except Exception as ex:
                print(f"Re-planning for {ex}, current seed: {seed}")
                seed += 1
                self.int_wpts, self.ts = self.generate_init_variables(head_state, tail_state, seed)

        # raise an Exception if the planning is still infeasible after 5 attempts
        raise Exception("No solution for the given target")

    def plan_once(self):
        '''
        Plan once using current wtps and ts (tau) settings
        '''
        self.tau = self.map_T2tau(self.ts)  # agent for ts

        x0 = np.concatenate((np.reshape(self.int_wpts, (self.D*(self.M - 1),)), self.tau), axis=0)

        res = opt.minimize(self.get_cost,
                                      x0,
                                      method='L-BFGS-B',
                                      jac=self.get_grad,
                                      bounds=None,
                                      tol=1e-4,
                                      callback=None,
                                      options={'disp': 0,
                                               'maxcor': 10,
                                               'maxfun': 15000,
                                               'maxiter': 15000,
                                               'iprint': 0,
                                               'maxls': 20})

        self.int_wpts = np.reshape(res.x[:self.D*(self.M - 1)], (self.D, self.M - 1))
        self.tau = res.x[self.D*(self.M - 1):]
        self.ts = self.map_tau2T(self.tau)
        self.iter_num += res.nit
        # print("iter_num: %d" % res.nit)
        self.opt_running_times += 1
        self.weighted_cost = self.costs * self.weights
        self.final_cost = self.weighted_cost.sum()
        collision_cost = self.weighted_cost[3]
        if collision_cost > self.collision_cost_tol:
            raise ValueError("collision cost too large")

    def print_results(self):
        print("-----------------------Final intermediate waypoints-----------------------")
        pprint.pprint(self.int_wpts.T)
        print("-----------------------Final T--------------------------------------------")
        pprint.pprint(self.ts)

        print("-----------------------Weighted cost--------------------------------------")
        self.weighted_cost = self.costs * self.weights
        print("Energy cost: %f, Time cost: %f, Feasibility cost: %f, Collision cost: %f" %
              (self.weighted_cost[0], self.weighted_cost[1], self.weighted_cost[2], self.weighted_cost[3]))

    def get_beta_full(self):
        t_array = np.arange(0, self.T_max, self.delta_t)
        beta_full = np.zeros((len(t_array), 4, 6))
        for i in range(len(t_array)):
            t = t_array[i]
            beta_full[i] = np.array([[1, t, t**2, t**3, t**4, t**5],
                                     [0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4],
                                     [0, 0, 2, 6*t, 12*t**2, 20*t**3],
                                     [0, 0, 0, 6, 24*t, 60*t**2]])
        self.beta_full = beta_full

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
        A[-1, -4] = 2.0
        A[-1, -3] = 6 * T1[-1]
        A[-1, -2] = 12 * T2[-1]
        A[-1, -1] = 20 * T3[-1]

        self.A = A

        self.coeffs = np.linalg.solve(A, b)

    def reset_cost(self):
        self.costs = np.zeros(len(self.weights))

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

            self.grad_C[2*self.s*i: 2*self.s*(i+1), :] += self.weights[0] * 2 * beta3_mat @ c

            for j in range(self.D):
                c_current_dim = c[:, j]  # (6,) vector
                self.grad_T[i] += self.weights[0] * (np.dot(c_current_dim, beta3).item())**2
                # the above 3 lines are equal to this following line?
                # self.grad_T[i] = c.T @ beta3 @ beta3.T @ c

    def add_time_cost(self):
        self.costs[1] += np.sum(self.ts)

    def add_time_grad_CT(self):
        self.grad_T += self.weights[1] * np.ones(self.M)

    def add_sampled_cost(self):
        '''
        get dynamic feasibility cost and collision cost according to self.coeffs and self.ts
        feasibility: refer to eq (6) in Decentralized...
        collision: refer to eq (17)-(18) in distributed...
        '''

        for i in range(self.M):  # for every piece
            c = self.coeffs[2*self.s*i:2*self.s*(i+1), :]
            sample_num = int(self.ts[i]/self.delta_t)

            for j in range(sample_num):  # for every time slot within the i-th piece
                beta = self.beta_full[j]
                pos = np.dot(c.T, beta[0])
                vel = np.dot(c.T, beta[1])
                omg = 0.5 if j in [0, sample_num-1] else 1

                # Feasibility
                violate_vel = sum(vel**2) - self.v_max**2

                if violate_vel > 0.0:
                    self.costs[2] += omg*self.delta_t*violate_vel**3

                # Collision
                pos_projected = pos[:2]
                edt_dis = self.map.get_edt_dis(pos_projected)
                violate_dis = self.safe_dis - edt_dis

                if violate_dis > 0.0:
                    # print("Collision at pos: ", pos_projected)
                    self.costs[3] += omg*self.delta_t*violate_dis**3

    def add_sampled_grad_CT(self):
        '''
        get dynamic feasibility grad and collision grad according to self.coeffs and self.ts
        stores self.grad_C and self.grad_T
        feasibility: refer to eq (6-16) in Decentralized...
        collision: refer to eq (17)-(22) in Decentralized...
        '''
        for i in range(self.M):  # for every piece
            c = self.coeffs[2*self.s*i:2*self.s*(i+1), :]
            sample_num = int(self.ts[i]/self.delta_t)

            for j in range(sample_num):  # for every time slot within the i-th piece
                beta = self.beta_full[j]
                pos = np.dot(c.T, beta[0])  # return a (2,) array
                vel = np.dot(c.T, beta[1])
                omg = 0.5 if j in [0, sample_num-1] else 1

                # Feasibility
                violate_vel = sum(vel**2) - self.v_max**2
                # print("violate_vel:%f" % violate_vel)

                if violate_vel > 0.0:
                    grad_v2c = 2 * np.dot(np.array([beta[1]]).T, np.array([vel]))  # col * row
                    grad_v2t = 2 * (np.array([beta[2]]) @ c @ np.array([vel]).T).item()
                    grad_K2v = 3 * self.delta_t * omg * violate_vel**2

                    self.grad_C[2*self.s*i: 2*self.s * (i+1), :] += self.weights[2] * grad_K2v * grad_v2c
                    self.grad_T[i] += self.weights[2] * (omg*violate_vel**3/sample_num + grad_K2v * grad_v2t * j/sample_num)

                # Collision
                pos_projected = pos[:2]
                edt_dis = self.map.get_edt_dis(pos_projected)
                violate_dis = self.safe_dis - edt_dis

                if violate_dis > 0.0:
                    edt_grad = self.map.get_edt_grad(pos_projected)  # list, len=2
                    # print("At point: ", pos_projected, "edt_grad: ", edt_grad, "violate_dis: ", violate_dis)
                    grad_K2psi = 3 * self.delta_t * omg * violate_dis**2
                    grad_psi2c = -np.array([beta[0]]).T @ np.array([edt_grad])  # (2*s,1) @ (1,2) = (2*s,2)
                    grad_psi2t = (-np.array([edt_grad]) @ np.array([vel]).T).item()

                    self.grad_C[2*self.s*i: 2*self.s*(i+1), :] += self.weights[3] * grad_K2psi * grad_psi2c
                    self.grad_T[i] += self.weights[3] * (omg*violate_dis**3/sample_num + grad_K2psi * grad_psi2t * j/sample_num)

    def map_T2tau(self, ts):
        '''
        inverse sigmoid func
        '''
        tau = np.zeros(self.M)
        for i in range(self.M):
            tau[i] = -math.log((self.T_max - self.T_min) / (ts[i]-self.T_min) - 1)
        return tau

    def map_tau2T(self, tau):
        # print("tau: ", tau)
        ts = np.zeros(self.M)
        for i in range(self.M):
            ts[i] = (self.T_max - self.T_min) / (1 + math.exp(-tau[i])) + self.T_min
        # print("ts: ", ts)
        return ts

    def get_grad_T2tau(self, grad_T):
        # print("grad_T: ", grad_T)
        grad_tau = np.zeros(self.M)
        for i in range(self.M):
            grad_tau[i] = grad_T[i] * (self.T_max - self.T_min) * \
                math.exp(-self.tau[i])/(1+math.exp(-self.tau[i]))**2
        # print("grad_tau: ", grad_tau)
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
        self.int_wpts = np.reshape(x[:self.D*(self.M - 1)], (self.D, self.M - 1))
        self.tau = x[self.D*(self.M - 1):]
        self.ts = self.map_tau2T(self.tau)

        # print("-----------------Current ts and tau-----------------")
        # print("ts: ", self.ts)
        # print("tau: ", self.tau)

        self.get_coeffs(self.int_wpts, self.ts)  # get A and coeffs by input
        self.reset_cost()
        self.add_energy_cost()
        self.add_time_cost()
        self.add_sampled_cost()

        # print("-----------Current weighted cost------------")
        # print("Energy cost: %f, Time cost: %f, Feasibility cost: %f, Collision cost: %f" %
        #       (self.costs[0], self.costs[1], self.costs[2], self.costs[3]))
        # print("\n")
        return np.dot(self.costs, self.weights)

    def get_grad(self, x):
        '''
        get the gradient of cost func to x
        1. Retrieve q and tau from x
        2. Get grad_q and grad_tau, size: grad_q(D,M-1), grad_tau(M,)
        3. Flatten and concatenate grad_q and grad_tau to return a vector        
        '''
        self.int_wpts = np.reshape(x[:self.D*(self.M - 1)], (self.D, self.M - 1))
        self.tau = x[self.D*(self.M - 1):]
        self.ts = self.map_tau2T(self.tau)

        self.get_coeffs(self.int_wpts, self.ts)  # get A and coeffs by input
        self.reset_grad_CT()  # init self.cost, self.grad_c, and self.grad_T as zero
        self.add_energy_grad_CT()
        self.add_time_grad_CT()
        self.add_sampled_grad_CT()

        grad_q, grad_tau = self.propagate_grad_q_tau()  # prop grad to q and tau

        grad = np.concatenate((np.reshape(grad_q, (self.D * (self.M - 1),)), grad_tau), axis=0)

        # print("----------------Current grad----------------")
        # print(grad)
        # print("\n")

        return grad
