import numpy as np


class TrajUtils():
    def __init__(self) -> None:
        pass

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
        self.get_coeffs(self.int_wpts, self.ts)

        total_time = sum(self.ts)
        t_samples = np.arange(0, total_time, 1/hz)
        sample_num = t_samples.shape[0]
        state_cmd = np.zeros((sample_num, 3, self.D))  # 3*D: [pos, vel, acc].T * D

        for i in range(sample_num):
            t = t_samples[i]
            state_cmd[i][0] = self.get_pos(t)
            state_cmd[i][1] = self.get_vel(t)
            state_cmd[i][2] = self.get_acc(t)

        return state_cmd

    def get_pos_array(self):
        '''
        return the full pos array
        '''
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
