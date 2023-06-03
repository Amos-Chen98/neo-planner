'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2023-06-03 21:20:26
'''
import numpy as np
import onnxruntime
from traj_utils import TrajUtils


IMG_WIDTH = 160
IMG_HEIGHT = 120


class NNPlanner(TrajUtils):
    def __init__(self, onnx_path='saved_net/planner_net.onnx'):
        super().__init__()
        # load onnx model, ONNX runtime reference: https://onnxruntime.ai/docs/api/python/api_summary.html
        self.onnx_path = onnx_path
        self.session = onnxruntime.InferenceSession(self.onnx_path,
                                                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

        self.onnx_input_name = self.session.get_inputs()[0].name
        self.onnx_output_name = self.session.get_outputs()[0].name

        print("ONNX runtime version: ", onnxruntime.__version__)
        print("ONNX available providers: ", self.session.get_providers())
        print("ONNX selected provider: ", self.session.get_providers()[0])
        print("ONNX input name: ", self.onnx_input_name)
        print("ONNX output name: ", self.onnx_output_name)

        # Planning parameters
        self.coeffs = []
        self.M = 3
        self.s = 3
        self.D = 3
        self.head_state = np.zeros((self.s, self.D))
        self.tail_state = np.zeros((self.s, self.D))

        print("NNPlanner initialized")

    def convert_input(self, depth_image_norm, motion_info, drone_global_pos):
        '''
        convert input to the format that ONNX model accepts
        also get self.head_state and self.tail_state from motion_info
        '''
        img_resized = np.resize(depth_image_norm, (IMG_WIDTH, IMG_HEIGHT))
        img_flatten = img_resized.reshape(-1)
        input_concat = np.array([np.concatenate((img_flatten, motion_info.astype(np.float32)))])  # dtype of ortvalue must be float32
        ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(input_concat)

        # get boundary conditions, this is used for calculating full state cmd
        self.head_state[:2, :] = motion_info[12:18].reshape(self.s - 1, self.D)  # only pos and vel are valid
        self.head_state[2, :] = np.zeros(3)  # acc is zero, since we cannot access the real acc of a drone
        self.tail_state[:2, :] = motion_info[18:24].reshape(self.s - 1, self.D)
        self.tail_state[2, :] = np.zeros(3)

        # get drone local state
        self.drone_attitude = motion_info[3:12].reshape(3, 3) # as a rotation matrix
        self.drone_global_pos = drone_global_pos

        return ortvalue

    def get_wpts_world(self, int_wpts):
        '''
        convert wpts from body frame to world frame
        '''
        # convert int_wpts to world frame
        int_wpts_world = np.zeros((self.D, self.M-1))
        for i in range(self.M-1):
            int_wpts_world[:, i] = self.drone_attitude @ int_wpts[:, i] + self.drone_global_pos

        return int_wpts_world

    def plan(self, depth_image_norm, motion_info, drone_global_pos):
        '''
        input:
        depth_image_norm: one depth img of original size
        motion_info = np.concatenate((drone_local_vel, (3,)
                                    drone_attitude, (3,3)
                                    plan_init_pos, (3,)
                                    plan_init_vel, (3,)
                                    plan_target_pos, (3,)
                                    plan_target_vel), axis=0) (3,)
        Stores the results in self.int_wpts (D, M-1), self.ts (M, 1)
        '''
        ortvalue = self.convert_input(depth_image_norm, motion_info, drone_global_pos)

        output = self.session.run([self.onnx_output_name],
                                  {self.onnx_input_name: ortvalue})[0]  # size: (1, 9)

        int_wpts_local = output[0][:self.D*(self.M-1)].reshape(self.D, self.M-1)  # the num of wpts is M-1, and the dim of each wpt is D
        self.ts = output[0][self.D*(self.M-1):]
        self.int_wpts = self.get_wpts_world(int_wpts_local)
        print("int_wpts: ", self.int_wpts)
        print("ts: ", self.ts)
        print("NN planning finished!!")
