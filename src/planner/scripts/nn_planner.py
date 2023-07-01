'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2023-07-01 18:08:42
'''
import os
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_path)
from nn_trainer import process_input_np
from traj_utils import TrajUtils
import numpy as np
import torch
import onnxruntime  # torch must be included before onnxruntime, ref:https://stackoverflow.com/questions/75267445/why-does-onnxruntime-fail-to-create-cudaexecutionprovider-in-linuxubuntu-20/75267493#75267493
from record_planner import form_nn_input



class NNPlanner(TrajUtils):
    def __init__(self, des_pos_z=2.0):
        super().__init__()

        rospkg_path = current_path[:-8]  # -8 remove '/scripts'
        model_path = rospkg_path + '/saved_net/planner_net.onnx'

        # load onnx model, ONNX runtime reference: https://onnxruntime.ai/docs/api/python/api_summary.html
        print("ONNX runtime version: ", onnxruntime.__version__)

        providers_available = onnxruntime.get_available_providers()
        print("Available providers: ", providers_available)

        if 'CUDAExecutionProvider' in providers_available:
            provider = ['CUDAExecutionProvider']
            print("CUDAExecutionProvider is available")
        else:
            provider = ['CPUExecutionProvider']
            print("CPUExecutionProvider is available")

        self.session = onnxruntime.InferenceSession(model_path,
                                                    providers=provider)

        self.onnx_input_name = self.session.get_inputs()[0].name
        self.onnx_output_name = self.session.get_outputs()[0].name

        print("ONNX selected provider: ", self.session.get_providers()[0])
        print("ONNX input name: ", self.onnx_input_name)
        print("ONNX output name: ", self.onnx_output_name)

        # Planning parameters
        self.M = 3
        self.s = 3
        self.D = 3
        self.head_state = np.zeros((self.s, self.D))
        self.tail_state = np.zeros((self.s, self.D))
        self.des_pos_z = des_pos_z

        print("NNPlanner initialized")

    def nn_traj_plan(self, depth_img, drone_state, plan_init_state, target_state):
        depth_image_norm, motion_info, drone_global_pos = form_nn_input(depth_img, drone_state, self.des_pos_z, plan_init_state, target_state)
        self.plan(depth_image_norm, motion_info, drone_global_pos)

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

        int_wpts_local = output[0][:self.D*(self.M-1)].reshape(self.M-1, self.D).T  # col major, so transpose
        self.ts = output[0][self.D*(self.M-1):]
        self.int_wpts = self.get_wpts_world(int_wpts_local)
        print("int_wpts: ", self.int_wpts)
        print("ts: ", self.ts)

    def convert_input(self, depth_image_norm, motion_info, drone_global_pos):
        '''
        convert input to the format that ONNX model accepts
        also get self.head_state and self.tail_state from motion_info
        '''

        input_np = np.array([process_input_np(depth_image_norm, motion_info)])  # to make the shape (xx,) to (1, xx)

        ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(input_np)

        # get boundary conditions, this is used for calculating full state cmd
        self.head_state[:2, :] = motion_info[12:18].reshape(self.s - 1, self.D)  # only pos and vel are valid, row major
        self.head_state[2, :] = np.zeros(3)  # acc is zero, since we cannot access the real acc of a drone
        self.tail_state[:2, :] = motion_info[18:24].reshape(self.s - 1, self.D)
        self.tail_state[2, :] = np.zeros(3)

        # get drone local state
        self.drone_attitude = motion_info[3:12].reshape(3, 3)  # as a rotation matrix
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
