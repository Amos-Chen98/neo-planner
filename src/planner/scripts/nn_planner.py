'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2023-08-07 23:53:55
'''
import os
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_path)
from record_planner import form_nn_input
import torch
import onnxruntime  # torch must be included before onnxruntime, ref:https://stackoverflow.com/questions/75267445/why-does-onnxruntime-fail-to-create-cudaexecutionprovider-in-linuxubuntu-20/75267493#75267493
import numpy as np
from traj_utils import TrajUtils
from nn_trainer import process_input_np
import time
# import pycuda.driver as cuda
# import tensorrt as trt
# import pycuda.autoinit # this must be included
# import cv2


IMG_WIDTH = 480
IMG_HEIGHT = 360
VECTOR_SIZE = 24
INPUT_SHAPE = (1, IMG_WIDTH*IMG_HEIGHT+VECTOR_SIZE)

class NNPlanner(TrajUtils):
    def __init__(self, des_pos_z=2.0):
        super().__init__()

        rospkg_path = current_path[:-8]  # -8 remove '/scripts'
        onnx_model_path = rospkg_path + '/saved_net/planner_net.onnx'
        trt_model_path = rospkg_path + '/saved_net/planner_net.trt'

        self.init_onnx_model(onnx_model_path)
        # self.init_trt_model(trt_model_path)

        self.init_planning_params(des_pos_z)

        print("NNPlanner initialized")

    def init_onnx_model(self, onnx_model_path):
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

        self.session = onnxruntime.InferenceSession(onnx_model_path,
                                                    providers=provider)

        self.onnx_input_name = self.session.get_inputs()[0].name
        self.onnx_output_name = self.session.get_outputs()[0].name

        print("ONNX selected provider: ", self.session.get_providers()[0])
        print("ONNX input name: ", self.onnx_input_name)
        print("ONNX output name: ", self.onnx_output_name)

    # def init_trt_model(self, trt_model_path):
    #     print("TensorRT version: ", trt.__version__)
    #     TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    #     with open(trt_model_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    #         self.engine = runtime.deserialize_cuda_engine(f.read())
        
    #     self.context = self.engine.create_execution_context()

    #     input_index = self.engine.get_binding_index("input")
    #     output_index = self.engine.get_binding_index("output")
    #     input_shape = self.engine.get_binding_shape(input_index)
    #     self.output_shape = self.engine.get_binding_shape(output_index)
    #     print("input shape: ", input_shape)
    #     print("output shape: ", self.output_shape)

    #     input_dtype = trt.nptype(self.engine.get_binding_dtype(input_index))
    #     self.output_dtype = trt.nptype(self.engine.get_binding_dtype(output_index))

    #     input_size = np.product(input_shape) * np.dtype(input_dtype).itemsize
    #     output_size = np.product(self.output_shape) * np.dtype(self.output_dtype).itemsize
    #     # input_size = trt.volume(input_shape) * trt.float16.itemsize
    #     # output_size = trt.volume(self.output_shape) * trt.float16.itemsize
        
    #     # Allocate device memory for input and output
    #     self.d_input = cuda.mem_alloc(int(input_size)) # allocate space in device (GPU)
    #     self.d_output = cuda.mem_alloc(int(output_size))
    #     # self.d_input = cuda.mem_alloc(int(input_size)) # allocate space in device (GPU)
    #     # self.d_output = cuda.mem_alloc(int(output_size))
        
    # def process_trt_input(self, depth_img, motion_info):
    #     '''
    #     :param depth_img: numpy array
    #     :param motion_info: numpy array
    #     :return: numpy array
    #     '''
    #     depth_img_resized = cv2.resize(depth_img, (IMG_WIDTH, IMG_HEIGHT))
    #     img_flatten = depth_img_resized.reshape(-1)
    #     return np.concatenate((img_flatten.astype(np.float16), motion_info.astype(np.float16)))

    # def trt_predict(self, depth_image_norm, motion_info):
    #     input = np.random.rand(*INPUT_SHAPE).astype(np.float16)
    #     print("real input shape: ", input.shape)
    #     cuda.memcpy_htod(self.d_input, input) # from host(memory) to device (GPU)

    #     # Execute inference
    #     self.context.execute_v2(bindings=[int(self.d_input), int(self.d_output)])

    #     # Copy output from device memory
    #     output = np.empty(self.output_shape, dtype=self.output_dtype)
    #     cuda.memcpy_dtoh(output, self.d_output)
        
    #     print("output: ", output)

        # int_wpts_local = output[0][:self.nn_output_D*(self.M-1)].reshape(self.M-1, self.nn_output_D).T  # col major, so transpose
        # self.ts = output[0][self.nn_output_D*(self.M-1):]

        # int_wpts_3d = self.get_wpts_world(int_wpts_local)
        # self.int_wpts = int_wpts_3d[:self.D, :]  # remove z axis

    # def trt_predict(self, depth_image_norm, motion_info):
    #     input = np.random.rand(*INPUT_SHAPE).astype(np.float16)

    #     with self.engine.create_execution_context() as context:
    #         # stream = cuda.Stream()
    #         # # Copy input to device memory
    #         # cuda.memcpy_htod_async(self.d_input, input, stream)
    #         # # Execute inference
    #         # context.execute_async_v2(bindings=[int(self.d_input), int(self.d_output)], stream_handle=stream.handle)
    #         # # Copy output from device memory
    #         # output = np.empty(self.output_shape, dtype=self.output_dtype)
    #         # cuda.memcpy_dtoh_async(output, self.d_output, stream)
    #         # # Synchronize the stream
    #         # stream.synchronize()
            
    #         # print("Current thread: ", threading.currentThread())
    #         # with self.engine.create_execution_context() as context:
            
    #         # Copy input to device memory
    #         cuda.memcpy_htod(self.d_input, input) # from host(memory) to device (GPU)

    #         # Execute inference
    #         context.execute_v2(bindings=[int(self.d_input), int(self.d_output)])

    #         # Copy output from device memory
    #         output = np.empty(self.output_shape, dtype=self.output_dtype)
    #         cuda.memcpy_dtoh(output, self.d_output)
        
    #     print("output: ", output)

    #     int_wpts_local = output[0][:self.nn_output_D*(self.M-1)].reshape(self.M-1, self.nn_output_D).T  # col major, so transpose
    #     self.ts = output[0][self.nn_output_D*(self.M-1):]

    #     int_wpts_3d = self.get_wpts_world(int_wpts_local)
    #     self.int_wpts = int_wpts_3d[:self.D, :]  # remove z axis

        # print("int_wpts: ", self.int_wpts)
        # print("ts: ", self.ts)

    def init_planning_params(self, des_pos_z):
        # Planning parameters
        self.M = 3
        self.s = 3
        self.D = 2
        self.nn_output_D = 3
        self.head_state = np.zeros((self.s, self.D))
        self.tail_state = np.zeros((self.s, self.D))
        self.des_pos_z = des_pos_z

    def nn_traj_plan(self, depth_img, drone_state, plan_init_state, target_state):
        time_start = time.time()
        depth_image_norm, motion_info = form_nn_input(depth_img, drone_state, self.des_pos_z, plan_init_state, target_state)
        self.drone_state = drone_state
        self.head_state[0, :self.D] = plan_init_state.global_pos[:2]
        self.head_state[1, :self.D] = plan_init_state.global_vel[:2]
        self.tail_state[0, :self.D] = target_state[0, :2]
        self.tail_state[1, :self.D] = target_state[1, :2]

        self.onnx_predict(depth_image_norm, motion_info)
        # self.trt_predict(depth_image_norm, motion_info)
        time_end = time.time()
        print("ONNX inference time: {}".format(time_end - time_start))

    def onnx_predict(self, depth_image_norm, motion_info):
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
        ortvalue = self.get_ortvalue(depth_image_norm, motion_info)

        output = self.session.run([self.onnx_output_name],
                                  {self.onnx_input_name: ortvalue})[0]  # size: (1, 9)

        int_wpts_local = output[0][:self.nn_output_D*(self.M-1)].reshape(self.M-1, self.nn_output_D).T  # col major, so transpose
        self.ts = output[0][self.nn_output_D*(self.M-1):]

        int_wpts_3d = self.get_wpts_world(int_wpts_local)
        self.int_wpts = int_wpts_3d[:self.D, :]  # remove z axis

        # print("int_wpts: ", self.int_wpts)
        # print("ts: ", self.ts)

    def get_ortvalue(self, depth_image_norm, motion_info):
        '''
        convert input to the format that ONNX model accepts
        '''
        input_np = np.array([process_input_np(depth_image_norm, motion_info)])  # to make the shape (xx,) to (1, xx)

        ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(input_np)

        return ortvalue

    def get_wpts_world(self, int_wpts):
        '''
        convert wpts from body frame to world frame
        '''
        # drone_attitude = motion_info[3:12].reshape(3, 3)  # as a rotation matrix
        drone_attitude = self.drone_state.attitude.rotation_matrix    # as a rotation matrix
        drone_global_pos_2d = self.drone_state.global_pos

        int_wpts_world = np.zeros((self.nn_output_D, self.M-1))
        for i in range(self.M-1):
            int_wpts_world[:, i] = drone_attitude @ int_wpts[:, i] + drone_global_pos_2d

        return int_wpts_world
