'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2024-03-08 22:29:45
'''
import os
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_path)
from expert_planner import MinJerkPlanner
import cv2
import numpy as np
import pandas as pd
import datetime
from PIL import Image


def form_nn_input(depth_img, drone_state, des_pos_z, plan_init_state, target_state):
    # make the max value of depth_img to be 255
    depth_img_norm = (depth_img / np.max(depth_img) * 255).astype(np.uint8)
    
    # current drone state
    drone_local_vel = drone_state.local_vel  # size: (3,)
    drone_quat = drone_state.attitude  # size: (4,)
    # drone_attitude = np.array([drone_quat.w, drone_quat.x, drone_quat.y, drone_quat.z])  # size: (4,)
    drone_attitude = drone_state.attitude.rotation_matrix.reshape(-1)  # size: (9, ), expand by Row

    # plan_init_state, in map frame
    plan_init_state_3d = np.zeros((2, 3))
    plan_init_state_3d[0, :2] = plan_init_state.global_pos[:2]  # NOTE
    plan_init_state_3d[0, 2] = des_pos_z
    plan_init_state_3d[1, :2] = plan_init_state.global_vel[:2]

    # plan_init_state, in body frame (current drone_state), row1: pos, row2: vel
    plan_init_pos = drone_quat.inverse.rotate(plan_init_state_3d[0] - drone_state.global_pos)  # size: (3,)
    plan_init_vel = drone_quat.inverse.rotate(plan_init_state_3d[1] - drone_state.global_vel)  # size: (3,) NOTE
    # plan_init_vel = drone_quat.inverse.rotate(plan_init_state_3d[1])  # size: (3,) NOTE

    # target_state, in map frame
    target_state_3d = np.zeros((2, 3))
    target_state_3d[:, :2] = target_state  # includes target_pos and target_vel
    target_state_3d[0, 2] = des_pos_z

    # target_state, in body frame
    plan_target_pos = drone_quat.inverse.rotate(target_state_3d[0] - drone_state.global_pos)  # size: (3,)
    plan_target_vel = drone_quat.inverse.rotate(target_state_3d[1] - drone_state.global_vel)  # size: (3,)
    # plan_target_vel = drone_quat.inverse.rotate(target_state_3d[1])  # size: (3,) NOTE

    motion_info = np.concatenate((drone_local_vel,
                                  drone_attitude,
                                  plan_init_pos,
                                  plan_init_vel,
                                  plan_target_pos,
                                  plan_target_vel), axis=0)

    # print("drone_local_vel: ", drone_local_vel)
    # print("drone_attitude: ", drone_attitude)
    # print("plan_init_pos_local: ", plan_init_pos)
    # print("plan_init_vel_local: ", plan_init_vel)
    # print("plan_target_pos_local: ", plan_target_pos)
    # print("plan_target_vel_local: ", plan_target_vel)

    return depth_img_norm, motion_info


def form_nn_output(drone_state, des_pos_z, int_wpts):
    # process output result: int_wpts, in body frame
    drone_quat = drone_state.attitude  # size: (4,)
    int_wpts_num = int_wpts.shape[1]
    int_wpts_local = np.zeros((3, int_wpts_num))  # col major
    for i in range(int_wpts_num):
        int_wpts_3d = np.array([int_wpts[0, i], int_wpts[1, i], des_pos_z])
        int_wpts_local[:, i] = drone_quat.inverse.rotate(int_wpts_3d - drone_state.global_pos)

    int_wpts_local = int_wpts_local.T.reshape(-1)  # size: (3*int_wpts_num,)

    return int_wpts_local


class RecordPlanner(MinJerkPlanner):
    def __init__(self, planner_config):
        super().__init__(planner_config)

        rospkg_path = current_path[:-21]  # -8 remove '/scripts', -13 remove '/traj_planner'
        self.csv_path = rospkg_path + '/training_data/train.csv'
        self.img_path = rospkg_path + '/training_data/depth_img'

        self.des_pos_z = planner_config.des_pos_z

        # Initialize the csv file collecting training data
        self.table_header = ['id',
                             'drone_vel_x',
                             'drone_vel_y',
                             'drone_vel_z',
                             'R11',
                             'R12',
                             'R13',
                             'R21',
                             'R22',
                             'R23',
                             'R31',
                             'R32',
                             'R33',
                             'init_pos_x',
                             'init_pos_y',
                             'init_pos_z',
                             'init_vel_x',
                             'init_vel_y',
                             'init_vel_z',
                             'target_pos_x',
                             'target_pos_y',
                             'target_pos_z',
                             'target_vel_x',
                             'target_vel_y',
                             'target_vel_z',
                             'wpts1_x',
                             'wpts1_y',
                             'wpts1_z',
                             'wpts2_x',
                             'wpts2_y',
                             'wpts2_z',
                             'ts1',
                             'ts2',
                             'ts3'
                             ]

        # create a blank csv file
        if not os.path.isfile(self.csv_path):
            df = pd.DataFrame(columns=self.table_header)
            df.to_csv(self.csv_path, index=False)
        
        if not os.path.exists(self.img_path):
            os.makedirs(self.img_path)

    def record_traj_plan(self, map, depth_img, drone_state, plan_init_state, target_state):
        '''
        trajectory planning + store the training data
        '''
        # state ahead of 1s
        drone_state_2d = np.array([plan_init_state.global_pos[:2],
                                   plan_init_state.global_vel[:2]])
        # self.plan(map, drone_state_2d, target_state)  # 2D planning, z is fixed
        self.batch_plan(map, drone_state_2d, target_state)

        # get planning results
        int_wpts = self.int_wpts
        ts = self.ts

        self.save_training_data(depth_img, drone_state, plan_init_state, target_state, int_wpts, ts)  # record training data

    def save_training_data(self, depth_img, drone_state, plan_init_state, target_state, int_wpts, ts):
        '''
        Record training data and write to csv file
        '''
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
        # mon-day-hour-min-sec-ms, [:-3] because the last 3 digits are microsecond

        # process input data
        depth_image_norm, motion_info = form_nn_input(depth_img, drone_state, self.des_pos_z, plan_init_state, target_state)

        # process output result: int_wpts, in body frame
        int_wpts_local = form_nn_output(drone_state, self.des_pos_z, int_wpts)

        new_line_df = pd.DataFrame(columns=self.table_header)
        new_line_df.loc[0] = [None] * 34
        new_line_df.loc[0][0] = 't' + timestamp
        # the extra 't' is to ensure the id is a string, otherwise it will be stored as a num in the csv, even using str(timestamp)
        new_line_df.loc[0][1:] = np.concatenate((motion_info, int_wpts_local, ts), axis=0).tolist()
        new_line_df.to_csv(self.csv_path, mode='a', header=False, index=False)

        # create a gray image from depth_image_norm
        depth_img_gray = Image.fromarray(depth_image_norm)

        # store the depth image
        depth_img_gray.save(self.img_path + '/' + str(timestamp) + '.png')
        
        print("Training data (ID: %s) saved!" % timestamp)
