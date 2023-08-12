'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2023-08-12 20:35:37
'''
import os
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_path)
from expert_planner import MinJerkPlanner
import numpy as np
from nn_planner import NNPlanner


class EnhancedPlanner(MinJerkPlanner):
    def __init__(self, planner_config):
        super().__init__(planner_config)
        des_pos_z = planner_config.des_pos_z
        self.nn_planner = NNPlanner(des_pos_z)

    # def enhanced_traj_plan(self, map, depth_img, drone_state, plan_init_state, target_state):
    #     # get init int_wpts and ts using nn_planner
    #     self.nn_planner.nn_traj_plan(depth_img, drone_state, plan_init_state, target_state)
    #     int_wpts = self.nn_planner.int_wpts
    #     ts = self.nn_planner.ts

    #     drone_state_2d = np.array([plan_init_state.global_pos[:2],
    #                                plan_init_state.global_vel[:2]])
    #     self.read_planning_conditions(map, drone_state_2d, target_state, int_wpts, ts)

    #     # evaluate the cost of traj obtained by nn_planner
    #     self.get_coeffs(int_wpts, ts)  # get A and coeffs by input
    #     self.reset_cost()
    #     self.add_sampled_cost()
    #     weighted_cost_partial = self.costs * self.weights
    #     collision_cost = weighted_cost_partial[3]

    #     if collision_cost > self.collision_cost_tol:
    #         print("Collision cost too large: %f, use expert planner" % collision_cost)
    #         drone_state_2d = np.array([plan_init_state.global_pos[:2],
    #                                    plan_init_state.global_vel[:2]])
    #         self.warm_start_plan(map, drone_state_2d, target_state, int_wpts, ts)  # 2D planning, z is fixed
    #     else:
    #         self.int_wpts = int_wpts
    #         self.ts = ts

    def enhanced_traj_plan(self, map, depth_img, drone_state, plan_init_state, target_state):
        # get init int_wpts and ts using nn_planner
        self.nn_planner.nn_traj_plan(depth_img, drone_state, plan_init_state, target_state)
        int_wpts = self.nn_planner.int_wpts
        ts = self.nn_planner.ts

        # use expert_planner to generate traj
        drone_state_2d = np.array([plan_init_state.global_pos[:2],
                                   plan_init_state.global_vel[:2]])
        self.warm_start_plan(map, drone_state_2d, target_state, int_wpts, ts)  # 2D planning, z is fixed
