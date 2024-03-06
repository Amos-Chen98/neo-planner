'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2024-03-06 21:04:49
'''
import os
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_path)
from expert_planner import MinJerkPlanner
import numpy as np
from nn_planner import NNPlanner


class PlanningResult():
    def __init__(self, planner_name):
        self.planner_name = planner_name
        self.int_wpts = []
        self.pos_array = []
        self.traj_cost = 0.0


class EnhancedPlanner(MinJerkPlanner):
    def __init__(self, planner_config):
        super().__init__(planner_config)
        des_pos_z = planner_config.des_pos_z
        self.nn_planner = NNPlanner(des_pos_z)
        self.planning_result = dict()
        self.planning_result["NN"] = PlanningResult("NN")
        self.planning_result["Enhanced"] = PlanningResult("Enhanced")

    def enhanced_traj_plan(self, map, depth_img, drone_state, plan_init_state, target_state):
        drone_state_2d = np.array([plan_init_state.global_pos[:2],
                                   plan_init_state.global_vel[:2]])

        # --------------------------------NN planner--------------------------------
        self.nn_planner.nn_traj_plan(depth_img, drone_state, plan_init_state, target_state)
        int_wpts = self.nn_planner.int_wpts
        ts = self.nn_planner.ts

        # to fill M, D, and boundary states into MinJerkPlanner
        self.read_planning_conditions(map, drone_state_2d, target_state, int_wpts, ts)
        # retrieve the pos_array from nn_planner
        nn_pos_array = self.get_pos_array()

        # evaluate the cost of traj obtained by nn_planner
        self.get_coeffs(int_wpts, ts)
        self.reset_cost()
        self.add_energy_cost()
        self.add_time_cost()
        self.add_sampled_cost()
        nn_cost = np.dot(self.costs, self.weights)

        # Store the result of nn_planner
        self.planning_result["NN"].int_wpts = int_wpts.T
        self.planning_result["NN"].pos_array = nn_pos_array
        self.planning_result["NN"].traj_cost = nn_cost

        # --------------------------------enhanced planner--------------------------------
        # plan from NN output
        self.warm_start_plan(map, drone_state_2d, target_state, int_wpts, ts)  # 2D planning, z is fixed
        enhanced_pos_array = self.get_pos_array()

        # evaluate the cost of traj obtained by enhanced_planner
        enhanced_cost = np.dot(self.costs, self.weights)

        # Store the result of enhanced_planner
        self.planning_result["Enhanced"].int_wpts = self.int_wpts.T
        self.planning_result["Enhanced"].pos_array = enhanced_pos_array
        self.planning_result["Enhanced"].traj_cost = enhanced_cost

        # print the planning result (by emualting the dict planning_result)
        print("Planning result:")
        for key, result in self.planning_result.items():
            print("Planner: ", result.planner_name)
            print("Trajectory cost: ", result.traj_cost)
            print("")
