import os
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_path)
import copy
from expert_planner import MinJerkPlanner
from astar_planner import AstarPlanner
import numpy as np
import math


class GeoPlanner(MinJerkPlanner):
    def __init__(self, planner_config):
        super().__init__(planner_config)
        self.astar_planner = AstarPlanner()
        # self.int_wpts_num = planner_config['init_wpts_num'] # default is 2
        self.int_wpts_num = 2

    def geo_traj_plan(self, map, plan_init_state, target_state):
        start_pos = plan_init_state.global_pos[:2]
        target_pos = target_state[0]  # the 2D position
        path = self.astar_planner.plan(map, start_pos, target_pos)
        path_pruned = self.prune_path_nodes(map, path)

        # return path_pruned

        int_wpts = np.array(path_pruned[1:3]).T
        ts = self.init_T * np.ones((self.int_wpts_num+1,))  # allocate time for each piece
        ts[0] *= 1.5
        ts[-1] *= 1.5

        # use expert_planner to generate traj
        drone_state_2d = np.array([plan_init_state.global_pos[:2],
                                   plan_init_state.global_vel[:2]])
        self.warm_start_plan(map, drone_state_2d, target_state, int_wpts, ts)  # 2D planning, z is fixed

    def seg_feasible_check(self, map, head_pos, tail_pos):
        '''
        check if the straight line from head_pos to tail_pos is feasible
        '''
        x0 = head_pos[0]
        y0 = head_pos[1]
        x1 = tail_pos[0]
        y1 = tail_pos[1]

        step_num = math.ceil(max(abs(x1 - x0), abs(y1 - y0))/0.1) + 1
        x_check_list = np.linspace(x0, x1, step_num)
        y_check_list = np.linspace(y0, y1, step_num)

        for i in range(step_num):
            if map.get_edt_dis([x_check_list[i], y_check_list[i]]) < 0.4:
                return False
        return True

    def prune_path_nodes(self, map, path):
        '''
        only keep 'init_wpts_num' nodes in the path (not including the start and target nodes)
        '''
        key_index = [0]
        head_index = int(0)
        tail_index = int(1)

        while tail_index < len(path):
            while self.seg_feasible_check(map, path[head_index], path[tail_index]) or tail_index - head_index == 1:  # head and tail included
                tail_index += 1
                if tail_index == len(path):
                    break

            # the path from head_index to [tail_index - 1] is feasible
            # the path from head_index to tail_index is not feasible
            key_index.append(tail_index - 1)
            head_index = copy.deepcopy(tail_index - 1)  # reset the head_index

        # keep the key nodes
        key_nodes_num = len(key_index)

        if key_nodes_num == 2:
            final_key_index = np.linspace(key_index[0], key_index[-1], 4).astype(int)
        elif key_nodes_num == 3:
            if key_index[1] - key_index[0] > key_index[2] - key_index[1]:
                extra_key_index = int((key_index[0] + key_index[1])/2)
                final_key_index = [key_index[0], extra_key_index, key_index[1], key_index[2]]
            else:
                extra_key_index = int((key_index[1] + key_index[2])/2)
                final_key_index = [key_index[0], key_index[1], extra_key_index, key_index[2]]
        elif key_nodes_num == 4:
            final_key_index = key_index
        else:
            ancher_index_left = 1/3*key_index[-1]
            ancher_index_right = 2/3*key_index[-1]

            index_left = min(key_index, key=lambda x: abs(x-ancher_index_left))
            index_right = min(key_index, key=lambda x: abs(x-ancher_index_right))

            final_key_index = [key_index[0], index_left, index_right, key_index[-1]]

        path_pruned = []
        for i in final_key_index:
            path_pruned.append(path[i])

        return path_pruned
