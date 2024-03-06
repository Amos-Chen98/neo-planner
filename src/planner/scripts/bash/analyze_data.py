'''
created by Jinjie LI, 2024/03/05
'''
import os
import rospkg
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def analyze_with_planner(in_data, world):
    planners = in_data['planner'].unique()
    for planner in planners:
        # filter the data
        filtered_data = in_data[in_data['planner'] == planner]

        # print success rate
        success_rate = filtered_data['if_reached_target'].sum() / len(filtered_data)
        print(planner + ' in ' + world + ' success rate: ' + str(success_rate))

        # print average weighted metric
        average_weighted_metric = filtered_data['weighted_metric'].mean()
        print(planner + ' in ' + world + ' average weighted metric: ' + str(average_weighted_metric))

        # print average planning duration
        average_planning_duration = filtered_data['average_planning_duration'].mean()
        print(planner + ' in ' + world + ' average planning duration: ' + str(average_planning_duration))

        # print average iteration number
        average_iter_num = filtered_data['average_iter_num'].mean()
        print(planner + ' in ' + world + ' average iteration number: ' + str(average_iter_num))

        # print average target find time
        average_target_find_time = filtered_data['target_find_time'].mean()
        print(planner + ' in ' + world + ' average target find time: ' + str(average_target_find_time))

        # print average total planning times
        average_total_planning_times = filtered_data['total_planning_times'].mean()
        print(planner + ' in ' + world + ' average total planning times: ' + str(average_total_planning_times) + '\n')


rospack = rospkg.RosPack()
param_path = os.path.join(rospack.get_path("planner"), "data", "planning_metrics.txt")

# read data
data = pd.read_csv(param_path, sep=' ', header=None)
data.columns = ['date', 'time', 'world', 'world_num_model', 'planner', 'replan_mode', 'if_reached_target', 'target_x',
                'target_y',
                'target_find_time', 'max_target_find_time', 'weighted_metric', 'average_iter_num',
                'average_planning_duration', 'total_planning_times']


multi_num_models = data['world_num_model'].unique()

for num_model in multi_num_models:
    data_num_model = data[data['world_num_model'] == num_model]
    print('For ' + str(num_model) + ' models:')
    if num_model == 0:
        worlds = data_num_model['world'].unique()
        for world in worlds:
            data_filtered = data_num_model[data_num_model['world'] == world]
            analyze_with_planner(data_filtered, world)
    else:
        analyze_with_planner(data_num_model, str(num_model))
