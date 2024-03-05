'''
created by Jinjie LI, 2024/03/05
'''
import os
import rospkg
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

rospack = rospkg.RosPack()
param_path = os.path.join(rospack.get_path("planner"), "data", "planning_metrics.txt")

# read data
data = pd.read_csv(param_path, sep=' ', header=None)
data.columns = ['date', 'time', 'world', 'planner', 'replan_mode', 'if_reached_target', 'target_x', 'target_y',
                'target_find_time', 'max_target_find_time', 'weighted_metric', 'average_iter_num',
                'average_planning_duration', 'total_planning_times']

# filter data based on the 'planner' and 'world' columns. Loop through all the planners and worlds
planners = data['planner'].unique()
worlds = data['world'].unique()

for planner in planners:
    for world in worlds:
        # filter the data
        filtered_data = data[(data['planner'] == planner) & (data['world'] == world)]

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


#                 file.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' ')
#                 file.write(self.gazebo_world + ' ')
#                 file.write(self.selected_planner + ' ')
#                 file.write(self.replan_mode + ' ')
#                 file.write(str(self.reached_target) + ' ')
#                 file.write(str(self.global_target[0]) + ' ')  # x
#                 file.write(str(self.global_target[1]) + ' ')  # y
#                 file.write(str(self.target_find_time) + ' ')
#                 file.write(str(self.max_target_find_time) + ' ')
#                 file.write(str(weighted_metric) + ' ')
#                 file.write(str(average_iter_num) + ' ')
#                 file.write(str(average_planning_duration) + ' ')
#                 file.write(str(self.total_planning_times) + '\n')
