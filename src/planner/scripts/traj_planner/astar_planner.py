'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2024-03-05 21:01:49
'''

import math
import matplotlib.pyplot as plt
import copy


class AstarPlanner():

    def __init__(self):
        self.motion = self.get_motion_model()

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = int(x)  # the x index of the grid in the grid map
            self.y = int(y)
            self.cost = cost
            self.parent_index = parent_index  # this index is ONE number defining the grid in the grid map

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(self.cost) + "," + str(self.parent_index)

    def plan(self, map, start_pos, target_pos):
        '''
        map: ESDF() defined in src/planner/scripts/esdf.py
        start_pos: [start_x, start_y]
        target_pos: [target_x, target_y]

        return:
        path: [[x1, y1], [x2, y2], ...]
        '''
        # read map info
        self.map = map
        self.resolution = map.map_resolution
        self.map_width = copy.deepcopy(map.map_width)  # grid num along the x axis
        self.map_height = copy.deepcopy(map.map_height)  # grid num along the y axis
        self.map_origin = copy.deepcopy(map.map_origin)  # in meter, the origin of the map is the west-south corner

        # expand the grid map to ensure that the target position is in the map
        map_expand_radius = 10.0  # in meter, this should be larger than longitu_step_dis in src/planner/launch/planner_config.yaml
        self.map_width += int(map_expand_radius/self.resolution)
        self.map_height += int(map_expand_radius/self.resolution)
        self.map_origin.x -= map_expand_radius/2
        self.map_origin.y -= map_expand_radius/2

        # read mission info
        self.start_pos = start_pos
        self.target_pos = target_pos

        # initialize the nodes
        start_node_index_x, start_node_index_y = self.calc_xy_index(self.start_pos[0], self.start_pos[1])
        self.start_node = self.Node(start_node_index_x, start_node_index_y, 0.0, -1)
        target_node_index_x, target_node_index_y = self.calc_xy_index(self.target_pos[0], self.target_pos[1])
        self.target_node = self.Node(target_node_index_x, target_node_index_y, 0.0, -1)

        # open and close set
        open_set, close_set = dict(), dict()
        open_set[self.calc_grid_index(self.start_node)] = self.start_node

        # search
        while True:
            if not open_set:
                print("Open set is empty, no path found")
                break

            # get the current node
            current_index = min(open_set, key=lambda o: open_set[o].cost + self.calc_heuristic(open_set[o], self.target_node))
            current = open_set[current_index]

            # check if the current node is the target node
            if current.x == self.target_node.x and current.y == self.target_node.y:
                self.target_node.parent_index = current.parent_index
                self.target_node.cost = current.cost
                break

            # move the current node to the close set
            del open_set[current_index]
            close_set[current_index] = current

            # expand the current node
            for move_x, move_y, move_cost in self.motion:
                node = self.Node(current.x + move_x, current.y + move_y, current.cost + move_cost, current_index)
                node_index = self.calc_grid_index(node)

                # check if the node is valid
                if node_index in close_set:
                    continue

                if not self.verify_node(node):
                    continue

                # check if the node is already in the open set
                if node_index not in open_set:
                    open_set[node_index] = node
                else:
                    if open_set[node_index].cost > node.cost:
                        open_set[node_index] = node

        # retrieve the path
        path = self.retrieve_final_path(close_set)

        # plot path
        # self.visualize_path(path)

        return path

    def get_motion_model(self):
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion

    def calc_real_pos(self, index_x, index_y):
        return self.map_origin.x + index_x * self.resolution, self.map_origin.y + index_y * self.resolution

    def calc_xy_index(self, pos_x, pos_y):
        return int((pos_x - self.map_origin.x) / self.resolution), int((pos_y - self.map_origin.y) / self.resolution)

    def calc_grid_index(self, node):
        # return a unique number representing the grid
        return node.x + node.y * self.map_width

    @staticmethod
    def calc_heuristic(node1, node2):
        return math.hypot(node1.x - node2.x, node1.y - node2.y)

    def collision_check(self, node):
        # check if the node is in collision
        return self.map.has_collision(self.calc_real_pos(node.x, node.y))

    def verify_node(self, node):
        # check if the node is valid, if valid return True
        if (
            node.x < 0 or node.x >= self.map_width
            or node.y < 0 or node.y >= self.map_height
            or self.collision_check(node)
        ):
            return False

        return True

    def retrieve_final_path(self, close_set):
        path = [list(self.calc_real_pos(self.target_node.x, self.target_node.y))]
        parent_index = self.target_node.parent_index
        while parent_index != -1:
            node = close_set[parent_index]
            path.append(list(self.calc_real_pos(node.x, node.y)))
            parent_index = node.parent_index

        return path[::-1]

    def visualize_path(self, path):
        plt.plot([x[0] for x in path], [x[1] for x in path], 'r-')
        plt.axis("equal")
        plt.show()
