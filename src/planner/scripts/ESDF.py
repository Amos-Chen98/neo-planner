'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2023-02-12 17:16:34
'''

import numpy as np
from scipy import ndimage


class ESDF():
    def __init__(self):
        pass

    def occupancy_map_cb(self, map):
        '''
        Subscribe to the OccupancyGrid and convert it to the ESDF map
        visualize the ESDF map in RViz
        '''
        occupancy_raw = map.data  # occupancy_raw is a 1D array
        self.map_resolution = map.info.resolution
        self.map_width = map.info.width
        self.map_origin = map.info.origin.position  # the origin of the map is the left-bottom corner
        self.map_height = map.info.height

        # binarize occupancy_raw
        occupancy_raw = tuple(1 if x == 100 else 0 for x in occupancy_raw)  # 0-free, 1-occupied, treat unknown as free

        # get an np.array of the occupancy map
        self.occupancy_2d = np.array(occupancy_raw).reshape(self.map_height, self.map_width)  # row-major order: row - height, column - width

        # get the ESDF map, in distance_transform_edt(), 0 is treated as occupied, so use 1-occupancy_2d
        self.esdf_map = ndimage.distance_transform_edt(1 - self.occupancy_2d) * self.map_resolution  # size: map_height * map_width

        # get the ESDF gradient map
        # grad_x = grad along x axis in map = grad along col in matrix, so y (row) first
        self.esdf_grad_y, self.esdf_grad_x = np.gradient(self.esdf_map)

    def is_occuiped(self, pos):
        '''
        input: x, y in map frame
        return True if the cell is occupied
        '''
        x = pos[0]
        y = pos[1]
        # get the real index of the cell
        row_index = int((y - self.map_origin.y) / self.map_resolution)  # row index = y in map frame
        col_index = int((x - self.map_origin.x) / self.map_resolution)  # column index = x in map frame
        if row_index < 0 or row_index >= self.map_height or col_index < 0 or col_index >= self.map_width:
            return False
        else:
            return self.occupancy_2d[row_index, col_index]

    def get_edt_dis(self, pos):
        '''
        input: x, y in map frame
        return the distance to the nearest obstacle
        '''
        x = pos[0]
        y = pos[1]
        # get the real index of the cell
        row_index = int((y - self.map_origin.y) / self.map_resolution)  # row index = y in map frame
        col_index = int((x - self.map_origin.x) / self.map_resolution)  # column index = x in map frame
        # if index out of range, return a large number
        if row_index < 0 or row_index >= self.map_height or col_index < 0 or col_index >= self.map_width:
            return 10000
        else:
            return self.esdf_map[row_index, col_index]

    def get_edt_grad(self, pos):
        '''
        input: x, y in map frame
        return the gradient of the distance to the nearest obstacle
        '''
        x = pos[0]
        y = pos[1]
        # get the real index of the cell
        row_index = int((y - self.map_origin.y) / self.map_resolution)
        col_index = int((x - self.map_origin.x) / self.map_resolution)
        if row_index < 0 or row_index >= self.map_height or col_index < 0 or col_index >= self.map_width:
            return [0, 0]
        else:
            return [self.esdf_grad_x[row_index, col_index], self.esdf_grad_y[row_index, col_index]]
