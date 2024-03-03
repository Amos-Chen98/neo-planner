'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2023-03-02 11:05:22
'''

import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid
from scipy import ndimage


class ESDFVis():

    def __init__(self, node_name="esdf_vis"):
        # Node
        rospy.init_node(node_name, anonymous=False)

        # Subscriber
        self.occupancy_map_sub = rospy.Subscriber('/projected_map', OccupancyGrid, self.show_esdf)

        # Publisher
        self.esdf_map_pub = rospy.Publisher('/esdf_map', OccupancyGrid, queue_size=10)

    def show_esdf(self, map):
        '''
        Get the occupancy map and convert it to the ESDF map
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

        # normalize the esdf_map to 0-100 for visualization, no acctual meaning
        esdf_map_show = 100 - (self.esdf_map - np.min(self.esdf_map)) / (np.max(self.esdf_map) - np.min(self.esdf_map)) * 100

        # visualize the map in Rviz
        esdf_map_msg = OccupancyGrid()
        esdf_map_msg.header.frame_id = "map"
        esdf_map_msg.info.resolution = self.map_resolution
        esdf_map_msg.info.width = self.map_width
        esdf_map_msg.info.height = self.map_height
        esdf_map_msg.info.origin = map.info.origin
        esdf_map_msg.data = tuple(int(x) for x in esdf_map_show.reshape(-1))  # convert to 1D array
        self.esdf_map_pub.publish(esdf_map_msg)


if __name__ == "__main__":
    esdf_vis = ESDFVis()

    rospy.spin()
