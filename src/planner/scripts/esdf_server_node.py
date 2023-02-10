#! /usr/bin/env python
import os
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_path)
import time
import rospy
from esdf_server import ESDF

if __name__ == "__main__":
    esdf = ESDF()

    # time.sleep(1)

    # test_pos = [3.4,0.8]
    # time_start = time.time()
    # occupied = esdf.is_occuiped(test_pos)
    # dis = esdf.get_edt_dis(test_pos)
    # grad = esdf.get_edt_grad(test_pos)

    # print("test_pos: ", test_pos, "")
    # print("occupied: ", occupied)
    # print("distance: ", dis)
    # print("gradient: ", grad)

    # time_end = time.time()
    # print("time cost of query: ", time_end - time_start)

    rospy.spin()
