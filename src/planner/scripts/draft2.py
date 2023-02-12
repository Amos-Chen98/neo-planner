from scipy import ndimage
import numpy as np
# a = np.array(([0,1,1,1,1],
#               [0,0,1,1,1],
#               [0,1,1,1,1],
#               [0,1,1,1,0],
#               [0,1,1,0,0]))
# distance_map = ndimage.distance_transform_edt(a)
# print(distance_map)

# # gradient_x, gradient_y = np.gradient(distance_map)

# # print(gradient_x)

# # gradient_magnitude = np.sqrt(np.power(gradient_x, 2) + np.power(gradient_y, 2))

# # # print(gradient_magnitude)

# # test_point = np.array([2,2])
# # print(distance_map[test_point[0], test_point[1]])
# # print(gradient_x[test_point[0], test_point[1]])
# # print(gradient_y[test_point[0], test_point[1]])

# # # x = np.array([1, 2, 4, 7, 11, 16], dtype=float)
# # mat = np.array([[1, 2, 4, 7, 11, 16],
# #                 [4, 4, 6, 2, 4, 9],
# #                 [5, 3, 4, 3, 5, 2]])
# # grad_x, grad_y = np.gradient(mat, edge_order=1)
# # print(grad_x)
# # print(grad_y)
# # # 注意：这里的梯度是沿着行和列方向的，所以是先行后列
vel_array = np.array([1,2,3])
vel_max = np.max(vel_array)
vel_nor = vel_array/vel_max*256
print(vel_nor)
# convert vel_nor to int
vel_nor = vel_nor.astype(int)
print(vel_nor)