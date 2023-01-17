'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2023-01-17 23:19:03
'''
from sympy import *


t = symbols('t')
scale = 0.1
amp = 20
a = scale*t
# using Lemniscate_of_Bernoulli, ref: https://zhuanlan.zhihu.com/p/448214022
p_x = amp*(3*sqrt(2)*cos(a) + cos(2*a) + 3) / \
    (2*sqrt(2)*cos(a) + 3)

p_y = amp*(sqrt(2)*sin(a) + sin(2*a)) / \
    (2*sqrt(2)*cos(a) + 3)

v_x = diff(p_x, t)
v_y = diff(p_y, t)
a_x = diff(v_x, t)
a_y = diff(v_y, t)

print('p_x:')
print(p_x)

print('p_y：')
print(p_y)

print('v_x：')
print(v_x)

print('v_y：')
print(v_y)

print('a_x：')
print(a_x)

print('a_y：')
print(a_y)
