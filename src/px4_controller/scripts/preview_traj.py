'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2023-02-07 20:13:03
'''
import os
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_path)
from math import sqrt
from math import sin
from math import cos
from math import pi
import matplotlib.pyplot as plt
import numpy as np


class State():
    def __init__(self):
        self.p_x = 0.0
        self.p_y = 0.0
        self.p_z = 1.0

        self.v_x = 0.0
        self.v_y = 0.0
        self.v_z = 0.0

        self.a_x = 0.0
        self.a_y = 0.0
        self.a_z = 0.0

        self.yaw = 0.0

    def get_state(self, t):
        self.p_x = (60*sqrt(2)*cos(0.1*t) + 20*cos(0.2*t) + 60) / \
            (2*sqrt(2)*cos(0.1*t) + 3)

        self.p_y = (20*sqrt(2)*sin(0.1*t) + 20*sin(0.2*t)) / \
            (2*sqrt(2)*cos(0.1*t) + 3)

        self.p_z = 10.0

        self.v_x = (-6.0*sqrt(2)*sin(0.1*t) - 4.0*sin(0.2*t))/(2*sqrt(2)*cos(0.1*t) + 3) + 0.2*sqrt(
            2)*(60*sqrt(2)*cos(0.1*t) + 20*cos(0.2*t) + 60)*sin(0.1*t)/(2*sqrt(2)*cos(0.1*t) + 3)**2

        self.v_y = 0.2*sqrt(2)*(20*sqrt(2)*sin(0.1*t) + 20*sin(0.2*t))*sin(0.1*t)/(2*sqrt(2)*cos(
            0.1*t) + 3)**2 + (2.0*sqrt(2)*cos(0.1*t) + 4.0*cos(0.2*t))/(2*sqrt(2)*cos(0.1*t) + 3)

        self.v_z = 0.0

        self.a_x = 0.4*sqrt(2)*(-6.0*sqrt(2)*sin(0.1*t) - 4.0*sin(0.2*t))*sin(0.1*t)/(2*sqrt(2)*cos(0.1*t) + 3)**2 + (-0.6*sqrt(2)*cos(0.1*t) - 0.8*cos(0.2*t))/(2*sqrt(2)*cos(0.1*t) + 3) + 0.02*sqrt(
            2)*(60*sqrt(2)*cos(0.1*t) + 20*cos(0.2*t) + 60)*cos(0.1*t)/(2*sqrt(2)*cos(0.1*t) + 3)**2 + 0.16*(60*sqrt(2)*cos(0.1*t) + 20*cos(0.2*t) + 60)*sin(0.1*t)**2/(2*sqrt(2)*cos(0.1*t) + 3)**3

        self.a_y = (-0.2*sqrt(2)*sin(0.1*t) - 0.8*sin(0.2*t))/(2*sqrt(2)*cos(0.1*t) + 3) + 0.02*sqrt(2)*(20*sqrt(2)*sin(0.1*t) + 20*sin(0.2*t))*cos(0.1*t)/(2*sqrt(2)*cos(0.1*t) + 3)**2 + 0.16 * \
            (20*sqrt(2)*sin(0.1*t) + 20*sin(0.2*t))*sin(0.1*t)**2/(2*sqrt(2)*cos(0.1*t) + 3)**3 + 0.4 * \
            sqrt(2)*(2.0*sqrt(2)*cos(0.1*t) + 4.0*cos(0.2*t)) * \
            sin(0.1*t)/(2*sqrt(2)*cos(0.1*t) + 3)**2

        self.a_z = 0.0


state = State()
t_samples = np.arange(23,2*pi*10+23, 0.1)

p_x = []
p_y = []
v_x = []
v_y = []
a_x = []
a_y = []

for i in range(len(t_samples)):
    t = t_samples[i]
    state.get_state(t)
    p_x.append(state.p_x)
    p_y.append(state.p_y)
    v_x.append(state.v_x)
    v_y.append(state.v_y)
    a_x.append(state.a_x)
    a_y.append(state.a_y)

t_samples = t_samples.tolist()

# Plot results
plt.figure()
plt.scatter(p_x, p_y)
plt.grid()
plt.axis('equal')

plt.figure()
plt.plot(t_samples, p_x, label='Pos_x')
plt.plot(t_samples, p_y, label='Pos_y')
plt.plot(t_samples, v_x, label='Vel_x')
plt.plot(t_samples, v_y, label='Vel_y')
plt.plot(t_samples, a_x, label='Acc_x')
plt.plot(t_samples, a_y, label='Acc_y')
plt.xlabel('t/s')
plt.legend()
plt.grid()
plt.show()
