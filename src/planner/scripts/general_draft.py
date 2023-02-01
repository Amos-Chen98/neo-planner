'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2023-01-30 22:37:07
'''
t = (1,2,3)
print(type(t))
delim = ','
# s = ''.join(map(str, t))
s = delim.join(map(str, t))
# s = delim.join(t)
print(s)