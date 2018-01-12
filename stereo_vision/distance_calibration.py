import numpy as np
import matplotlib.pyplot as plt

# Measured - Calculated
"""
# 120deg cameras
data = np.asarray([[2933, 2510],
                     [3780, 3130],
                     [4769, 3800],
                     [5647, 4290],
                     [7070, 5020],
                     [8208, 5660],
                     [9366, 6190],
                     [10756, 6660],
                     [12048, 7190],
                     [13096, 7610],
                     [14302, 7830],
                     [15229, 8320],
                     [16273, 8590]])

data = np.asarray([[3188, 2720],
                      [4890, 3860],
                      [6261, 4670],
                      [8335, 5660],
                      [10094, 6490],
                      [12290, 7190],
                      [14861, 8070],
                      [17412, 8870],
                      [20144, 9510],
                      [22545, 9860],
                      [24888, 10200]])

# 60deg cameras, measured distance - disparity
data = np.asarray([[21096, 48],
                   [19728, 50],
                   [18613, 52],
                   [17620, 52],
                 [16504, 53],
                 [15200, 55],
                 [13867, 59],
                 [12715, 61],
                 [11305, 64],
                 [10060, 68],
                 [8873, 73],
                 [7544, 80],
                 [6516, 87],
                 [5244, 102],
                 [3876, 127],
                 [2934, 159],
                 [2238, 202],
                 ])
"""

# 60deg cameras, measured distance - reprojected distance
data = np.asarray([[21096, 5440],
                 [19728, 5220],
                 [18613, 5020],
                 [17620, 5020],
                 [16504, 4920],
                 [15200, 4740],
                 [13867, 4580],
                 [12715, 4280],
                 [11305, 4080],
                 [10060, 3840],
                 [8873,  3570],
                 [7544,  3260],
                 [6516,  3000],
                 [5244,  2560],
                 [3876,  2050],
                 [2934,  1660],
                 [2238,  1300],
                 ])

print(data)

x = np.asarray(data[:,1])
y = np.asarray(data[:,0])
x1 = 1/x

p = np.polyfit(x, y, 4)
print(p)

"""
r = np.polyval(p, data[:,1]) - data[:,0]
print(r)
"""

ideal = np.linspace(1000, 6000, num=100)
ideal1 = 1/ideal
l = np.polyval(p, ideal)


fig = plt.figure(1)
plt.plot(data[:,1], data[:,0], '.')
plt.plot(ideal, l)
plt.xlabel('Re-projected Z from disparity map (mm)')
plt.ylabel('Laser measurement (mm)')
plt.grid(1)
# plt.hist(r)
plt.show()
