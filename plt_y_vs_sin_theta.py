# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 21:13:13 2020

@author: lehrs
"""

import numpy as np
import matplotlib.pyplot as plt


theta = np.arange(0,360,1)
y1    = np.sin(np.deg2rad(theta))
y2    = np.cos(np.deg2rad(theta))
y3    = y1**2 + y2**2

plt.plot(theta,  y1, 'b.')   # y = f(sin(theta))
plt.plot(theta, y2, 'g.')   # y2 = f(cos(theta))
#plt.plot(theta, y3, 'k.')   # sin^2 + cos^2 should = 1 
plt.grid(True)
plt.show()
# add legend
# make angle ticks every 30 degree, every 45, every 15..

plt.plot(y2, y1, 'b.')
plt.plot(0,0, 'b.')
# plt.axes().set_aspect('equal', 'datalim')
plt.axis("equal")   # see page 82 in text
plt.grid(True)
plt.show()
# from orgin to point is the slope m
# tangent line is -1/m

# pick a value compute x, y, angle, rnd(360)

