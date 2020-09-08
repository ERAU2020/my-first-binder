# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 13:54:18 2020

@author: lehrs
"""
import numpy as np
import matplotlib.pyplot as plt
import sys

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
r = 55
theta = 0
print("theta cos(theta) sin(theta) x   y")
sys.tracebacklimit = 0

plt.title('Radius Vector')
plt.ylabel('y')
plt.xlabel('x')
plt.plot([-r, r], [0,0], 'k')# plot, x,y with points as black .'s (note the k is for black, blue is b, red is r)
plt.plot([0,0], [-r, r], 'k')# plot, x,y with points as black .'s (note the k is for black, blue is b, red is r)
#plt.plot([0, 8], [0,5], 'k')# plot, x,y with points as black .'s (note the k is for black, blue is b, red is r)
#plt.axis([1.5,1.85,50,90])      # limit the domain of the x axis and range of y axis
            
plt.axis([-0.25, r*1.1, -0.25, r*1.1])

while theta <= 90:
    y = np.sin(np.deg2rad(theta))
    x = np.cos(np.deg2rad(theta))
    #print(theta, x, y)
    print("%.2f\t%.2f\t%.2f\t%.2f\t%.2f" % (theta, x, y, r*x, r*y))
    plt.plot(x*r,y*r, 'b*')
    theta = theta + 10

plt.grid(True)
plt.axes().set_aspect('equal', 'datalim')
plt.show()