# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 12:48:08 2020

@author: lehrs
"""

import numpy as np
import matplotlib.pyplot as plt

x = 5
y = 8
y_hold = y;
x_hold = x;

r = np.sqrt(x**2 + y**2)
print("Utilizing the point: (%.2f, %.2f)" % (x,y))
print("radius is %.2f" % r)

plt.title('Radius Vector')
plt.ylabel('y')
plt.xlabel('x')

#plt.plot([0, 8], [0,5], 'k')# plot, x,y with points as black .'s (note the k is for black, blue is b, red is r)
plt.plot([0, x], [0,y], 'k')# plot, x,y with points as black .'s (note the k is for black, blue is b, red is r)

#plt.axis([1.5,1.85,50,90])      # limit the domain of the x axis and range of y axis
plt.grid(True)                  # python supports Boolean values - True and False, pass in True to flip grid on
print("Plot Points")
while y >= 0:
    plt.plot(x,y,'b*')
    print("(%.2f, %.2f)" % (x,y))
    y = y - 1
    x = np.sqrt(r**2 - y**2)
    
# or
y = y_hold
x = x_hold
x_values = np.array([])
y_values = np.array([])
while y >= 0:
    x_values = np.append(x_values, x)
    y_values = np.append(y_values, y)
    y = y - 1
    x = np.sqrt(r**2 - y**2)

plt.plot(x_values,y_values,'r.')
plt.plot(x_values,y_values,'g')    
plt.axes().set_aspect('equal', 'datalim')
plt.show()

    