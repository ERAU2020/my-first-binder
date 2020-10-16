# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 17:09:35 2020

@author: lehrs
"""
# import libraries used
import numpy as np
import matplotlib.pyplot as plt
import time

sleepee = 2;
radius = 6;
area   = np.pi * np.power(radius, 2);
# circumference
# how to plot a circle?
# alternatives?
# use Jupyter Notebook
# use an IPyWidget to implement the code below without screen flutter
# start a github at github.com
# for now:  https://github.com/ERAU2020/my-first-binder   alani_project.py
# get another person to join it
# draw a * around the circumference, and refresh the screen show 

theta = np.arange(0,360+1,15)
y = radius * np.sin(np.deg2rad(theta))
x = radius * np.cos(np.deg2rad(theta))
print(np.c_[theta, x, y])

for i in range(len(x)):
  # print(np.c_[theta, x, y])
  plt.plot([-1.5*radius, 1.5*radius], [0, 0])
  plt.plot( [0, 0], [-1.5*radius, 1.5*radius])
  plt.plot(x,y, 'g*')
  plt.axis('Equal')
  plt.plot([0,x[i]],[0,y[i]], 'k')
  plt.plot([0,x[i]],[0,y[i]], 'k*')
  plt.show()
  print('Theta: %.2f Degrees, %.3f Radians  (%.3f, %.3f)' % (theta[i], np.deg2rad(theta[i]), x[i], y[i]))
  time.sleep(sleepee) 
  
  
#  sleep


