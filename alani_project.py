# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 17:09:35 2020

@author: lehrs
"""
# import libraries used
import numpy as np
import matplotlib.pyplot as plt
import time

sleepee = 0.8;
radius = 6;
area   = np.pi * np.power(radius, 2);

# how to plot a circle?

# alternatives?

# use Jupyter Notebook

# use an IPyWidget to implement the code below without screen flutter

# start a github at github.com
# for now:  github.com/ERAU2020/myfirstproject/alani_project.py
# get another person to join it

# circumference

# draw a * around the circumference, and refresh the screen show 

#print(theta, x, y, radius)
# for theta in (range(0,361, 10))
#if theta==0: x=radius;
# x = radius * np.cos(np.degrees2radians(theta))
# y = radius * np.sin(n)
theta = np.arange(0,360+1,10)
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
  time.sleep(sleepee)
  
  
#  sleep


