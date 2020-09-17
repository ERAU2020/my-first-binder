# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 07:26:47 2020

@author: lehrs
"""

# pip install scikit-learn
# pip install scipy

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import pandas as pd

df = pd.read_csv("http://www.mosaic-web.org/go/datasets/galton.csv")
print(df.head())
print(df.describe())

mu = df.height.mean()
sigma = df.height.std()   #  3.58291846997281
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma))
# add a mean line, from y = 0, y=pdf(x=mean)
# ok this means...
# stats.norm.pdf(mu, mu, sigma) is the y value along the pdf
# here's where jupyter is very useful to show building on the algorithm

#plt.plot()

print('mean height ', mu)
print('my height 74 inches: ', stats.norm.pdf(74, mu, sigma))
print('+/- 1 std: ', mu - sigma, mu + sigma)

n, bins, patches = plt.hist(df.height, 50, density=1, facecolor='green', alpha=0.75)
plt.show()
