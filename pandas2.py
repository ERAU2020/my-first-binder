# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 05:23:45 2020

@author: lehrs
"""

#%reset -f
# %cls
import pandas as pd
import numpy as np

def outliers_iqr(data):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    return np.where((data > upper_bound) | (data < lower_bound))

# hmm ... what's np.percentiles return ??
# https://numpy.org/doc/stable/reference/generated/numpy.percentile.html
# hmm... what does np.where() return ?
# https://numpy.org/doc/stable/reference/generated/numpy.where.html
# having to break it apart to understand, the function call returned a tuple
# first element of the tuple is the numpy array hence the [0]
# now i is each element in the array
    
def outliers_z_score(data):
    threshold = 3
    mean = np.mean(data)
    std = np.std(data)
    z_scores = [(y - mean) / std for y in data]
    # that's pretty complex, what the heck....break it down...
    # we are using someone else's code, what did they do? what can we learn?
    return np.where(np.abs(z_scores) > threshold)


df = pd.read_csv("http://www.mosaic-web.org/go/datasets/galton.csv")
print(df.head())
print(df.describe())

print("Outliers using outliers_iqr()")
print("=============================")
for i in outliers_iqr(df.height)[0]:
    print(i)
    print(df[i:i+1])


print("Outliers using outliers_z_score()")
print("=================================")
for i in outliers_z_score(df.height)[0]:
    print(df[i:i+1])
print()



iqr = q3 - q1
lower_bound = q1 - (iqr * 1.5)
upper_bound = q3 + (iqr * 1.5)