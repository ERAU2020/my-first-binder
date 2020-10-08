# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 03:23:48 2020

@author: lehrs
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas_profiling

from sklearn.datasets import load_boston
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html
dataset = load_boston()
print(dataset.feature_names)
print(dataset.DESCR)
print(dataset.target)  # Median Value

# load data set into a pandas dataframe
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df.head(30)

df['MEDV'] = dataset.target
df.head()

# run these in the console..show what's produced, che
# demonstrate how the correlation function works, and that it produces another dataframe
cor = df.corr()
cor = df.corr().abs()
print(df.corr().abs().nlargest(3, 'MEDV').index)

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(18,15))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df['LSTAT'],
           df['RM'],
           df['MEDV'],
           c='b')

ax.set_xlabel("LSTAT")
ax.set_ylabel("RM")
ax.set_zlabel("MEDV")
plt.show()

# run from command shell
# python chp6_pml.py

# Training the model...
x = pd.DataFrame(np.c_[df['LSTAT'], df['RM']], columns = ['LSTAT','RM'])
Y = df['MEDV']

# run in console..column stack...
# (not super useful)
# https://numpy.org/doc/stable/reference/generated/numpy.c_.html 
# https://numpy.org/doc/stable/reference/generated/numpy.column_stack.html#numpy.column_stack
td = np.c_[df['LSTAT'], df['RM']]

from sklearn.model_selection import train_test_split
x_train, x_test, Y_train, Y_test = train_test_split(x, Y, test_size = 0.3,
                                                    random_state=5)

print(x_train.shape)
print(Y_train.shape)
print(x_test.shape)
print(Y_test.shape)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train, Y_train)

price_pred = model.predict(x_test)
print('R-squared: %.4f' % model.score(x_test, Y_test))

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(Y_test, price_pred)
print(mse)

plt.scatter(Y_test, price_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted prices")
plt.title("Actual prices vs Predicted prices")

print(model.intercept_)
print(model.coef_)
print(model.predict([[30,5]]))




fig = plt.figure(figsize=(18,15))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x['LSTAT'],
           x['RM'],
           Y,
           c='b')

ax.set_xlabel("LSTAT")
ax.set_ylabel("RM")
ax.set_zlabel("MEDV")

#---create a meshgrid of all the values for LSTAT and RM---
x_surf = np.arange(0, 40, 1)   #---for LSTAT---
y_surf = np.arange(0, 10, 1)   #---for RM---
x_surf, y_surf = np.meshgrid(x_surf, y_surf)
# https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
# https://stackoverflow.com/questions/36013063/what-is-the-purpose-of-meshgrid-in-python-numpy

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x, Y)

#---calculate z(MEDC) based on the model---
z = lambda x,y: (model.intercept_ + model.coef_[0] * x + model.coef_[1] * y)

# https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
ax.plot_surface(x_surf, y_surf, z(x_surf,y_surf),
                rstride=1,
                cstride=1,
                color='None',
                alpha = 0.4)

plt.show()





