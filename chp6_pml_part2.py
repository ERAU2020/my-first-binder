# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 04:14:16 2020

@author: lehrs
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

df = pd.read_csv('.\\PML\\polynomial.csv')
plt.scatter(df.x,df.y)
plt.show()
#plt.plot(df.x, df.y, 'b*')

model = LinearRegression()

x = df.x[0:6, np.newaxis]     #---convert to 2D array---
y = df.y[0:6, np.newaxis]     #---convert to 2D array---

# or...
# x = np.array(df.x).reshape(-1,1)


model.fit(x,y)

#---perform prediction---
y_pred = model.predict(x)

#---plot the training points---
plt.scatter(x, y, s=10, color='b')

#---plot the straight line---
plt.plot(x, y_pred, color='r')
plt.show()

#---calculate R-squared---
print('R-squared for training set: %.4f' % model.score(x,y))
print(model.intercept_)
print(model.coef_)

# end of page 137

# https://scikit-learn.org/stable/modules/classes.html

from sklearn.preprocessing import PolynomialFeatures
degree = 2

polynomial_features = PolynomialFeatures(degree = degree)
x_poly = polynomial_features.fit_transform(x)
print(x_poly)
print(x)

model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)

#---plot the points---
plt.scatter(x, y, s=10)

#---plot the regression line---
plt.plot(x, y_poly_pred)
plt.show()
print(model.intercept_)
print(model.coef_)

print('R-squared for training set: %.4f' % model.score(x_poly,y))

#http://polynomialregression.drque.net/math.html
