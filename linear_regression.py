# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:36:53 2020

@author: lehrs
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# import pandas_profiling
import statsmodels.api as sm
from sklearn.metrics import r2_score


# initialize x as a range of values from -10 to 10
x = np.arange(-10,10+1,1)  # x = -10:10
y = 0.7*x -2




x.shape=-1,1
y.shape=-1,1

plt.plot(x, y,'k.')

#create and fit the model
model = LinearRegression()
model.fit(x,y)

#plot the regression line
plt.plot(x, model.predict(x), color='r')
plt.plot(0,0, 'b.')
plt.title('y=0.7x - 2')
plt.xlabel('x fixed -10..10')
plt.ylabel('y f(x)')
plt.grid(True)
plt.show()

print('MODEL:\n y = ' + str(model.coef_) + "* x +" + str(model.intercept_))
print('R-squared: ' + str(r2_score(y, model.predict(x))))

print(sm.OLS(y, sm.add_constant(x)).fit().summary())


# recall some numpy array functions
print("One row of data", x)
print('Sum is', x.sum())
print('Max is', x.max())
print('Min is', x.min())
print('Std is', x.std())
print('Count is', x.size)
print('Shape is', x.shape)



# compute the Linear Regression Values with numpy manually
#  https://www.statisticshowto.datasciencecentral.com/probability-and-statistics/regression-analysis/find-a-linear-regression-equation/


a = ( y.sum() * np.array(x*x).sum() - x.sum()*np.array(x*y).sum() ) / ( x.size*np.array(x*x).sum() - x.sum()*x.sum())

b = (x.size*np.array(x*y).sum() -  x.sum() * y.sum()) / ( x.size*np.array(x*x).sum() - x.sum()*x.sum())

print("y=", b, "*x + ", a)
model.score(x,y)

# write fix for Correlation Coefficient ->
# https://www.statisticshowto.datasciencecentral.com/probability-and-statistics/correlation-coefficient-formula/
# compute R like I did ax+b above using the numpy arrays...



