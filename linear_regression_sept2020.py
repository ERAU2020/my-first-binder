# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 14:14:44 2020

@author: lehrs
"""
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# represents the heights of a group of people in metres
heights = [[1.6], [1.65], [1.7], [1.73], [1.8]]

# represents the weights of a group of people in kgs
weights = [[60], [65], [72.3], [75], [80]]

plt.title('Weights plotted against heights')
plt.xlabel('Heights in metres')
plt.ylabel('Weights in kilograms')

plt.plot(heights, weights, 'k.')

# axis range for x and y
plt.axis([1.5, 1.85, 50, 90])
plt.grid(True)

# Create and fit the model
model = LinearRegression()
model.fit(X=heights, y=weights)

plt.show()

# make a prediction, expects multidimension array
# make a single prediction
a1 = model.predict([[1.75]])
a1[0,0] # comes back as a multi-dimensional array first row, first column [0][0] or [0,0]
a1[0][0]
# Out[25]: 76.0387

# plot the regression line
extreme_heights = [[0], [1.8]]
extreme_weights = model.predict(extreme_heights)
plt.plot(extreme_heights, extreme_weights, 'b*')

print(model.intercept_[0])
print(np.round(model.intercept_[0], 2))

print(model.coef_)
print(model.coef_[0])
print(model.coef_[0][0])
print(np.round(model.coef_[0][0], 2))

pw = model.predict(heights)      # compute predicted weights from the model
plt.plot(heights, weights, 'b*')
plt.plot(heights, pw, 'k.')
plt.plot(heights, pw, 'r')
plt.show()

# bottom of page 104 Residual Sum of Squares
# verify this old school way
weights - pw
((weights - pw)**2)
np.sum((weights-pw)**2)

mu = np.mean(weights)
print('Mean weight %.3f' % mu)
dw_sum = 0;
tss = 0;
for i in range(len(weights)):
    dw =  weights[i][0]-pw[i][0]
    dw_squared = dw**2
    dw_sum = dw_sum + dw_squared
    
    var = weights[i] - mu
    var_squared = var**2
    tss = tss + var_squared
    print('%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f' % (weights[i][0], pw[i][0], dw, dw_squared, var, var_squared))

print('residual sum is %.3f' % dw_sum)
print('total sum is %.3f' % tss)
print('R Squared %.4f' % (1 - dw_sum/tss))


print('Residual sum of squares: %.2f' %
       np.sum((weights - model.predict(heights)) ** 2))

# RSS should be small as possible

# test data
heights_test = [[1.58], [1.62], [1.69], [1.76], [1.82]]
weights_test = [[58], [63], [72], [73], [85]]

# Total Sum of Squares (TSS)
weights_test_mean = np.mean(np.ravel(weights_test))
TSS = np.sum((np.ravel(weights_test) -
              weights_test_mean) ** 2)
print("TSS: %.2f" % TSS)

# Residual Sum of Squares (RSS)
RSS = np.sum((np.ravel(weights_test) -
              np.ravel(model.predict(heights_test)))
                 ** 2)
print("RSS: %.2f" % RSS)

# R_squared
R_squared = 1 - (RSS / TSS)
print("R-squared: %.2f" % R_squared)

# using scikit-learn to calculate r-squared
print('R-squared: %.4f' % model.score(heights_test,
                                      weights_test))


