# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:35:33 2020

@author: lehrs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn import linear_model
import math

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
initial_class = cancer.target  # target data stored in an inverted manner 1 is benign 0 malignant, switch!!!!!
actual_class = []
  
for i in range(len(initial_class)):
    if initial_class[i]==0: 
        new_class = 1
        c_str = 'ro'
    else:
        new_class = 0;
        c_str = 'b.'
    # print(i, initial_class[i], new_class)
    radius = df.at[i, 'mean radius']
    plt.plot(radius, new_class, c_str)
    actual_class = np.append(actual_class, new_class)

#plt.plot([18,18], [0, 1], 'k')
#plt.plot([11,11], [0, 1], 'k')
plt.xlabel("mean radius")
plt.ylabel("actual classification ")
plt.show()



x = cancer.data[:,0]
y = actual_class  # 0 benign blue, 1 malignant red
# 
# y = cancer.target     # 0 malignant (red) 1 benign (blue)
colors = {0:'blue', 1:'red'}
blue  = mpatches.Patch(color='blue',  label='0: benign')
red   = mpatches.Patch(color='red',   label='1: malignant')


plt.scatter(x,y,facecolors='none', edgecolors=pd.DataFrame(y)[0].apply(lambda x: colors[x]), cmap=colors)
plt.xlabel("mean radius")
plt.ylabel("actual classification")
plt.legend(handles=[blue,red], loc=1)
plt.axis([0,30,-0.2, 1.2])
plt.show()

# end of page 161

# page 162
log_regress = linear_model.LogisticRegression()
# train model
log_regress.fit(X = np.array(x).reshape(len(x),1), y = y)

#---print trained model intercept---
print(log_regress.intercept_)
#---print trained model coefficients---
print(log_regress.coef_)

# plot the sigmoid curve..
def sigmoid(x):
    return (1 / (1 +
        np.exp(-(log_regress.intercept_[0] +
        (log_regress.coef_[0][0] * x)))))

x1 = np.arange(0, 30, 0.01)
y1 = [sigmoid(n) for n in x1]
plt.plot(x1,y1)
plt.plot([-log_regress.intercept_, -log_regress.intercept_], [0, 1], 'b')
plt.show()

plt.scatter(x,y,facecolors='none', edgecolors=pd.DataFrame(y)[0].apply(lambda x: colors[x]), cmap=colors)
plt.xlabel("mean radius")
plt.ylabel("actual classification")
plt.legend(handles=[red, blue], loc=1)
plt.plot(x1,y1)
plt.show()

# make some predictions....
radius = 11
print('For Radius: %.2f: Predicted class: %d  Probabilities [%.4f, %.4f]' % 
      (radius, log_regress.predict([[radius]])[0], log_regress.predict_proba([[radius]])[0][0], log_regress.predict_proba([[radius]])[0][1]))   

radius = 14
print('For Radius: %.2f: Predicted class: %d  Probabilities [%.4f, %.4f]' % 
      (radius, log_regress.predict([[radius]])[0], log_regress.predict_proba([[radius]])[0][0], log_regress.predict_proba([[radius]])[0][1]))   

radius = -log_regress.intercept_[0]
print('For Radius: %.2f: Predicted class: %d  Probabilities [%.4f, %.4f]' % 
      (radius, log_regress.predict([[radius]])[0], log_regress.predict_proba([[radius]])[0][0], log_regress.predict_proba([[radius]])[0][1]))   

radius = 17
print('For Radius: %.2f: Predicted class: %d  Probabilities [%.4f, %.4f]' % 
      (radius, log_regress.predict([[radius]])[0], log_regress.predict_proba([[radius]])[0][0], log_regress.predict_proba([[radius]])[0][1]))   

radius = 20
print('For Radius: %.2f: Predicted class: %d  Probabilities [%.4f, %.4f]' % 
      (radius, log_regress.predict([[radius]])[0], log_regress.predict_proba([[radius]])[0][0], log_regress.predict_proba([[radius]])[0][1]))   




#---print trained model intercept---
print('0-benign, 1-malignant')
print(log_regress.intercept_)
#---print trained model coefficients---
print(log_regress.coef_)


#### now look at the probabilities as we increase the radius..
# generate prediction over the range of radius
min_radius = math.floor(np.min(x))
max_radius = math.ceil(np.max(x))
# list(range(math.ceil(np.max(x))+1))

    
#######  Lets take another look at the actual data....
# see page 124
df['CLASS'] = actual_class

print('\nActual Data...')
print('Examine all records between df.radius >= radius and df.radius <radius+1')
print('Radius NumRecords DominateClass (#class0, #class1) [prob c0, prob c1]')
for rad in range(min_radius-1, max_radius+1):
    l1 = x >=rad     # find indexes of x where the the value is >= radius
    l2 = x <rad+1    # find indexes of x where the value is < radius + 1
    num_records = len(x[l1 & l2])
    # need to use the dataframe, not X so we can get the raw classes
    
    l1 = df['mean radius'] >= rad     # boolean array
    l2 = df['mean radius'] < rad+1    # boolean array
    num_records = len(df[l1 & l2])    # and the two booleans lists count # true occurences
    #c0 = c1 = 0
    l1l = df[l1 & l2].CLASS==0        # from the list where radius meet criteria, check those that have class=0
    l2l = df[l1 & l2].CLASS==1
    num_records_class_0 = len(l1l[l1l])  # count number of occurences where list is true
    num_records_class_1 = len(l2l[l2l])
    dominate_class = 0
    if num_records_class_1 > num_records_class_0:
        dominate_class = 1
        
    pc0 = np.nan
    pc1 = np.nan
    
    if num_records > 0:
        pc0 = num_records_class_0 / num_records
        pc1 = num_records_class_1 / num_records
    
    #actual_class = np.nan
    print('>=%3d %3d %d (%2d, %2d) [%.3f, %.3f]' % (rad, num_records, dominate_class, num_records_class_0, num_records_class_1, pc0, pc1))
    
print('Notice from the data above in the region of radius 10..18 we have mixing')

# looks like they pick 60 40 probability as tipping point ~ about 15.1
# where one class dominates the prediction % 60/40 flip to right
# threshold is overwhelmed 60/40 1.5
#---print trained model intercept---
print('\n\nNow examine predictions based on the radius\'s over the range of values')
print('0-benign, 1-malignant')
print(log_regress.intercept_)
#---print trained model coefficients---
print(log_regress.coef_)

print('radius predicted_class [probability class 0, probability class 1]')
for rad in range(min_radius-1, max_radius+1):
    p0, p1 = log_regress.predict_proba([[rad]])[0]
    predicted_class = log_regress.predict([[rad]])[0]
    print('%d\t%d [%.4f, %.4f]' % (rad, predicted_class, p0, p1))


# ok build the contingency table...  count # of occurences of each to measure
# cut off's, and the trade off's
# netflix predicts you would like this movie.... not a huge deal if there wrong
# here we are making a model to determine whether a patient has cancer or not...


# 569 records  len(df)
# 357 benign cases 0    len(df[df['CLASS']==0])
# 212 malignant cases 1  len(df[df['CLASS']==1])
# benign = 63%
# malignant  = 37%
# have to becareful of class in balance....
# just tell everyone they are benign and you'll be correct 63% of the time based on the data...

# compute the contingency table
#   actual classes
#   0  |    1

# versus predicted classes 0 | 1

# notice a 2x2 matrix arises

tn = 0     # true negative, predict benign or class 0, actual data is benign class 0 (0,0)
fn = 0     # false negative, predict benign class 0, actual data is malignant class 1 (0, 1)
fp = 0     # false positive, predict malignant class, actual data is benign (1, 0)
tp = 0     # true positive, predict malignant class 1, actual data is malignant class 1 (1,1)

# note those coordinates of the contingency matrix (0,0)
# ROWS:  Predicted values, since there are two classes the possible predictions are 0, 1
# COLS:  Actual values, again two classes, 0 1
# first row of the contingency matrix
# hence the matrix 0,0  predict 0 actual 0   true negative -- note negative is false like the value 0
# and 0,1 predict 0, actual 1, predict false, actually positive -- hence false negative
# next row of the contingency matrix
# 1,0  predict 1 actual 0, false positive.  
# 1,1 predict 1, actual 1, true positive

# Build the matrix, loop over all the data, make the predictions, compare to the actual, update the counts
actual_class = np.nan
predicted_class = np.nan


# display the matrix
print('\nContingency Matrix')
print('TN=%3d\tFN=%3d' % (tn, fn))
print('FP=%3d\tTN=%3d' % (fp, tp))
# compute the metrics


# this contingency data is used to compute precision, accuracy, and recall measures
