# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 20:23:45 2020

@author: lehrs
"""

# chapter 7 Classification Logistic Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn import linear_model
import math

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
x = cancer.data[:,0]
y = cancer.target     # 0 malignant (red) 1 benign (blue)
colors = {0:'red', 1:'blue'}
red   = mpatches.Patch(color='red',   label='0: malignant')
blue  = mpatches.Patch(color='blue',  label='1: benign')


plt.scatter(x,y,facecolors='none', edgecolors=pd.DataFrame(cancer.target)[0].apply(lambda x: colors[x]), cmap=colors)
plt.xlabel("mean radius")
plt.ylabel("Result")
plt.legend(handles=[red, blue], loc=1)
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
plt.plot([log_regress.intercept_, log_regress.intercept_], [0, 1], 'b')
plt.show()

plt.scatter(x,y,facecolors='none', edgecolors=pd.DataFrame(cancer.target)[0].apply(lambda x: colors[x]), cmap=colors)
plt.xlabel("mean radius")
plt.ylabel("Result")
plt.legend(handles=[red, blue], loc=1)
plt.plot(x1,y1)
plt.show()



radius = 20
print('For Radius: %.2f' % radius)
print(log_regress.predict_proba([[radius]])) 
print(log_regress.predict([[radius]])[0])    

radius = 17
print('For Radius: %.2f' % radius)
print(log_regress.predict_proba([[radius]])) 
print(log_regress.predict([[radius]])[0])    

radius = 14
print('For Radius: %.2f' % radius)
print(log_regress.predict_proba([[radius]])) 
print(log_regress.predict([[radius]])[0])    

radius = 11
print('For Radius: %.2f' % radius)
print(log_regress.predict_proba([[radius]])) 
print(log_regress.predict([[radius]])[0])    


radius = log_regress.intercept_[0]
print('For Radius: %.2f' % radius)
print(log_regress.predict_proba([[radius]])) 
print(log_regress.predict([[radius]])[0])    



#---print trained model intercept---
print('0-malignant, 1-benign')
print(log_regress.intercept_)
#---print trained model coefficients---
print(log_regress.coef_)


# generate prediction over the range of radius
max_radius = np.max(x)
# list(range(math.ceil(np.max(x))+1))
# looks like they pick 60 40 probability as tipping point (15.1)
# where one class dominates the prediction % 60/40 flip to right
# threshold is overwhelmed 60/40 1.5
# though this trips as 15
print('radius predicted_class [probability class 0, probability class 1]')
for rad in range(math.ceil(max_radius)+1):
    p0, p1 = log_regress.predict_proba([[rad]])[0]
    predicted_class = log_regress.predict([[rad]])[0]
    actual_class = np.nan
    print('%d\t%d [%.4f, %.4f]' % (rad, predicted_class, p0, p1))
    

# raw data profile
# load data into a dataframe
# sort based on radius
# grab average value at that index (loop where in range of index)
# how many at each class at that index (display)
# dominate class at that value (display)
# could make a dictionary object of this and print out or utilize later..
# num_records at whole number level i.e. radius=15  >=15 && <=15+1
# num_records_class_0
# num_records_class_1
# dominate_class = class_0 if num_records_class_0 >= num_records_class_1 else dominate_class= class_1
# print the table like I did above  


# see page 124
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['CLASS'] = cancer.target
min_actual_radius = math.floor(np.min(df['mean radius']))
max_actual_radius = np.max(df['mean radius'])
print()
print('Radius NumRecords DominateClass (#class0, #class1)')
for rad in range(min_actual_radius, math.ceil(max_actual_radius)+1):
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
    
    #actual_class = np.nan
    print('%3d %3d %d (%2d, %2d)' % (rad, num_records, dominate_class, num_records_class_0, num_records_class_1))
  
