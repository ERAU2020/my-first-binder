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

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
x = cancer.data[:,0]
y = cancer.target     # 0 malignant (red) 1 benign (blue)
colors = {0:'red', 1:'blue'}
plt.scatter(x,y,facecolors='none', edgecolors=pd.DataFrame(cancer.target)[0].apply(lambda x: colors[x]), cmap=colors)
plt.xlabel("mean radius")
plt.ylabel("Result")

red   = mpatches.Patch(color='red',   label='malignant')
blue  = mpatches.Patch(color='blue',  label='benign')

plt.legend(handles=[red, blue], loc=1)
#plt.show()

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


#---print trained model intercept---
print('0-malignant, 1-benign')
print(log_regress.intercept_)
#---print trained model coefficients---
print(log_regress.coef_)
