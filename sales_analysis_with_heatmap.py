# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 11:55:33 2020

@author: lehrs
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# downloaded curl -OL https://wx.erau.edu/faculty/lehrs/ma305/sales_data.zip
# extract somewhere, mine is here...
file_name = 'c:\\temp\\ma305\\sales_data.sav'

df = pd.read_pickle(file_name)

# plot the correlation matrix in a color coded format
corr = df.corr()
print(corr)

import seaborn as sns
sns.heatmap(df.corr(),annot=True)
#---get a reference to the current figure and set its size---
fig = plt.gcf()
fig.set_size_inches(12,12)
plt.show()
# next lets load GIS Data Frame
# see announcement 10/27/2020
# curl -OL http://wx.erau.edu/faculty/lehrs/ma305/volusia_gis_data.zip
gis_location = "c:\\temp\\ma305\\volusia\\volusia_gis_data.txt"
gis_df =  pd.read_csv(gis_location,sep='\t',header=(0))
gis_df.tail()

# join the data frame
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.join.html
df_left = pd.merge(df, gis_df, how='left', left_on='parid', right_on='parid')

(llx, lly, urx, ury) = 438000,1554000,743000,1853000
xbins = np.linspace(llx, urx, 10)
ybins = np.linspace(ury, lly, 10)

cols = np.digitize(df_left['gis_x'], xbins)
rows = np.digitize(df_left['gis_y'], ybins)

# append these to the df_left
df_left['ROW']=rows
df_left['COL']=cols


bs = df_left[(df_left['ROW'] == 6) & (df_left['COL'] == 7)]
bs = df_left[(df_left['COL']==2)]
bs.saleprice.mean()

plt.plot(df_left['gis_x'], df_left['gis_y'], 'b.')
# plot cell grid lines
for x in (xbins):
    # plot vertical lines
    plt.plot([x,x], [lly,ury], 'k')
for y in (ybins):
    # plot horizontal lines
    plt.plot([llx,urx], [y,y], 'k')

index = 1
for y in range(len(ybins)-1):
    for x in range(len(xbins)-1):
        print('%d  (%d, %d)' % (index, y, x))
        text = plt.text((xbins[x]+xbins[x+1])/2, (ybins[y]+ybins[y+1])/2, index,
                       ha="center", va="center", color="r")
        index = index + 1

plt.axis('Equal')
plt.show()



# add a cell # in the boxes

