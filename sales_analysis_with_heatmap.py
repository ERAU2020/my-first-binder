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

corr = df.corr()
print(corr)

# plot the correlation matrix in a color coded format
# see chapter 12
import seaborn as sns
sns.heatmap(df.corr(),annot=True)
#---get a reference to the current figure and set its size---
fig = plt.gcf()
fig.set_size_inches(10,10)
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

# this is the spatial bounding box of volusia county in FL State Plane East Feet CRS
(llx, lly, urx, ury) = 438000,1554000,743000,1853000
xbins = np.linspace(llx, urx, 10)
ybins = np.linspace(ury, lly, 10)

# digitize will place values in 0 which is below or upto first element
# digitize will place value in nbins+1 if its larger then last bin
# digitize places value between elements, i.e. return 1 if > first element &
# .... less then second element... and so on...

cols = np.digitize(df_left['gis_x'], xbins)
rows = np.digitize(df_left['gis_y'], ybins)
#np.digitize(440000, xbins)
# append these to the df_left
df_left['ROW']=rows
df_left['COL']=cols

bs = df_left[(df_left['ROW'] == 8) & (df_left['COL'] == 8)]
bs.saleprice.mean()
fig = plt.gcf()
fig.set_size_inches(10,10)

plt.plot(df_left['gis_x'], df_left['gis_y'], 'b.')
# plot cell grid lines
for x in (xbins):
    # plot vertical lines
    plt.plot([x,x], [lly,ury], 'k')
for y in (ybins):
    # plot horizontal lines
    plt.plot([llx,urx], [y,y], 'k')

# create a list of lookups for ROW,COL,INDEX (using a list is recommended for performance)
lookups = [[0,0,0]]
index = 1
for y in range(len(ybins)-1):  # note row 1 is on top of grid, note how we ordered bins above
    for x in range(len(xbins)-1):
        # print('%d  (%d, %d)' % (index, y+1, x+1))
        # print('%d, %d, %d' % (y+1, x+1, index))
        lookups.append([y+1, x+1, index])
        # cant do this have to do the above... and 
        # df_left[(df_left['ROW']==x+1) & (df_left['COL']==y+1)].CELL=index
        text = plt.text((xbins[x]+xbins[x+1])/2, (ybins[y]+ybins[y+1])/2, index,
                       ha="center", va="center", color="r")
        index = index + 1
        
plt.axis('Equal')
plt.show()

# convert the lookups list to data frame
df_lookups = pd.DataFrame(lookups, columns=['ROW','COL','CELL'])
# perform a multi column merge, adding the CELL field to the df_left dataframe to match the grid
df_left = pd.merge(df_left, df_lookups, how='left', left_on=['ROW', 'COL'], right_on=['ROW', 'COL'])

df_left[df_left['CELL']==25]['saleprice'].mean()
df_left[df_left['ROW']==1]['gis_y'].mean()

