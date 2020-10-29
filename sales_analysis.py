# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 13:01:03 2020

@author: lehrs
"""

import pandas as pd
# downloaded curl -OL https://wx.erau.edu/faculty/lehrs/ma305/sales_data.zip
# extract somewhere, mine is here...

file_name = 'c:\\temp\\ma305\\sales_data.sav'

df = pd.read_pickle(file_name)

corr = df.corr()
print(corr)

# how many records in the data frame
# describe
# summary
# pandas profiling report
# correlation

df[df['zip1']=='32114']
# what are the zip codes ?
df['zip1'].unique()

# how many zip codes are there?
len(df['zip1'].unique())

#how many of each are there ?
df['zip1'].value_counts()

# what to do with None or nans
df['zip1'].value_counts(dropna=False)

s = df['zip1'].value_counts()

# note the command above made a pandas Series
s.values
s.index


# series have an index and values you can sort by either using the correct command
s.sort_values()
s.sort_index()

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.at.html

# what is number of buildings sold, average price, with 3 bed rooms per zip code?
#zipcode, #sales, avg sale_price
