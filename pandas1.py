# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 11:25:05 2020

@author: lehrs
"""
# 
# %reset -f
import matplotlib.pyplot as plt
import pandas as pd
import os.path
import os as sys
from pathlib import Path
sys.tracebacklimit = 0

my_ma305_folder =  "c:\\temp\\ma305\\PML\\"
local_csv_filename = my_ma305_folder + "NaNDataset.csv"
# local_csv_filename = my_ma305_folder + "drivinglicense.csv"

# https://stackoverflow.com/questions/82831/how-do-i-check-whether-a-file-exists-without-exceptions
myfile = Path(local_csv_filename)

# alternatively
if myfile.exists() == True:
    print('file %s exists' % local_csv_filename)
else:
    print('File %s does not exist.  Must exist...exiting.\n' % local_csv_filename)
    # raise ValueError(1)


try:
    mfile2 = myfile.resolve(strict=True)
except FileNotFoundError:
    print("file %s does not exist..exiting\n" % local_csv_filename)
    raise ValueError()
    print("Not here -- already exited...\n") 
else:
    # exists  # this code is not necessary.... PASS
    print("sweet")


user_response=input('is this a MAC or UNIX ? Y or N -->')
if user_response == 'Y' or user_response == 'y':
    print('can\'t open notepad this way.  Need path to other editor.....\n')
    raise ValueError(3)
else:
    # print('here')
    df = pd.read_csv(local_csv_filename)
    command = 'notepad.exe ' + local_csv_filename
    sys.system(command)


df.head(7)
df.tail(10)
len(df)

df.describe()  # prints each column and statistics
df.values      # data from array without indexes

print(df)
df.values.shape
# make some for loops to print out df.values data

# grab column 2
x = df.index.values  # grab the unique, seqential indexes store for x values
y = df.values[:,0]   #note col2 is a numpy array 1 -D
plt.plot(x,y)
plt.grid(True)
plt.plot(x, df['C'], 'y*')
#plt.plot(df.index, df['C'], 'g.')
plt.show()

print(y.shape)
# its a list right now
# make it rows cols
# make it cols rows
# make it 2x3
# make it 3x2
# make it 1 x figure it out)


# useful information
index = df.index
columns = df.columns
values = df.values


