# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 09:29:25 2020

@author: lehrs
"""

import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import pickle

# this should become a data bunch, could use sales, and gis_parcel_points too
(llx, lly, urx, ury) = 438000,1554000,743000,1853000
pickled_polygon_filename = 'c:\\temp\\ma305\\pickled_zip_polygons.sav'
pickled_polygon_dictionary_filename = 'c:\\temp\\ma305\\pickled_zip_dict.sav'
zip_code_file = "c:\\temp\\ma305\\volusia_zip_codes2.png"

polygons = pickle.load(open(pickled_polygon_filename, 'rb'))
zip_polygon_dict = pickle.load(open(pickled_polygon_dictionary_filename, 'rb'))

fig, ax = plt.subplots()
fig.set_size_inches(10,10)
zip_img = plt.imread(zip_code_file)
plt.imshow(zip_img, alpha=.8, extent=[llx, urx, lly, ury])

transparency = .6
p = PatchCollection(polygons, cmap=plt.cm.jet, alpha=transparency)

color_values = 1000000*np.random.rand(len(polygons))
#p.set_array(color_values)

# Get Color Values from your data frame analysis
# need to build this array of color_values
# length of polygons
# build this array from the polygon zip code dictionary/list
zip_list['32759'] = 300000
color_values2 = np.zeros(len(polygons))
for dzip, i in zip_polygon_dict.items():
    print(i, dzip)
    if dzip in zip_list.keys():
        color_values2[i] = zip_list[dzip]
    else:
        color_values2[i] = 0
 
p.set_array(color_values2)
p.set_clim(vmin=min(color_values2[color_values2>0]), vmax=max(color_values2))

ax.add_collection(p)

# https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
fig.colorbar(p,ax=ax, fraction=0.044, pad=0.04, alpha=transparency)
plt.show()