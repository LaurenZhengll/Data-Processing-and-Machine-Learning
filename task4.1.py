# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 22:48:45 2020

@author: laure
"""

import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt


file = pd.read_csv("C:/Users/laure/OneDrive - Deakin University/SIT384/task4/data.csv", encoding = "ISO-8859-1", usecols=[1,2,3,4,5]);
df = DataFrame(data = file);
# use shape to get rows and cols. rows = 4, cols = 6 including index
rows, cows = df.shape; 
# axis=1 means sum in horizontal direction
df["Total"] = df.apply(lambda x:x[0] + x[1] + x[2] + x[3] + x[4], axis=1);

#create a plot with specific size, return figure and axises
fig, ax = plt.subplots(figsize = (7, 5), dpi = 100);
colors =  ['blue', 'red', 'green', 'yellow'];
# x is discrete
x = np.arange(rows);
# get y, only array is available in bar
array = np.zeros(rows);
i = 0
for i in range(0, 4):
    array[i] = df.iloc[i, 5];

# draw the bar chart by ax.bar()
ax.bar(x, array, align = 'center', color = colors);

# set x ticks
ax.set_xticks(x);
# set labels of x ticks and rotation of the labels is 90
labels = ['Cyber incident', 'Theft of paperwork or data storage device', 'Rogue employee', 'Social engineering / impersonation'];
ax.set_xticklabels(labels, rotation = 90);

# set x y labels and title
ax.set_xlabel("Attack type");
ax.set_ylabel("Number of attacks per attacj type");
ax.set_title("Number of malicious or criminal attack July-December-2019");

# default axis=0 means sum in vertical direction, x is columns
df.loc['Total per sector'] = df.apply(lambda x: x.sum())
fig, ax = plt.subplots(figsize = (10, 10))
# array2 get numbers of attack per sectors
array2 = np.zeros(5);
j = 0
for j in range(0, 5):
    array2[j] = df.iloc[4, j];
#default colors = ['blue', 'orange','green','red','purple']
labels = ['Health service providers', 'Finance', 'Education', 'Legal,accounting & management services', 'Personal services']
ax.pie(array2, labels = labels, autopct='%1.1f')