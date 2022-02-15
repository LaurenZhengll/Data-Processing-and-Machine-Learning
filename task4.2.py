# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 13:52:48 2020

@author: laure
"""

import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt

file = pd.read_csv("C:/Users/laure/OneDrive - Deakin University/SIT384/task4/Malicious_or_criminal_attacks_breakdown-Top_five_industry_sectors_July-Dec-2019.csv", encoding = "ISO-8859-1", index_col=0, engine='python');
df = DataFrame(data = file);
# use shape to get rows and cols. rows = 4, cols = 5
rows, cols = df.shape; 

#create two plots in a row with specific size, return figure and axises
fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(14,5), dpi=100);
colors=['red', 'yellow', 'blue', 'green']
labels = ['Cyber incident', 'Theft of paperwork or data storage device', 'Rogue employee', 'Social engineering / impersonation'];
x=np.arange(cols)
width = 0.2
w2 = 0.4
#df.values transfer df to array
array=df.values;

for i in range(rows):
    # x+i*width is drawing group bar, width is the width of a bar
    ax[0].bar(x+i*width, array[i,:], width= width, align='center', label=labels[i], color=colors[i])
# set bottom to draw stack bar
ax[1].bar(x,array[0,:],width=w2, align='center',label=labels[0],color=colors[0])
ax[1].bar(x,array[1,:],width=w2, bottom=array[0,:], align='center',label=labels[1],color=colors[1])
ax[1].bar(x,array[2,:],width=w2, bottom=array[0,:]+array[1,:], align='center',label=labels[2],color=colors[2])
ax[1].bar(x,array[3,:],width=w2, bottom=array[0,:]+array[1,:]+array[2,:], align='center',label=labels[3],color=colors[3])

# set x ticks
ax[0].set_xticks(x);
# set labels of x ticks and rotation of the labels is 90
ax[0].set_xticklabels(['Health service providers', 'Finance', 'Education', 'Legal,accounting & management services', 'Personal services'], rotation = 90);
# set x y labels and title
ax[0].set_xlabel("The top five industry sectors");
ax[0].set_ylabel("Number of attack");
ax[0].set_title("Type of attack by top five industry sectors");
# legend
ax[0].legend()
# set x ticks
ax[1].set_xticks(x);
# set labels of x ticks and rotation of the labels is 90
ax[1].set_xticklabels(['Health service providers', 'Finance', 'Education', 'Legal,accounting & management services', 'Personal services'], rotation = 90);
# set x y labels and title
ax[1].set_xlabel("The top five industry sectors");
ax[1].set_ylabel("Number of attack");
ax[1].set_title("Type of attack by top five industry sectors");
# legend  legend(loc=0) put the legend to the middle top of the chart
ax[1].legend()
