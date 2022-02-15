# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 01:15:27 2020

@author: laure
"""

import pandas as pd
from pandas import DataFrame

# read file and store in DataFrame structure
file = pd.read_csv("C:/Users/laure/OneDrive - Deakin University/SIT384/task3-resources/result_withoutTotal.csv");
df = DataFrame(data = file);

# apply fuction to specific columns or rows[[]] and the result is in a new column; 
# lambda x, function parameter x is current row; axis=1 means operating in horizontal direction, axis=0 means operating in vertical direction
df["Total"]=df[["Ass1","Ass2","Ass3","Ass4","Exam"]].apply(lambda x: 0.05 * (x["Ass1"] + x["Ass3"]) + 0.15 * (x["Ass2"] + x["Ass4"]) + 0.6 * x["Exam"], axis=1);
#df["Total"]>100 gets rows   ,"Total": set these rows' Total to 100
df.loc[df["Total"]>100, "Total"]=100;

# df["Total"] is the Total column; round() gets the nearest integar
df["Final"] = df["Total"].map(lambda x: round(x));
# for two conditions, use ()&()
df.loc[(df["Final"]>=44) & (df["Exam"]<48),"Final"]=44;

# set Grade
df.loc[df["Final"]<=49.45,"Grade"]='N';
df.loc[(df["Final"]>49.45) & (df["Final"]<=59.45),"Grade"]='P';
df.loc[(df["Final"]>59.45) & (df["Final"]<=69.45),"Grade"]='C';
df.loc[(df["Final"]>69.45) & (df["Final"]<=79.45),"Grade"]='D';
df.loc[df["Final"]>79.45,"Grade"]='HD';

#df.to_csv('./feature.csv', columns=[0, 1, 2, 3], header=False, index = False) means save specific columns to csv without header and index.
df.to_csv("C:/Users/laure/OneDrive - Deakin University/SIT384/task3-resources/result_updated.csv", index=False);
df.loc[df["Exam"]<48].to_csv("C:/Users/laure/OneDrive - Deakin University/SIT384/task3-resources/failedhurdle.csv", index=False);

print("Result with 3 new columns:\n",df);
print("\nStudents with exam score < 48:\n",df.loc[df["Exam"]<48]);
print("\nStudents with exam score > 100:\n",df.loc[df["Exam"]>100]);

