# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 19:30:15 2020

@author: laure
"""
import pandas
import numpy as np


file = pandas.read_csv("C:/Users/laure/OneDrive - Deakin University/SIT384/3.1-resources/result.csv", header = None, skiprows = 1, dtype = float);

# np.zeros() creates an array; file.shape[1] returns column number of the file, not including index; file[1] is the assin1 column
arrayTotal = np.zeros(len(file[file.shape[1]-1]));
# store a column in an array
i = 0;
for x in file[file.shape[1]-1]:  
    arrayTotal[i] = x;
    i += 1;
# numpy.ndarray object has no attribute 'index', so transfer into list; list.index(value) returns value's correspending index
index = arrayTotal.tolist().index(max(arrayTotal));
print("Total max:", max(arrayTotal), ", min:", min(arrayTotal), ", average: ", sum(arrayTotal)/len(file[1]));  

#arrayHigh store scores from assin1 to exam with specific index
array = np.zeros(len(file[1]));
arrayHigh = np.zeros(file.shape[1]-1);
m = 0;
# from assin1 to exam
for i in range(1, file.shape[1] - 1): 
    j = 0;   
    for x in file[i]:  
        array[j] = x;
        j += 1;
    arrayHigh[m] = array[index];
    m += 1;
    if i < 5:
        print("assin", i, " max:", max(array), ", min:", min(array), ", average:", sum(array)/len(file[1]));
    else:
        print("Exam max:", max(array), ", min:", min(array), ", average:", sum(array)/len(file[1]));
print("Student with highest total:\n", "ID:",index + 1, "assin1:", arrayHigh[0], " assin2:", arrayHigh[1], "assign3:", arrayHigh[2], " assign4:",arrayHigh[3], "exam:", arrayHigh[4], "total:", max(arrayTotal));