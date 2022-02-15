# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 14:39:04 2020

@author: laure
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model

data = pd.read_csv("C:/Users/laure/OneDrive - Deakin University/SIT384/5.2C-resources/admission_predict.csv", index_col=0);
# split into train dataset and test dataset
split = 0.75
split_index = int(np.round(split * len(data)))
train_data=data[0:split_index-1]
test_data=data[split_index:len(data)]

# prepare x and y for train data and test data of GRE
GRE_train_X = train_data['GRE Score'].values
GRE_train_X = np.c_[GRE_train_X]
train_Y1 = train_data['Chance of Admit'].tolist()
GRE_test_X = test_data['GRE Score'].values
GRE_test_X = np.c_[GRE_test_X]
test_Y1 = test_data['Chance of Admit'].tolist()

# prepare x and y for train data and test data of GPA
GPA_train_X = train_data['CGPA'].values
GPA_train_X = np.c_[GPA_train_X]
train_Y2 = train_data['Chance of Admit'].tolist()
GPA_test_X = test_data['CGPA'].values
GPA_test_X = np.c_[GPA_test_X]
test_Y2 = test_data['Chance of Admit'].tolist()

# create a linear regression model to fit GRE train data
lr1 = linear_model.LinearRegression()
lr1.fit(GRE_train_X, train_Y1)
# create a linear regression model to fit GPA train data
lr2 = linear_model.LinearRegression()
lr2.fit(GPA_train_X, train_Y2)

fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=100)
# plot scatter charts for real x y in GRE train dataset. set x and x_label
train_data.plot(kind='scatter', x='GRE Score', y='Chance of Admit', ax=axs[0,0])
axs[0,0].set_title('Linear regression with GRE score and chance of admit')
train_data.plot(kind='scatter', x='CGPA', y='Chance of Admit', color='orange',ax=axs[1,0])
axs[1,0].set_title('Linear regression with CGPA score and chance of admit')

# get predict y for GRE train set. plot the regression line
train_Y1_predic = lr1.predict(GRE_train_X)
axs[0,0].plot(GRE_train_X,train_Y1_predic)
train_Y2_predic = lr2.predict(GPA_train_X)
axs[1,0].plot(GPA_train_X,train_Y2_predic)

# plot the predicted points along the prediction line
test_Y1_predic = lr1.predict(GRE_test_X)
scales = 30*np.ones(len(GRE_test_X))
axs[0,1].scatter(GRE_test_X,test_Y1_predic,s=scales,color='r',edgecolor='none') #predicted points
axs[0,1].plot(GRE_test_X,test_Y1_predic,color='b',linewidth=1) #prediction line
axs[0,1].set_title('GRE score VS change of admit: true value and residual')
axs[0,1].set_xlabel('GRE score')
axs[0,1].set_ylabel('chance of admit')
# plot the true values
axs[0,1].scatter(GRE_test_X,test_Y1,s=scales,color='r',edgecolor='none') #test_Y1: true value of change of admit to GRE test
# plot the residual line
res = np.reshape(GRE_test_X,[1,len(GRE_test_X)])[0]  # get all the GRE_test_X from the test dataset
res_x = []
res_y = []
for i in range(len(GRE_test_X)): #for each x in GRE test set
    res_x = np.append(res_x,res[i]) #get x coordinate
    res_y = np.append(res_y,test_Y1_predic[i]) #get predicted y in GRE test set
    res_x = np.append(res_x,res[i]) #get x coordinate again
    res_y = np.append(res_y,test_Y1[i]) #get test y coordinate
    axs[0,1].plot(res_x,res_y,color='red',linewidth=0.5) #draw the vertical residual line (x,test_Y1_predic), (x,test_Y1)
    res_x = []
    res_y = []
    
# plot the predicted points along the prediction line
test_Y2_predic = lr2.predict(GPA_test_X)
scales = 30*np.ones(len(GPA_test_X))
axs[1,1].scatter(GPA_test_X,test_Y2_predic,s=scales,color='g',edgecolor='none') #predicted points
axs[1,1].plot(GPA_test_X,test_Y2_predic,color='b',linewidth=1) #prediction line
axs[1,1].set_title('CGPA score VS change of admit: true value and residual')
axs[1,1].set_xlabel('CGPA score')
axs[1,1].set_ylabel('chance of admit')
# plot the true values
axs[1,1].scatter(GPA_test_X,test_Y2,s=scales,color='g',edgecolor='none') #test_Y1: true value of change of admit to GRE test
# plot the residual line
res2 = np.reshape(GPA_test_X,[1,len(GPA_test_X)])[0]  # get all the GPA_test_X from the test dataset
res2_x = []
res2_y = []
for i in range(len(GPA_test_X)): #for each x in GRE test set
    res2_x = np.append(res2_x,res2[i]) #get x coordinate
    res2_y = np.append(res2_y,test_Y2_predic[i]) #get predicted y in GPA test set
    res2_x = np.append(res2_x,res2[i]) #get x coordinate again
    res2_y = np.append(res2_y,test_Y2[i]) #get test y coordinate, the real y in GPA test set
    axs[1,1].plot(res2_x,res2_y,color='r',linewidth=0.5) #draw the vertical residual line (x,test_Y1_predic), (x,test_Y1)
    res2_x = []
    res2_y = []