# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 00:21:13 2020

@author: laure
"""
import pandas as pd
import numpy as np
from sklearn import neighbors
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

data = pd.read_csv("C:/Users/laure/OneDrive - Deakin University/SIT384/task6-resources/task6_1_dataset.csv", index_col=0);
# create an array in column, so there are 2 columns
x_train = np.c_[data['x1'], data['x2']]
y_train = data['y']

#define K and create kNN classifier
k = 1
knn = neighbors.KNeighborsClassifier(k)
knn.fit(x_train,y_train)

fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
cmap_bold = ListedColormap(['green', 'blue', 'magenta'])  #green map to 0  blue:1  magenta:2
#plot the classification of x_train, c is color
ax.scatter(x_train[:, 0], x_train[:, 1], cmap=cmap_bold, c=y_train, alpha=0.6, s=75)

#create an input to predict its y value
x_test = [[-4, 8]]
y_pred = knn.predict(x_test)

colors=['green', 'blue', 'magenta']
ax.scatter(x_test[0][0], x_test[0][1], marker="x", lw=2, s=120, c=colors[y_pred.astype(int)[0]])
plt.text(-4, 8, ' (-4,8) test point')  # test label in specific position
ax.set_title("3-Class classification (k = {})\n the test point is predicted as class {}".format(k, colors[y_pred.astype(int)[0]]))

#define K and create kNN classifier
k = 15
knn = neighbors.KNeighborsClassifier(k)
knn.fit(x_train, y_train)

# get min and max of x1 x2
x1_min, x1_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
x2_min, x2_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
h = 0.05  # step size in the mesh
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max,h), np.arange(x2_min, x2_max,h))
Z = knn.predict(np.c_[xx1.ravel(), xx2.ravel()])
Z = Z.reshape(xx1.shape)

fig,ax = plt.subplots(figsize=(7, 5), dpi=100)
# Create colour maps
cmap_light = ListedColormap(['#AAFFAA', '#AAAAFF', '#FFAAAA'])
# plot the decision boundaries
ax.pcolormesh(xx1, xx2, Z, cmap=cmap_light)
# plot the training points
ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cmap_bold, alpha=0.6, s=75)
plt.xlim(xx1.min(), xx1.max())
plt.ylim(xx2.min(), xx2.max())

x_test = [[-2, 5]]
y_pred = knn.predict(x_test)
ax.scatter(x_test[0][0], x_test[0][1], marker="x", lw=2, s=120, color=colors[y_pred.astype(int)[0]])
plt.text(-2, 5, ' (-2,5) test point')
ax.set_title("3-Class classification (k = {})\n the test point is predicted as class {}".format(k, colors[y_pred.astype(int)[0]]))