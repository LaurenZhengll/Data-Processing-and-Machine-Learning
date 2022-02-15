# -*- coding: utf-8 -*-
"""
Created on Wed May  6 13:19:19 2020

@author: laure
"""
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import AgglomerativeClustering  
from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix 

np.random.seed(0)
# make random clusters. n_samples: The total number of points equally divided among clusters.
# X: Feature matrix, array [n_samples, n_features]
# y: The integer labels of each sample.
X, y = make_blobs(n_samples=200, centers=[[3,2], [6, 4], [10, 5]], cluster_std=0.9)
plt.scatter(X[:, 0], X[:, 1]) # plot created dataset

# n_init: Number of time the k-means algorithm will be run with different centroid seeds. 
k_means = KMeans(init="k-means++", n_clusters=3, n_init=12)
k_means.fit(X)
k_means_labels = k_means.labels_
k_means_cluster_centers = k_means.cluster_centers_

fig = plt.figure(figsize=(6, 4))
# Colors use a color map, the number of color is the number of labels. 
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels)))) # Use set(k_means_labels) to get unique labels.
# Create a plot with a black background
ax = fig.add_subplot(1, 1, 1, facecolor = 'black')
# k ranges from 0 1 2
for k, col in zip(range(len([[3,2], [6, 4], [10, 5]])), colors):
    # data points in the cluster (ex. cluster 0) are labeled as true, else they are labeled as false.
    my_members = (k_means_labels == k)   
    # Plots the datapoints with color col. 'w'- white
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    
    # Define cluster center
    cluster_center = k_means_cluster_centers[k] 
    # Plots the centroids. 'o' circle marker; 'k' - black
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)
ax.set_title('KMeans') #set title
ax.set_xticks(()) # Remove x-axis ticks
ax.set_yticks(()) # Remove y-axis ticks
plt.show() # Show the plot

agglom = AgglomerativeClustering(n_clusters = 3, linkage = 'average')
agglom.fit(X,y)
#plt.figure(figsize=(6,4))
fig2,ax2=plt.subplots(figsize=(6,4))
# scale the scattered very far data points down
x_min, x_max = np.min(X, axis=0), np.max(X, axis=0) # Create a minimum and maximum range of X
X = (X - x_min) / (x_max - x_min) # Get the average distance for X
cmap = plt.cm.get_cmap("Spectral")
# The loop displays all data points
for i in range(X.shape[0]):        
    ax2.text(X[i, 0], X[i, 1], str(y[i]), color=cmap(agglom.labels_[i] / 10.), fontdict={'weight': 'bold', 'size': 9})
ax2.set_title('Agglomerative Hierarchical') #set title
ax2.set_xticks(()) # Remove x-axis ticks
ax2.set_yticks(()) # Remove y-axis ticks

dist_matrix = distance_matrix(X,X)  #calculate distance matrix
print(dist_matrix)
condensed_dist_matrix= hierarchy.distance.pdist(X,'euclidean')
Z = hierarchy.linkage(condensed_dist_matrix, 'complete') # display dendrogram
dendro = hierarchy.dendrogram(Z)


