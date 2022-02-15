# -*- coding: utf-8 -*-
"""
Created on Thu May 14 16:11:13 2020

@author: laure
"""
import pandas as pd
from sklearn.cluster import KMeans 
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.cluster import AgglomerativeClustering  
from sklearn.cluster import DBSCAN

data = pd.read_csv("C:/Users/laure/OneDrive - Deakin University/SIT384/8.3HD-resources/Complex8_N15.csv");
X = data.iloc[:,1:3]
y = data.iloc[:,3]

# fine KMeans models
algorithms = ["auto", "full"]
n_inits=range(10,20) # n_init: Number of time the k-means algorithm will be run with different centroid seeds.
n_clusters = range(8,20) #15 12
kmeans_results=[]
maxAccuracy=0
maxPred = np.zeros(len(y));
for n_init in n_inits:
    for n_cluster in n_clusters:
        k_means = KMeans(n_init=n_init,n_clusters=n_cluster).fit(X)  # define KMeans model train KMeans model
        y_pred=k_means.predict(X)       
        cluster_accuracy = np.mean(y_pred ==y)*100
        if cluster_accuracy > maxAccuracy:
            maxPred = y_pred
        kmeans_results.append({  # use dictionary to store results
       'KMeans n_inits':n_init,
       'n_clusters':n_cluster,
       'cluster accuracy':cluster_accuracy
       })
df=DataFrame(kmeans_results)
print(df)
fig,ax = plt.subplots(figsize=(7,7), dpi=100)
cmap = plt.cm.get_cmap("Spectral")
ax.scatter(X['V1'], X['V2'], c=maxPred, alpha=0.25, s=60,linewidth=0,cmap=cmap)
ax.set_title('KMeans predict clusters')


# fine agglomerative clustering models
linkages = ['ward', 'average'] 
n_clusters = range(5,20)
ac_results=[]
maxAccuracy=0
maxPred = np.zeros(len(y));
for linkage in linkages:
    for n_cluster in n_clusters:
        agglom = AgglomerativeClustering(n_clusters =n_cluster, linkage = linkage)
        y_pred=agglom.fit_predict(X) # fit and predict models
        cluster_accuracy = np.mean(y_pred ==y)*100
        if cluster_accuracy > maxAccuracy:
            maxPred = y_pred
        ac_results.append({  # use dictionary to store results
       'Agglomerative linkage':linkage,
       'n_clusters':n_cluster,
       'cluster accuracy':cluster_accuracy
       })
df2=DataFrame(ac_results)
print(df2)
fig2,ax2=plt.subplots(figsize=(7, 7), dpi=100)
ax2.scatter(X['V1'], X['V2'], c=maxPred, alpha=0.25, s=60,linewidth=0,cmap=cmap)     
ax2.set_title('Agglomerative predict clusters')

# fine agglomerative clustering models
#epss = [40,80,100]
#min_sampless = [50,60,70]
#14 7
epss = range(10,20)
min_sampless=range(5,10)
db_results=[]
maxAccuracy=0
maxPred = np.zeros(len(y));
for eps in epss:
    for min_samples in min_sampless:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        y_pred = dbscan.fit_predict(X)
        cluster_accuracy = np.mean(y_pred ==y)*100
        if cluster_accuracy > maxAccuracy:
            maxPred = y_pred
        db_results.append({  # use dictionary to store results
       'DBSCAN eps':eps,
       'min_samples':min_samples,
       'cluster accuracy':cluster_accuracy
       })
df3=DataFrame(db_results)
print(df3)
fig3,ax3=plt.subplots(figsize=(7, 7), dpi=100)
ax3.scatter(X['V1'], X['V2'], c=maxPred, alpha=0.25, s=60,linewidth=0,cmap=cmap)     
ax3.set_title('DBSCAN predict clusters')

# plot original y clusters
fig4,ax4=plt.subplots(figsize=(7, 7), dpi=100)
ax4.scatter(X['V1'], X['V2'], c=y, alpha=0.25, s=60,linewidth=0,cmap=cmap)     
ax4.set_title('Original y clusters')
