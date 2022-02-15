# -*- coding: utf-8 -*-
"""
Created on Mon May 11 16:44:16 2020

@author: laure
"""
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# prepare data for modeling
cancer = load_breast_cancer()
X=cancer.data
y=cancer.target
# scaling data
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X)
# PCA transform data
pca = PCA(n_components=2) 
X_traned = pca.fit_transform(X_scaled)
print('Scaled dataset shape:', X_scaled.shape)
print('PCA transformed dataset shape:', X_traned.shape)
print('PCA component shape:',pca.components_.shape)
print('PCA component values:', pca.components_)
# plot first 2 principal components
fig, ax = plt.subplots(figsize=(7, 7), dpi=100)
ax.scatter(X_traned[:, 0], X_traned[:, 1], c=y, edgecolor='none', alpha=0.75,cmap=plt.cm.get_cmap('nipy_spectral', 10))
ax.set_xlabel('First principal component')
ax.set_ylabel('Second principal component')
ax.set_title('First 2 components of transformed dataset')
# 3D plot with the first 3 features of the scaled cancer.data set
fig2 = plt.figure(figsize=(10, 8))
cmap = plt.cm.get_cmap("Spectral")
ax2 = Axes3D(fig2, rect=[0, 0, .95, 1], elev=10, azim=10)
ax2.scatter(X_scaled[:,0],X_scaled[:,1],X_scaled[:,2], c=y, cmap=cmap)
ax2.set_xlabel('First principal component')
ax2.set_ylabel('Second principal component')
ax2.set_zlabel('Third principal component')
ax2.set_title('First 3 components of scaled dataset')
# 3D plot with the first 2 features of the transformed cancer.data set
fig3 = plt.figure(figsize=(10, 8))
ax3 = Axes3D(fig3, rect=[0, 0, .95, 1], elev=10, azim=10)
ax3.scatter(X_traned[:,0],X_traned[:,1], c=y, cmap=cmap)
ax3.set_xlabel('First principal component')
ax3.set_ylabel('Second principal component')
ax3.set_title('First 2 components of transformed dataset')
