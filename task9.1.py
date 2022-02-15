# -*- coding: utf-8 -*-
"""
Created on Wed May 20 17:35:28 2020

@author: laure
"""
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC # "Support Vector Classifier"
# load dataset
digits = load_digits()
X = digits.data
y = digits.target
# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
#set parameter grid
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
# use GridSearchCV to fit data
clf = GridSearchCV(SVC(), param_grid,cv=5,return_train_score=True)
clf.fit(X_train, y_train)
print('Score of test dataset:',clf.score(X_test,y_test))
print('Best params:',clf.best_params_)
print('Best score:',clf.best_score_)
print('Best estimator:',clf.best_estimator_)



