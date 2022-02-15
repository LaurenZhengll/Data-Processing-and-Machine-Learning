# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 16:49:22 2020

@author: laure
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier

data = pd.read_csv("C:/Users/laure/OneDrive - Deakin University/SIT384/task6-resources/payment_fraud.csv");
data = pd.get_dummies(data, columns=['paymentMethod'])  # replace the column of text with 3 binary columns 
X_columns=['accountAgeDays', 'numItems', 'localTime', 'paymentMethodAgeDays', 'paymentMethod_creditcard','paymentMethod_paypal', 'paymentMethod_storecredit']
X=data[X_columns]
#X = np.c_[data['accountAgeDays'],data['numItems'], data['localTime'],data['paymentMethodAgeDays'],data['paymentMethod_creditcard'],data['paymentMethod_paypal'],data['paymentMethod_storecredit']]
y = data['label']

# split into train dataset and test dataset with ramdon order
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
logreg = LogisticRegression() #create and define logistic regression model
logreg.fit(X_train, y_train) # train the model
# use dataframe to store coeffitients
coeff = pd.DataFrame({'C1_coefficient': logreg.coef_.transpose().flatten()})
fig, ax = plt.subplots(figsize=(7, 7), dpi=100)
x=np.arange(7)
# plot coeffients of model C=1
ax.scatter(x, coeff['C1_coefficient'].values, c='r', alpha=0.7, s=75,edgecolor='None')
ax.set_xticks(x); # set x ticks
# set labels of x ticks and rotation of the labels is 90
ax.set_xticklabels(X_columns, rotation = 90);
ax.set_xlabel('Feature')
ax.set_ylabel('Coefficient')
ax.set_title('Logistic regression feature coefficients with different C')
y_pred1 = logreg.predict(X_test) # use trained model to predict y
print('C=1, training set score is', logreg.score(X_train,y_train))
print('C=1, test set score is', logreg.score(X_test,y_pred1))
# train a model with C=100
logreg2 = LogisticRegression(solver='liblinear', C=100.0, random_state=0)
logreg2.fit(X_train, y_train)
coeff['C100_coeff'] = logreg2.coef_.transpose().flatten()
ax.scatter(x, coeff['C100_coeff'].values, c='orange', alpha=0.7, s=75,edgecolor='None')
y_pred2 = logreg2.predict(X_test)
print('C=100, training set score is', logreg2.score(X_train,y_train))
print('C=100, test set score is', logreg2.score(X_test,y_pred2))
# train a model with C=10
logreg3 = LogisticRegression(solver='liblinear', C=10.0, random_state=0)
logreg3.fit(X_train, y_train)
coeff['C10_coeff'] = logreg3.coef_.transpose().flatten()
ax.scatter(x, coeff['C10_coeff'].values, c='g', alpha=0.7, s=75,edgecolor='None')
y_pred3 = logreg3.predict(X_test)
print('C=10, training set score is', logreg3.score(X_train,y_train))
print('C=10, test set score is', logreg3.score(X_test,y_pred3))
# train a model with C=0.01
logreg4 = LogisticRegression(solver='liblinear', C=0.01, random_state=0)
logreg4.fit(X_train, y_train)
coeff['C0.01_coeff'] = logreg4.coef_.transpose().flatten()
ax.scatter(x, coeff['C0.01_coeff'].values, c='b', alpha=0.7, s=75,edgecolor='None')
y_pred4 = logreg4.predict(X_test)
print('C=0.01, training set score is', logreg4.score(X_train,y_train))
print('C=0.01, test set score is', logreg4.score(X_test,y_test))
# train a model with C=0.001
logreg5 = LogisticRegression(solver='liblinear', C=0.001, random_state=0)
logreg5.fit(X_train, y_train)
coeff['C0.001_coeff'] = logreg5.coef_.transpose().flatten()
ax.scatter(x, coeff['C0.001_coeff'].values, c='purple', alpha=0.7, s=75,edgecolor='None')
ax.legend(['C=1','C=100','C=10','C=0.01','C=0.001'])  # legend the plot
y_pred5 = logreg5.predict(X_test)
print('C=0.001, training set score is', logreg5.score(X_train,y_train))
print('C=0.001, test set score is', logreg5.score(X_test,y_test))

# create Decision Tree classifer object
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)  # train Decision Tree Classifer
print('\nDecision tree training set score is', dt.score(X_train,y_train))
print('Test set score is', dt.score(X_test,y_test))
print('The decision tree depth is',dt.get_depth())  # get depth of decision tree
# use dataframe, one column is feature, anther is importance
importances = pd.DataFrame({'feature':X_columns,'importance':dt.feature_importances_})
print(importances) #print dataframe in origin format
fig2, ax2 = plt.subplots(figsize=(7, 7), dpi=100)
ax2.bar(x, importances['importance'].values, align = 'center', color = 'b');
ax2.set_xticks(x); # set x ticks
ax2.set_xticklabels(X_columns, rotation = 90);
ax2.set_xlabel("Features"); # set x y labels and title
ax2.set_ylabel("Importance");
ax2.set_title("Decision tree feature importances");

