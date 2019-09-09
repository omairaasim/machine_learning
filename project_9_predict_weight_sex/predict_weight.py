#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 07:41:20 2019

@author: omairaasim
"""

# Step 1 - Load data
import pandas as pd
dataset = pd.read_csv("weight-height.csv")

# Step 2 - Analyze data
dataset.info()
dataset.describe()
dataset.isnull().sum()

# Step 3 - Convert Gender to number 
# Using LabelEncoder Start # Comment this section if using other option
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 2].values
from sklearn.preprocessing import LabelEncoder
labelEncoder_gender =  LabelEncoder()
X[:,0] = labelEncoder_gender.fit_transform(X[:,0])

import numpy as np
X = np.vstack(X[:, :]).astype(np.float)
# Using LabelEncoder End #

############    OR     ##############


# Step 3 - Convert Gender to number 
# Replace directly in dataframe Start #
# dataset['Gender'].replace('Female',0, inplace=True)
# dataset['Gender'].replace('Male',1, inplace=True)
# X = dataset.iloc[:, :-1].values
# y = dataset.iloc[:, 2].values
# Replace directly in dataframe End #

# Step 4 - Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 5 - Fit Regression Model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Step 6 - Make Prediction using test data
lin_pred = lin_reg.predict(X_test)


# Step 7 - Model Accuracy
from sklearn import metrics
print('R square = ',metrics.r2_score(y_test, lin_pred))
print('Mean squared Error = ',metrics.mean_squared_error(y_test, lin_pred))
print('Mean absolute Error = ',metrics.mean_absolute_error(y_test, lin_pred))

# Step 8 - Predict my weight
my_weight_pred = lin_reg.predict([[0,74]])
print('My predicted weight = ',my_weight_pred)
