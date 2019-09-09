#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 07:41:20 2019

@author: omairaasim
"""

# Step 1 - Load data
import pandas as pd
dataset = pd.read_csv("weight-height.csv")

dataset.info()
dataset.describe()
dataset.isnull().sum()

# Step 2 - Convert Gender to number
dataset['Gender'].replace('Female',0, inplace=True)
dataset['Gender'].replace('Male',1, inplace=True)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 2].values

# Step 2 - Use LabelEncoder
# from sklearn.preprocessing import LabelEncoder
# labelEncoder_gender =  LabelEncoder()
# X[:,0] = labelEncoder_gender.fit_transform(X[:,0])

# import numpy as np
# float_arr = np.vstack(X[:, :]).astype(np.float)

# Step 3 - Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 4 - Fit Regression Model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Step 5 - Make Prediction
lin_pred = lin_reg.predict(X_test)


from sklearn import metrics
import numpy as np
print('R square = ',metrics.r2_score(y_test, lin_pred))
print('Mean squared Error = ',metrics.mean_squared_error(y_test, lin_pred))
print('Root Mean squared Error = ',np.sqrt(metrics.mean_squared_error(y_test, lin_pred)))
print('Mean absolute Error = ',metrics.mean_absolute_error(y_test, lin_pred))

# Step 7 - Predict my weight
my_weight_pred = lin_reg.predict([[0,74]])

