#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 19:28:27 2018

@author: omairaasim
"""

# Step 1 - Load Data
import pandas as pd
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[: ,1:2].values
y = dataset.iloc[:, 2:].values

# Step 2 - Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Step 3 - Fit SVR
from sklearn.svm import SVR
regressor = SVR(kernel = "rbf")
regressor.fit(X,y)

# Step 4 - Visualization
import matplotlib.pyplot as plt
plt.scatter(X, y , color="red")
plt.plot(X, regressor.predict(X), color="blue")
plt.title("SVR")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()

# Step 5 - Predict Results
import numpy as np
# First transform 6.5 to feature scaling
sc_X_val = sc_X.transform(np.array([[6.5]]))
# Second predict the value
scaled_y_pred = regressor.predict(sc_X_val)
# Third - since this is scaled - we have to inverse transform
y_pred = sc_y.inverse_transform(scaled_y_pred) 
print('The predicted salary of a person at 6.5 Level is ',y_pred)
