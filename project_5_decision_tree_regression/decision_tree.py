#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 20:35:24 2018

@author: omairaasim
"""

# Step 1 - Load Dataset
import pandas as pd
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Step 2 - Fit Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(criterion="mse")
regressor.fit(X, y)

# Step 3 - Visualize
import matplotlib.pyplot as plt

import numpy as np
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid),1))

plt.scatter(X, y, color="red")
plt.plot(X_grid, regressor.predict(X_grid), color="blue")
plt.title("Decision Tree Regressor")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()

# Step 4 - Predict
y_pred = regressor.predict([[6.5]])
print('The predicted salary of a person at 6.5 Level is ',y_pred)
