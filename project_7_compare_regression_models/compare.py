#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 11:39:26 2018

@author: omairaasim
"""

# Step 1 - Load Data
import pandas as pd
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


###########################
### Linear Regression ###
###########################
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)

# Predict
lin_pred = linear_regressor.predict([[6.5]])
print('The predicted salary of a person at 6.5 Level with Linear Regression is ',lin_pred)

################################
### Polynomial Regression ###
################################

# ** NOTE - conver X to X_poly of required degree
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=4)
X_poly = poly_features.fit_transform(X)

from sklearn.linear_model import LinearRegression
poly_regressor = LinearRegression()
poly_regressor.fit(X_poly, y)

# Predict - have to convert 6.5 to poly format
poly_pred = poly_regressor.predict(poly_features.fit_transform([[6.5]]))
print('The predicted salary of a person at 6.5 Level with Polynomial Regression is ',poly_pred)

################################
### SVR Regression ###
################################

# ** NOTE - SVR does not do feature scaling
from sklearn.preprocessing import StandardScaler
ss_x = StandardScaler()
ss_y = StandardScaler()
X_scaled = ss_x.fit_transform(X)
y_scaled = ss_y.fit_transform(y.reshape(-1,1))


from sklearn.svm import SVR
svr_regressor = SVR(kernel="rbf")
svr_regressor.fit(X_scaled, y_scaled)

# Predict - since we did feature scaling -
# So have to scale/transform 6.5 also
position_val = ss_x.transform([[6.5]])
pred_val_scaled = svr_regressor.predict(position_val)
# The above statement will return scaled predicted value
# So have to convert that using inverse transform
svr_pred = ss_y.inverse_transform(pred_val_scaled)
print('The predicted salary of a person at 6.5 Level with Support Vector Regression is ',svr_pred)

################################
### Decision Tree Regression ###
################################
from sklearn.tree import DecisionTreeRegressor
tree_regressor = DecisionTreeRegressor(criterion="mse")
tree_regressor.fit(X, y)

# Predict
tree_pred = tree_regressor.predict([[6.5]])
print('The predicted salary of a person at 6.5 Level with Decision Tree Regression is ',tree_pred)

################################
### Random Forest Regression ###
################################
from sklearn.ensemble import RandomForestRegressor
forest_regressor = RandomForestRegressor(n_estimators=300, random_state=0)
forest_regressor.fit(X, y)

# Predict
forest_pred = forest_regressor.predict([[6.5]])
print('The predicted salary of a person at 6.5 Level with Random Forest Regression is ',forest_pred)


################################
### Visualizations ###
################################
import matplotlib.pyplot as plt
import numpy as np

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid),1))

plt.scatter(X, y,color="red")
plt.plot(X_grid, linear_regressor.predict(X_grid), color="blue")
plt.plot(X_grid, poly_regressor.predict(poly_features.fit_transform(X_grid)), color="green")
plt.plot(X_grid, ss_y.inverse_transform(svr_regressor.predict(ss_x.transform(X_grid))), color="orange")
plt.plot(X_grid, tree_regressor.predict(X_grid), color="black")
plt.plot(X_grid, forest_regressor.predict(X_grid), color="purple")
#plt.xticks(np.arange(min(X), max(X)+1, 1))
#plt.yticks(np.arange(min(y), max(y)+1, 50000))
plt.title("Regression")
plt.xlabel("Position")
plt.ylabel("Salaries")
#plt.figure(figsize=(20,10))
#fig = plt.gcf()
#fig.set_size_inches(10.5, 10)
plt.show()
