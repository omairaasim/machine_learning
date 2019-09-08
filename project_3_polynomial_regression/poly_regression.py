#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: omairaasim
"""

# Step 1 - Load Data
import pandas as pd
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Step 2 - Fitting Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

# Step 3 - Visualize Linear Regression Results
import matplotlib.pyplot as plt

plt.scatter(X,y, color="red")
plt.plot(X, lin_reg.predict(X))
plt.title("Linear Regression")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

# Step 4 Linear Regression prediction
lin_reg.predict([[6.5]])

# Step 5 - Convert X to polynomial format
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)


# Step 6 - Passing X_poly to LinearRegression
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

# Step 7 - Visualize Poly Regression Results
plt.scatter(X,y, color="red")
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)))
plt.title("Poly Regression - Degree 4")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()


# Step 8 Polynomial Regression prediction
new_salary_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print('The predicted salary of a person at 6.5 Level is ',new_salary_pred)
