#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 08:49:06 2019

@author: omairaasim
"""

# Step 1 : Load Dataset 
import pandas as pd
dataset = pd.read_csv("Height_Weight_single_variable_data_101_series_1.0.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Step 2: Check for missing values
dataset.isnull().sum()

# Step 3: Split dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 4: Fit Linear Regression Model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Step 5: Predict values for test data
lin_pred = lin_reg.predict(X_test)

# Step 6: Compare predictions with real results
from sklearn import metrics
print('R square = ',metrics.r2_score(y_test, lin_pred))
print('Mean squared Error = ',metrics.mean_squared_error(y_test, lin_pred))


# Step 7: Visualize Training set
import matplotlib.pyplot as plt
plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, lin_reg.predict(X_train), color="blue" )
plt.title("Height and Weight - Training Set")
plt.xlabel("Height")
plt.ylabel("Weight")
plt.show()

# Step 8: Visualize Test set
import matplotlib.pyplot as plt
plt.scatter(X_test, y_test, color="red")
plt.plot(X_train, lin_reg.predict(X_train), color="blue" )
plt.title("Height and Weight - Test Set")
plt.xlabel("Height")
plt.ylabel("Weight")
plt.show()

# Step 9: Make new Prediction
lin_pred_new = lin_reg.predict([[166]])
print('If a person has height 166, the predicted weight is ',lin_pred_new)




