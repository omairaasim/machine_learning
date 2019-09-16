# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


# Step 1 - Load Data
import pandas as pd
dataset = pd.read_csv("iphone_purchase_records.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 3].values

# Step 2 - Convert Gender to number
from sklearn.preprocessing import LabelEncoder
labelEncoder_gender =  LabelEncoder()
X[:,0] = labelEncoder_gender.fit_transform(X[:,0])

# Optional - if you want to convert X to float data type
import numpy as np
X = np.vstack(X[:, :]).astype(np.float)

# Step 3 - Split Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# Step 4 - Fit Classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=0)
classifier.fit(X_train, y_train)

# Step 5 - Predict
y_pred = classifier.predict(X_test)

# Step 6 - Metrics
#from sklearn import metrics
#cm = metrics.confusion_matrix(y_test, y_pred) ## 5,3 errors
#accuracy = metrics.accuracy_score(y_test, y_pred)  ## 0.92
#precision = metrics.precision_score(y_test, y_pred)  ## 0.85
#recall = metrics.recall_score(y_test, y_pred)  ## 0.90

# Step 6 - Evaluate the model performance
from sklearn import metrics
cm = metrics.confusion_matrix(y_test, y_pred) 
print(cm)
accuracy = metrics.accuracy_score(y_test, y_pred) 
print("Accuracy score:",accuracy)
precision = metrics.precision_score(y_test, y_pred) 
print("Precision score:",precision)
recall = metrics.recall_score(y_test, y_pred) 
print("Recall score:",recall)

