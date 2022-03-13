# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 19:33:32 2020

@author: User
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree

data = pd.read_excel("D:\AanshFolder\datasets\pacific-actualdata.xlsx")
#print(data.head())

data.Status = pd.Categorical(data.Status)
data['Status'] = data.Status.cat.codes
#print(data['Status'])
sns.countplot(data['Status'],label='count')
plt.show()

pred_columns= data[:]
pred_columns.drop(['Status'],axis=1,inplace=True)
pred_columns.drop(['Event'],axis=1,inplace=True)
pred_columns.drop(['Latitude'],axis=1,inplace=True)
pred_columns.drop(['Longitude'],axis=1,inplace=True)
pred_columns.drop(['Name'],axis=1,inplace=True)
pred_columns.drop(['ID'],axis=1,inplace=True)

prediction_var = pred_columns.columns
#print(list(prediction_var))

train,test = train_test_split(data,test_size=0.3)
print(train.shape)
print(test.shape)

train_X = train[prediction_var]
train_Y = train['Status']
#print(list(train.columns))

test_X = test[prediction_var]
test_Y = test['Status']
#print(list(test.columns))

#=================================== Desicion Tree ============================
model = tree.DecisionTreeClassifier()
model.fit(train_X,train_Y)

prediction = model.predict(test_X)
#df = pd.DataFrame(prediction,test_Y)
#print(df)
print("Decision Tree=",metrics.accuracy_score(prediction,test_Y))

#================================ Random Forest ===============================
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(train_X,train_Y)
prediction = model.predict(test_X)
print("Random Forest=",metrics.accuracy_score(prediction,test_Y))
