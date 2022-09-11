# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 14:24:13 2022

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import tensorflow as tf
# File system manangement
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import pickle

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv(r"C:\Users\ABHISHEK\Downloads\train_data_evaluation_part_2.csv")
print('Training data shape: ', train.shape)
train.head()

test = pd.read_csv(r"C:\Users\ABHISHEK\Downloads\test_data_evaluation_part2.csv")
print('Testing data shape: ', test.shape)
test.head()

train['BookingsCheckedIn'].value_counts()
sns.countplot(x = 'BookingsCheckedIn',data = train)

#finding null value
print("Null in Training set")
print("---------------------")
print(train.isnull().sum())
print("---------------------")
print("Null in Testing set")
print("---------------------")
print(test.isnull().sum())

def add_age(cols):
    Age = cols[0]
    if pd.isnull(Age):
        return int(train["Age"].mean())
    else:
        return Age
    
train_corr = train.iloc[:,1:-1].corr()
colormap = sns.diverging_palette(220, 10, as_cmap = True)
plt.figure(figsize=(14,14))
sns.heatmap(train_corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 12},cmap = colormap, linewidths=0.1, linecolor='white')
plt.title('Correlation of Features ', y=1.05, size=15)

test_corr = test.iloc[:,1:-1].corr()
colormap = sns.diverging_palette(220, 10, as_cmap = True)
plt.figure(figsize=(14,14))
sns.heatmap(test_corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 12},cmap = colormap, linewidths=0.1, linecolor='white')
plt.title('Correlation of Features ', y=1.05, size=15)

train['Age'] = train[['Age']].apply(add_age,axis=1)
test['Age'] = test[['Age']].apply(add_age,axis=1)

train['Age'].describe()

#Dropping unnecessary Column
train.drop(['DistributionChannel','MarketSegment','SRHighFloor','SRLowFloor','SRAccessibleRoom','SRMediumFloor','SRBathtub','SRShower','SRCrib','SRKingSizeBed','SRTwinBed','SRNearElevator','SRAwayFromElevator','SRNoAlcoholInMiniBar','SRQuietRoom'],inplace=True,axis=1)
test.drop(['DistributionChannel','MarketSegment','SRHighFloor','SRLowFloor','SRAccessibleRoom','SRMediumFloor','SRBathtub','SRShower','SRCrib','SRKingSizeBed','SRTwinBed','SRNearElevator','SRAwayFromElevator','SRNoAlcoholInMiniBar','SRQuietRoom'],inplace=True,axis=1)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
columns = ['ID', 'Age', 'DaysSinceCreation','AverageLeadTime','LodgingRevenue', 'OtherRevenue', 'BookingsCanceled', 'BookingsNoShowed', 'PersonsNights', 'RoomNights', 'DaysSinceLastStay','DaysSinceFirstStay']
sc.fit(train[columns], train["BookingsCheckedIn"])

X = train[columns]
y = train['BookingsCheckedIn']

X_train , X_test , y_train , y_test = train_test_split(X, y, test_size=0.15,random_state=0)


#Feature Scaling 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#import logistic regression classifier 
from sklearn.linear_model import LogisticRegression 
logistic_classifier = LogisticRegression()
logistic_classifier.fit(X_train , y_train)

#predicting test set values
logistic_y_pred = logistic_classifier.predict(X_test)

#creating confusion matrix
from sklearn.metrics import confusion_matrix
logistic_cm=confusion_matrix(y_test,logistic_y_pred)
print(logistic_cm)

#Logistic regression accuracy score
from sklearn.metrics import accuracy_score
logistic_ac=accuracy_score(y_test, logistic_y_pred)
print(logistic_ac)

pickle.dump(logistic_classifier, open('model.pkl', 'wb'))

