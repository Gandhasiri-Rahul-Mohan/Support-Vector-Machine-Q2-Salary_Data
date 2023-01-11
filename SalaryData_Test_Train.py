# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 16:34:21 2023

@author: Rahul
"""

import pandas as pd
import numpy as np

# train Data
df_train = pd.read_csv("D:\\DS\\books\\ASSIGNMENTS\\Support vector machines\\SalaryData_Train(1).csv")
df_train
df_train.shape
df_train.info()
df_train.isnull().any()
# there is no null values

# Test data
df_test=pd.read_csv("D:\\DS\\books\\ASSIGNMENTS\\Support vector machines\\SalaryData_Test(1).csv")
df_test
df_test.shape
df_test.info()
df_test.isnull().any()
# There are no null values

# Concating
df=pd.concat([df_train,df_test],axis=0)
df.shape
df.info()
df.corr()
df.corr().to_csv('SVM.csv')
# There is no relation b/w independent variables
df.head()
df.dtypes

#Finding the special characters in the data frame 

df.isin(['?']).sum(axis=0)
print(df[0:5])

import matplotlib.pyplot as plt
import seaborn as sns

# Outliers dectection and treating outliers
df.boxplot("age",vert=False)
Q1=np.percentile(df["age"],25)
Q3=np.percentile(df["age"],75)
IQR=Q3-Q1
LW=Q1-(2.0*IQR)
UW=Q3+(2.0*IQR)
df["age"]<LW
df[df["age"]<LW]
df[df["age"]<LW].shape
df["age"]>UW
df[df["age"]>UW]
df[df["age"]>UW].shape
df["age"]=np.where(df["age"]>UW,UW,np.where(df["age"]<LW,LW,df["age"]))

df.boxplot("educationno",vert=False)
Q1=np.percentile(df["educationno"],25)
Q3=np.percentile(df["educationno"],75)
IQR=Q3-Q1
LW=Q1-(2.0*IQR)
UW=Q3+(2.0*IQR)
df["educationno"]<LW
df[df["educationno"]<LW]
df[df["educationno"]<LW].shape
df["educationno"]>UW
df[df["educationno"]>UW]
df[df["educationno"]>UW].shape
df["educationno"]=np.where(df["educationno"]>UW,UW,np.where(df["educationno"]<LW,LW,df["educationno"]))

df.boxplot("hoursperweek",vert=False)
Q1=np.percentile(df["hoursperweek"],25)
Q3=np.percentile(df["hoursperweek"],75)
IQR=Q3-Q1
LW=Q1-(2.0*IQR)
UW=Q3+(2.0*IQR)
df["hoursperweek"]<LW
df[df["hoursperweek"]<LW]
df[df["hoursperweek"]<LW].shape
df["hoursperweek"]>UW
df[df["hoursperweek"]>UW]
df[df["hoursperweek"]>UW].shape
df["hoursperweek"]=np.where(df["hoursperweek"]>UW,UW,np.where(df["hoursperweek"]<LW,LW,df["hoursperweek"]))

df.boxplot("capitalgain",vert=False)
Q1=np.percentile(df["capitalgain"],25)
Q3=np.percentile(df["capitalgain"],75)
IQR=Q3-Q1
LW=Q1-(2.0*IQR)
UW=Q3+(2.0*IQR)
df["capitalgain"]<LW
df[df["capitalgain"]<LW]
df[df["capitalgain"]<LW].shape
df["capitalgain"]>UW
df[df["capitalgain"]>UW]
df[df["capitalgain"]>UW].shape
df["capitalgain"]=np.where(df["capitalgain"]>UW,UW,np.where(df["capitalgain"]<LW,LW,df["capitalgain"]))

t1 = pd.crosstab(index=df["education"],columns=df["workclass"])
t1.plot(kind='bar')

t2 = pd.crosstab(index=df["education"],columns=df["Salary"])
t2.plot(kind='bar')

t3 = pd.crosstab(index=df["sex"],columns=df["race"])
t3.plot(kind='bar')

t4 = pd.crosstab(index=df["maritalstatus"],columns=df["sex"])
t4.plot(kind='bar')

df["age"].hist()
df["educationno"].hist()
df["capitalgain"].hist()
df["capitalloss"].hist()
df["hoursperweek"].hist()

# Check Correlation amoung parameters
corr = df.corr()
fig, ax = plt.subplots(figsize=(8,8))
# Generate a heatmap
sns.heatmap(corr, cmap = 'magma', annot = True, fmt = ".2f")
plt.xticks(range(len(corr.columns)), corr.columns)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.show()

from sklearn.preprocessing import LabelEncoder
df = df.apply(LabelEncoder().fit_transform)
df.head()

drop_elements = ['education', 'native', 'Salary']
X = df.drop(drop_elements, axis=1)
y = df['Salary']

#Data partition 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=31)


#support Vector mission
svc = SVC()
svc.fit(X_train, y_train)
# make predictions
prediction = svc.predict(X_test)
# summarize the fit of the model
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

print("Accuracy:",metrics.accuracy_score(y_test, prediction))
print("Precision:",metrics.precision_score(y_test, prediction))
print("Recall:",metrics.recall_score(y_test, prediction))


svc = SVC(kernel='rbf',gamma=2, C=1)
svc.fit(X_train, y_train)
# make predictions
prediction = svc.predict(X_test)
# summarize the fit of the model
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

print("Accuracy:",metrics.accuracy_score(y_test, prediction))
print("Precision:",metrics.precision_score(y_test, prediction))
print("Recall:",metrics.recall_score(y_test, prediction))


svc = SVC(kernel='poly',degree=3,gamma="scale")
svc.fit(X_train, y_train)
# make predictions
prediction = svc.predict(X_test)
# summarize the fit of the model
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

print("Accuracy:",metrics.accuracy_score(y_test, prediction))
print("Precision:",metrics.precision_score(y_test, prediction))
print("Recall:",metrics.recall_score(y_test, prediction))

# Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)

#prediction
y_pred_test = logreg.predict(X_test)

print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

print("Accuracy:",metrics.accuracy_score(y_test, prediction))
print("Precision:",metrics.precision_score(y_test, prediction))
print("Recall:",metrics.recall_score(y_test, prediction))

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train,y_train)

y_pred_test = classifier.predict(X_test)

print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

print("Accuracy:",metrics.accuracy_score(y_test, prediction))
print("Precision:",metrics.precision_score(y_test, prediction))
print("Recall:",metrics.recall_score(y_test, prediction))













