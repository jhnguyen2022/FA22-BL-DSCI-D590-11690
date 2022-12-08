import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import pickle

#Load dataset
data = pd.read_csv('h1b.csv')
data.drop(['Unnamed: 0', 'lon', 'lat'], axis=1, inplace=True)
data.dropna(subset=['CASE_STATUS'], inplace=True)
data['EMPLOYER_NAME'].fillna('NO COMPANY NAME', inplace=True)
data['SOC_NAME'].fillna('OTHER', inplace=True)
data['JOB_TITLE'].fillna('OTHER', inplace=True)
data['FULL_TIME_POSITION'].fillna(data['FULL_TIME_POSITION'].mode()[0], inplace=True)
data['PREVAILING_WAGE'].fillna(data['PREVAILING_WAGE'].mode()[0], inplace=True)

data2 = data[data['PREVAILING_WAGE'] <= 500000]
data2 = data2[data2['CASE_STATUS'].isin(['CERTIFIED', 'CERTIFIED-WITHDRAWN', 'DENIED'])]

data2['CASE_STATUS'] = data2['CASE_STATUS'].map({'DENIED':0,'CERTIFIED':1, 'CERTIFIED-WITHDRAWN':2})
data2['FULL_TIME_POSITION'] = data2['FULL_TIME_POSITION'].map({'N':0, 'Y':1})
data2['SOC_NAME1'] = 'others'
data2['SOC_NAME1'][data2['SOC_NAME'].str.contains('Computer','SOFTWARE')] = 'IT'
data2['SOC_NAME1'][data2['SOC_NAME'].str.contains('CHIEF','MANAGEMENT')] = 'manager'
data2['SOC_NAME1'][data2['SOC_NAME'].str.contains('MECHANICAL', 'ENGINEER')] = 'engineer'
data2['SOC_NAME1'][data2['SOC_NAME'].str.contains('DATABASE')] = 'data science'
data2['SOC_NAME1'][data2['SOC_NAME'].str.contains('SALES','MARKET')] = 'retail'
data2['SOC_NAME1'][data2['SOC_NAME'].str.contains('FINANCIAL')] = 'finance'
data2['SOC_NAME1'][data2['SOC_NAME'].str.contains('PUBLIC','FUNDRAISING')] = 'PR'
data2['SOC_NAME1'][data2['SOC_NAME'].str.contains('EDUCATION','LAW')] = 'admin'
data2['SOC_NAME1'][data2['SOC_NAME'].str.contains('AUDITORS','COMPLIANCE')] = 'audit'
data2['SOC_NAME1'][data2['SOC_NAME'].str.contains('DISTRIBUTION','LOGISTICS')] = 'supply chain'
data2['SOC_NAME1'][data2['SOC_NAME'].str.contains('RECRUITERS','HUMAN')] = 'HR'
data2['SOC_NAME1'][data2['SOC_NAME'].str.contains('AGRICULTURAL','FARM')] = 'agri'
data2['SOC_NAME1'][data2['SOC_NAME'].str.contains('CONSTRUCTION','ARCHITECTURAL')] = 'real estate'
data2['SOC_NAME1'][data2['SOC_NAME'].str.contains('FORENCSIC','HEALTH')] = 'medical'
data2['SOC_NAME1'][data2['SOC_NAME'].str.contains('TEACHERS')] = 'education'

data2['SOC'] = preprocessing.LabelEncoder().fit_transform(data2['SOC_NAME1'])
data2.drop(['EMPLOYER_NAME', 'SOC_NAME', 'SOC_NAME1', 'JOB_TITLE', 'WORKSITE'], axis=1, inplace=True)

# Building model
X = data2.drop(['CASE_STATUS'], axis=1)
y = data2['CASE_STATUS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=30)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

pickle.dump(logreg, open('model.pkl', 'wb'))