#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


import os
for dirname, _, filenames in os.walk('D:/Paper/datasets'):
    for filename in filenames:
        desktopData= pd.read_csv("D:\Paper\datasets\dataPlos\dyt-desktop.csv", index_col=0, na_values=['(NA)'])
def SeparateColumns(dataSetName):
    columns = defaultdict(list)
    with open(dataSetName, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        headers = next(reader)
        column_nums = range(len(headers)) 
        for row in reader:
            for i in column_nums:
            
                columns[headers[i]].append(row[i])
    
    return dict(columns)
def cleanData(data) :
    for col in data.columns.values:
        data[col] = data[col].astype('string')
    #----------
    for col in data.columns.values:
        data[col] = data[col].astype('float',errors = 'ignore')
    #-----------
    data['Gender']=data.Gender.map({'Male': 1, 'Female': 2})
    data['Dyslexia']=data.Dyslexia.map({'No': 0, 'Yes': 1})
    data['Nativelang']=data.Nativelang.map({'No': 0, 'Yes': 1})
    data['Otherlang']=data.Otherlang.map({'No': 0, 'Yes': 1})
columns = SeparateColumns('D:\Paper\datasets\dataPlos\dyt-desktop.csv')
desktopData=pd.DataFrame.from_dict(columns)

desktopData
        


# In[12]:


cleanData(desktopData)

desktopData.head()


# In[14]:


columns = SeparateColumns('D:\Paper\datasets\dataPlos\dyt-desktop.csv')
tabletData=pd.DataFrame.from_dict(columns)
tabletData.replace(["NULL"], np.nan, inplace = True)

tabletData


# In[15]:


cleanData(tabletData)

tabletData.head()


# In[22]:


stateOfNUll= tabletData.isnull().any()
i = 0
for state in stateOfNUll : 
    if(state):  
        tabletData[stateOfNUll.index[i]].fillna(round(tabletData[stateOfNUll.index[i]].mean() , 4), inplace=True)
    i = i + 1    

tabletData.head()


# In[23]:


cols_with_missing = [col for col in tabletData.columns if tabletData[col].isnull().any()]
reduced_desktopData = desktopData.drop(cols_with_missing, axis=1)
reduced_tabletData = tabletData.drop(cols_with_missing, axis=1)


# In[24]:


commonalityColumns = ['Gender','Nativelang','Otherlang','Age' , 'Dyslexia']
for i in  range(30):
    if((i>=0 and i<12) or (i>=13 and i<17) or i==21 or i==22 or i==29):
        commonalityColumns.append('Clicks'+str(i+1))
        commonalityColumns.append('Hits'+str(i+1))
        commonalityColumns.append('Misses'+str(i+1))
        commonalityColumns.append('Score'+str(i+1))
        commonalityColumns.append('Accuracy'+str(i+1))
        commonalityColumns.append('Missrate'+str(i+1))    
reduced_desktopData=reduced_desktopData.loc[:,commonalityColumns]
reduced_tabletData=reduced_tabletData.loc[:,commonalityColumns]


# In[25]:


y=reduced_desktopData['Dyslexia']
X=reduced_desktopData.loc[:, reduced_desktopData.columns != 'Dyslexia']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=42)


# In[26]:


rfc = RandomForestClassifier()
rfc.fit(X_train , y_train)
y_pred = rfc.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[28]:


yTest=reduced_tabletData['Dyslexia']
XTest=reduced_tabletData.loc[:, reduced_tabletData.columns != 'Dyslexia']
rfc2 = RandomForestClassifier()
rfc2.fit(X_train , y_train)
y_pred = rfc2.predict(XTest)
print("Accuracy:",metrics.accuracy_score(yTest, y_pred))


# In[ ]:




