#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Libraries 

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import os 
import pickle


# In[3]:


#Load Data Set 

os.chdir('C:\\Users\\USER\\Downloads\\drive-download-20240617T100841Z-001\\')
data = pd.read_csv('data.csv')
display (data)


# In[4]:


#Create X 

x= data.drop ('diagnosis',axis =1).drop('id',axis =1)
display (x)


# In[5]:


#Create Y

y = data['diagnosis']
display (y)


# In[6]:


#Update Y to Category Column 

y = y.astype('category')
display (y)


# In[8]:


#Apply Cat Codes (Label Encoding)

y = y.cat.codes
display (y)


# In[9]:


#Load XG Boost  and Predict 

load_model =pickle.load(open("XGBoost","rb"))
pred1 = load_model.predict (x)
print (load_model.best_params_)
print (accuracy_score (pred1,y))
display (pred1)


# In[10]:


#Load Support Vector Machine 

load_model =pickle.load(open("SVC","rb"))
pred1 = load_model.predict (x)
print (load_model.best_params_)
print (accuracy_score (pred1,y))
display (pred1)


# In[11]:


#Load Random Forest 

load_model =pickle.load(open("Random Forest","rb"))
pred1 = load_model.predict (x)
print (load_model.best_params_)
print (accuracy_score (pred1,y))
display (pred1)


# In[ ]:




