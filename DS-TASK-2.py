#!/usr/bin/env python
# coding: utf-8

# # DS-TASK-2

# # MOVIE RATING PREDICTION

#  Build a model that predicts the rating of a movie based on features like genre, 
# director, and actors. You can use regression techniques to tackle this problem. 
# • The goal is to analyze historical movie data and develop a model that accurately 
# estimates the rating given to a movie by users or critics. 
# • Movie Rating Prediction project enables you to explore data analysis, 
# preprocessing, feature engineering, and machine learning modeling techniques. 
# It provides insights into the factors that influence movie ratings and allows you 
# to build a model that can estimate the ratings of movies accurately

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv(r"C:\Users\mural\Downloads\wps_download\IMDb Movies India.csv",encoding='latin1')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.size


# In[6]:


df.memory_usage()


# In[7]:


df.describe()


# In[8]:


df.info()


# In[9]:


df.isnull().sum()


# In[10]:


df.columns


# In[11]:


columns_to_fill={'Year', 'Duration', 'Genre', 'Rating', 'Votes', 'Director',
       'Actor 1', 'Actor 2', 'Actor 3'}
for column in df.columns:
    df[column]=df[column].fillna(df[column].mode()[0])


# In[12]:


df.isnull().sum()


# In[13]:


df.duplicated().sum()


# In[14]:


df=df.drop_duplicates()


# In[15]:


df.duplicated().sum()


# In[16]:


df.head()


# In[17]:


df=df.drop(columns=['Name','Year','Duration'])


# In[18]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# In[19]:


df.dropna(subset=['Rating'], inplace=True)


# In[20]:


df.fillna({'Director': 'Unknown', 'Actor 1': 'Unknown', 'Actor 2': 'Unknown', 'Actor 3': 'Unknown'}, inplace=True)


# In[21]:


label_encoder = LabelEncoder()
df['Director'] = label_encoder.fit_transform(df['Director'])
df['Actor 1'] = label_encoder.fit_transform(df['Actor 1'])
df['Actor 2'] = label_encoder.fit_transform(df['Actor 2'])
df['Actor 3'] = label_encoder.fit_transform(df['Actor 3'])


# In[23]:


X = df.drop(columns=[ 'Rating']) 
y = df['Rating']


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[25]:


sc=StandardScaler()
sc.fit(X_train,y_train)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




