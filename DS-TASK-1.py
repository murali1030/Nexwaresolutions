#!/usr/bin/env python
# coding: utf-8

# # DS-TASK-1

# # TITANIC SURVIVAL PREDICTION

# Use the Titanic dataset to build a model that predicts whether a passenger on the 
# Titanic survived or not. 
# • This is a classic beginner project with readily available data. 
# • The dataset typically used for this project contains information about individual 
# passengers, such as their age, gender, ticket class, fare, cabin, and whether or 
# not they survived.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv(r"C:\Users\mural\Downloads\Titanic-Dataset.csv")
df.head()


# In[3]:


df.shape


# In[4]:


df.size


# In[5]:


df.memory_usage()


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[8]:


df['Age']=df['Age'].fillna('86')


# In[9]:


df['Cabin']=df['Cabin'].fillna('327')
df['Embarked']=df['Embarked'].fillna('2')


# In[10]:


df.isnull().sum()


# In[11]:


df.duplicated().sum()


# In[12]:


df.head()


# In[13]:


df=df.drop(columns=['Name','Ticket','Fare','Cabin'],axis=1)


# In[14]:


df.head()


# In[15]:


df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)


# In[16]:


df.head()


# In[17]:


X = df.drop('Survived', axis=1)
y = df['Survived']


# In[18]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[19]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[20]:


model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)


# In[21]:


y_pred = model.predict(X_test)


# In[22]:


accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


# In[23]:


print(classification_report(y_test, y_pred))


# In[24]:


conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)


# In[25]:


sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()


# In[ ]:




