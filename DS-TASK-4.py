#!/usr/bin/env python
# coding: utf-8

# # DS-TASK-4

# # IRIS FLOWER CLASSIFICATION

# The Iris flower dataset consists of three species: setosa, versicolor, and virginica. 
# • These species can be distinguished based on their measurements. Now, imagine 
# that you have the measurements of Iris flowers categorized by their respective 
# species. 
# • Your objective is to train a machine learning model that can learn from these 
# measurements and accurately classify the Iris flowers into their respective 
# species. 
# • Use the Iris dataset to develop a model that can classify iris flowers into different 
# species based on their sepal and petal measurements. This dataset is widely 
# used for introductory classification tasks

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv('Iris.csv')
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


df.duplicated().sum()


# In[9]:


df.head()


# In[10]:


sns.pairplot(df, hue='Species')
plt.show()


# In[11]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# In[12]:


df.head()


# In[13]:


le=LabelEncoder()
df['Species']=le.fit_transform(df['Species'])


# In[14]:


#df.head()


# In[15]:


x=df.drop(['Species'],axis=1)
y=df['Species']


# In[16]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[17]:


x_train.head()


# In[18]:


x_test.head()


# In[19]:


y_train


# In[20]:


y_test.head()


# In[21]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[22]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


# In[23]:


model=LogisticRegression(max_iter=200)
model.fit(x_train,y_train)


# In[24]:


y_pred=model.predict(x_test)


# In[25]:


accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


# In[26]:


print(classification_report(y_test, y_pred, target_names=le.classes_))


# In[27]:


conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()


# In[28]:


import joblib
joblib.dump(model, 'iris_model.pkl')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




