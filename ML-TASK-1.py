#!/usr/bin/env python
# coding: utf-8

# # ML-TASK-1

# # MOVIE GENRE PREDICTION

# Create a machine learningmodel that can predict the genre of a 
# movie based on its plot summary or other textual information. You 
# can use techniques like TF-IDF or word embeddings with classifiers 
# such as Naive Bayes, Logistic Regression, or Support Vector 
# Machines.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_parquet(r"C:\Users\mural\Downloads\test-00000-of-00001-35e9a9274361daed.parquet")
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


df.describe()


# In[8]:


df.isnull().sum()


# In[9]:


df.duplicated().sum()


# In[10]:


df.head()


# In[11]:


df=df.drop(columns=['id','movie_name'],axis=1)


# In[12]:


df.head()


# In[13]:


from sklearn.preprocessing import LabelEncoder

# Label encoding the genre column
le = LabelEncoder()
df['genre_encoded'] = le.fit_transform(df['genre'])

# Now you can compare with the mean of the encoded values
df['genre_new'] = df['genre_encoded'].apply(lambda x: 1 if x > df['genre_encoded'].mean() else 0)

# Check the results
print(df[['genre', 'genre_encoded', 'genre_new']].head())


# In[14]:


df['genre'].value_counts()


# In[15]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


# In[16]:


nltk.download('stopwords')
nltk.download('punkt')


# In[17]:


stop_words=set(stopwords.words('english'))
ps=PorterStemmer()


# In[18]:


def preprocess(text):
    text=text.lower()
    tokens=word_tokenize(text)
    tokens=[word for word in tokens if word.isalnum()]
    tokens=[word for word in tokens if word not in stop_words]
    tokens=[ps.stem(word)for word in tokens]
    return ' '.join(tokens)


# In[19]:


df['processed_synopsis'] = df['synopsis'].apply(preprocess)
print(df[['synopsis','processed_synopsis']])


# In[20]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[21]:


Tfidf_Vectorizer=TfidfVectorizer()
x=Tfidf_Vectorizer.fit_transform(df['processed_synopsis'])
y=df['genre']


# In[22]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[23]:


x_train


# In[24]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report,matthews_corrcoef


# In[25]:


model=MultinomialNB()
model.fit(x_train,y_train)


# In[26]:


y_pred=model.predict(x_test)


# In[27]:


accuracy=accuracy_score(y_test,y_pred)
report=classification_report(y_test,y_pred)


# In[28]:


print(f'Accuracy:{accuracy}')
print(f'Classification_report:{report}')


# In[29]:


print(df['genre'].value_counts())
print(df['genre'].unique()) 


# In[30]:


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# In[31]:


model=RandomForestClassifier()
model.fit(x_train,y_train)


# In[32]:


y_pred=model.predict(x_test)


# In[33]:


accuracy=accuracy_score(y_test,y_pred)
report=classification_report(y_test,y_pred)


# In[34]:


print(f'Accuracy:{accuracy}')
print(f'Classification_report:{report}')


# In[35]:


MCC=matthews_corrcoef(y_test,y_pred)
print('The matthews_corrcoef is {}'.format(MCC))


# In[36]:


from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay


# In[37]:


y_pred=model.predict(x_test)


# In[38]:


cm=confusion_matrix(y_test,y_pred,labels=model.classes_)


# In[39]:


cm=confusion_matrix(y_test,y_pred)


# In[40]:


print(f'Confusion_matrix:{cm}')


# In[41]:


disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model.classes_)


# In[42]:


disp.plot(cmap=plt.cm.Reds)


# In[ ]:




