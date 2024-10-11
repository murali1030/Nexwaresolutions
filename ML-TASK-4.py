#!/usr/bin/env python
# coding: utf-8

# # ML-TASK-4

# # SPAM SMS DETECTION

# Build a model that can classify SMS messages as spam or 
# legitimate. Use techniques like TF-IDF or word embeddings with 
# classifiers like Naive Bayes, Logistic Regression, or Support Vector 
# Machines to identify spam messages

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import pandas as pd

df = pd.read_csv(r"C:\Users\mural\Downloads\smsspamcollection.tsv", sep='\t', encoding='latin1', error_bad_lines=False)


# In[3]:


df.columns


# In[4]:


df=df.drop(columns=['length','punct'],axis=1)


# In[5]:


df.head()


# In[6]:


df.isnull().sum()


# In[7]:


df.duplicated().sum()


# In[8]:


df=df.drop_duplicates()


# In[9]:


df.duplicated().sum()


# In[10]:


df.head()


# In[11]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string


# In[12]:


nltk.download('stopwords')
nltk.download('punkt')


# In[13]:


def preprocess_text(text):
    text = text.lower()   
    text = text.translate(str.maketrans('', '', string.punctuation)) 
    tokens = word_tokenize(text)  
    tokens = [word for word in tokens if word not in stopwords.words('english')]  
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)


# In[14]:


df['processed_message']=df['message'].apply(preprocess_text)
print(df[['message','processed_message']])


# In[15]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer()
X=tfidf_vectorizer.fit_transform(df['processed_message'])
y=df['label']


# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[18]:


X_train


# In[19]:


from sklearn.naive_bayes import MultinomialNB


# In[20]:


model=MultinomialNB()
model.fit(X_train,y_train)


# In[21]:


from sklearn.metrics import accuracy_score,classification_report


# In[22]:


y_pred=model.predict(X_test)


# In[23]:


accuracy=accuracy_score(y_test,y_pred)


# In[24]:


report=classification_report(y_test,y_pred)


# In[25]:


print(f'Accuracy:{accuracy}')
print(f'Classification Report:n/',report)


# In[26]:


from sklearn.linear_model import LogisticRegression


# In[27]:


model=LogisticRegression()
model.fit(X_train,y_train)


# In[28]:


y_pred=model.predict(X_test)


# In[29]:


accuracy=accuracy_score(y_test,y_pred)
report=classification_report(y_test,y_pred)


# In[30]:


print(f'Accuracy:{accuracy}')
print(f'Classification Report:n/',report)


# In[31]:


from sklearn.ensemble import  RandomForestClassifier,GradientBoostingClassifier


# In[32]:


model=RandomForestClassifier()
model.fit(X_train,y_train)


# In[33]:


y_pred=model.predict(X_test)


# In[34]:


accuracy=accuracy_score(y_test,y_pred)
report=classification_report(y_test,y_pred)


# In[35]:


print(f'Accuracy:{accuracy}')
print(f'Classification Report:n/',report)


# In[36]:


model=GradientBoostingClassifier()
model.fit(X_train,y_train)


# In[37]:


y_pred=model.predict(X_test)


# In[38]:


accuracy=accuracy_score(y_test,y_pred)
report=classification_report(y_test,y_pred)


# In[39]:


print(f'Accuracy:{accuracy}')
print(f'Classification Report:n/',report)


# In[40]:


from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,matthews_corrcoef


# In[41]:


y_pred=model.predict(X_test)


# In[42]:


cm=confusion_matrix(y_test,y_pred,labels=model.classes_)


# In[43]:


print(f'Confusion Matrix: {cm}')


# In[44]:


disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model.classes_)


# In[46]:


disp.plot(cmap=plt.cm.Reds)


# In[47]:


MCC=matthews_corrcoef(y_test,y_pred)
print('The matthews_corrcoef is {}'.format(MCC))


# In[ ]:




