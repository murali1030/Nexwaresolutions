#!/usr/bin/env python
# coding: utf-8

# # ML-TASK-3

# # CUSTOMER CHURN PREDICTION
# 

# Develop a model to predict customer churn for a subscriptionbased service or business. Use historical customer data, including 
# features like usage behavior and customer demographics, and try 
# algorithms like Logistic Regression, Random Forests, or Gradient 
# Boosting to predict churn. 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 


# In[2]:


df=pd.read_csv(r"C:\Users\mural\Downloads\WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.head()


# In[3]:


df.shape


# In[4]:


df.size


# In[5]:


df.dtypes


# In[6]:


df.info()


# In[7]:


df['SeniorCitizen'].describe()


# In[8]:


df['tenure'].describe()


# In[9]:


df['MonthlyCharges'].describe()


# In[10]:


df['TotalCharges'].describe()


# In[11]:


df.isnull().sum()


# In[12]:


df.duplicated().sum()


# In[13]:


df['PaymentMethod'].value_counts()


# In[14]:


df.gender.value_counts()


# In[15]:


df['Contract'].value_counts()


# In[16]:


### univaraint analysis 
plt.figure(figsize=(10, 4))
plt.figure(figsize=(10,4))
plt.figure(figsize=(10,6))
sns.countplot(data=df,x='PaymentMethod')
sns.countplot(data=df,x='PaymentMethod')


# In[17]:


sns.histplot(data=df,x='Contract')


# In[18]:


df['gender'].value_counts().plot.pie(autopct='1%.1f%%')
plt.legend()


# In[19]:


sns.histplot(data=df,x='MonthlyCharges')


# In[20]:


sns.countplot(data=df,x='MultipleLines')


# In[21]:


pd.crosstab(df['gender'],df['SeniorCitizen']).plot(kind='bar')


# In[22]:


pd.crosstab(df['Contract'],df['DeviceProtection']).plot(kind='bar')


# In[23]:


df.head()


# In[24]:


pd.crosstab(df['Contract'],df['TechSupport']).plot(kind='bar')


# In[25]:


pd.crosstab(df['Contract'],df['StreamingTV']).plot(kind='bar')


# In[26]:


pd.crosstab(df['Contract'],df['StreamingMovies']).plot(kind='bar')


# In[27]:


pd.crosstab(df['Contract'],df['PaymentMethod']).plot(kind='bar')


# In[28]:


pd.crosstab(df['Contract'],df['PaperlessBilling']).plot(kind='bar')


# In[29]:


df['Contract'].value_counts()


# In[30]:


sns.pairplot(df)


# In[31]:


df[['SeniorCitizen','tenure']].corr()


# In[32]:


def func(x):
    if x.isspace()==True:
        return 0
    else:
        return float(x)


# In[33]:


df['TotalCharges']= df['TotalCharges'].apply(func)


# In[34]:


sns.heatmap(df.corr(),annot=True)


# In[35]:


df['Churn'].value_counts().plot(kind='bar',color='pink')


# In[36]:


print(df.columns)


# In[37]:


df=df.drop(columns=['customerID','tenure','SeniorCitizen','MonthlyCharges','TotalCharges'],axis=1)


# In[38]:


df.head()


# In[39]:


from sklearn.preprocessing import LabelEncoder,StandardScaler


# In[40]:


le=LabelEncoder()
df=df.apply(le.fit_transform)
df.head()
le=LabelEncoder()
df=df.apply(le.fit_transform)
df.head()
le=LabelEncoder()
df=df.apply(le.fit_transform)


# In[41]:


X=df.drop(['Churn'],axis=1)
y=df['Churn']


# In[42]:


PredictorScaler=StandardScaler()


# In[43]:


PredictorScalerFit=PredictorScaler.fit(X)


# In[44]:


X=PredictorScaler.fit_transform(X)


# In[45]:


from sklearn .model_selection import train_test_split


# In[46]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[47]:


X_train


# In[48]:


X_test


# In[49]:


y_train


# In[50]:


y_test


# In[51]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[52]:


from sklearn.linear_model import LogisticRegression
from sklearn .metrics import accuracy_score,classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn .tree import DecisionTreeClassifier


# In[53]:


models={'GradientBoostingClassifier':GradientBoostingClassifier(),
       'RandomForestClassifier':RandomForestClassifier(),
       'KNeighborsClassifier':KNeighborsClassifier(),
       'GaussianNB':GaussianNB(),
       'LogisticRegression':LogisticRegression(),
       'DecisionTreeClassifier':DecisionTreeClassifier()}


# In[54]:


for model_name,model in models.items():
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    class_report=classification_report(y_test,y_pred)
    print(f'Model:{model_name}')
    print(f'Accuracy:{accuracy}')
    print(f'ClassificationReport:{class_report}')


# In[55]:


from sklearn.pipeline import Pipeline
from sklearn .ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# In[56]:


pipeline_rfc= Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])


# In[57]:


# Define the parameter grid for hyperparameter tuning
param_grid_rfc= {
    'classifier__n_estimators': [50, 100, 150],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10]
}


# In[58]:


# Perform grid search with cross-validation
grid_search= GridSearchCV(pipeline_rfc, param_grid_rfc, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)


# In[59]:


# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_


# In[60]:


y_pred=grid_search.best_estimator_.predict(X_test)


# In[61]:


# Evaluate the final model on the test set
accuracy = accuracy_score(y_test, y_pred)


# In[62]:


# Print the results
print(f"Best parameters: {best_params}")
print(f"Best cross-validated score: {best_score}")
print(f"Test set accuracy: {accuracy}")
print(f'Classification Report:n/{classification_report(y_test,y_pred)}')


# In[63]:


from sklearn.pipeline import Pipeline
from sklearn. linear_model import LogisticRegression


# In[64]:


pipeline_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier',LogisticRegression(random_state=42,max_iter=1000))
])


# In[65]:


param_grid_lr = {
    'classifier__penalty': ['l1', 'l2'],
    'classifier__C': [0.01, 0.1, 1.0, 10, 100],
    'classifier__solver': ['liblinear',]
}


# In[66]:


from sklearn.model_selection import GridSearchCV


# In[67]:


grid_search=GridSearchCV(pipeline_lr,param_grid_lr,cv=5,n_jobs=-1,verbose=2)


# In[68]:


grid_search.fit(X_train,y_train)


# In[69]:


best_params=grid_search.best_params_
best_score=grid_search.best_score_


# In[70]:


y_pred=grid_search.best_estimator_.predict(X_test)


# In[71]:


accuracy=grid_search.score(X_test,y_test)


# In[72]:


print(f'Best paramters :{best_params}')
print(f'Best cross-validated score:{best_score}')
print(f' Test set accuracy:{accuracy}')
print(f'Classification Report:n/{classification_report(y_test,y_pred)}')


# In[73]:


pipeline_gb=Pipeline([
    ('scaler',StandardScaler()),
    ('classifier',GradientBoostingClassifier(random_state=42))
])


# In[74]:


param_grid_gb={
    'classifier__n_estimators':[50,100,150],
    'classifier__learning_rate':[0.001,0.01,0.1,1],
    'classifier__max_depth':[2,5,10]
}


# In[75]:


grid_search_gb=GridSearchCV(pipeline_gb,param_grid_gb,cv=5,n_jobs=-1,verbose=2)
grid_search_gb.fit(X_train,y_train)


# In[76]:


best_params=grid_search_gb.best_params_
best_score=grid_search_gb.best_score_


# In[77]:


y_pred=grid_search_gb.best_estimator_.predict(X_test)


# In[78]:


accuracy=accuracy_score(y_test,y_pred)


# In[79]:


print(f'Best Paramater:{best_params}')
print(f'Cross-validation score:{best_score}')
print(f'Accuracy:{accuracy}')
print(f'Classifcation Report:n/{classification_report(y_test,y_pred)}')


# In[80]:


from sklearn.metrics import confusion_matrix,matthews_corrcoef


# In[81]:


MCC=matthews_corrcoef(y_test,y_pred)
print('The matthews correlation coefficinet is {}'.format(MCC))


# In[82]:


LABELS=['1','0']
conf_matrix=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix,xticklabels=LABELS,
           yticklabels=LABELS,annot=True,fmt='d')
plt.title('Confusion_matrix')
plt.xlabel('Predited class')
plt.ylabel('True class')
plt.show()


# In[ ]:





# In[ ]:




