#!/usr/bin/env python
# coding: utf-8

# In[1]:


#IMPORT LIBRARIES
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#LOADING DATASET
df = pd.read_csv('Iris.csv')
df.head(10)


# In[3]:


#STATISTICS ABOUT THE DATA
df.describe()


# In[4]:


#INFORMAION ABOUT DATATYPE
df.info()


# In[5]:


df['species'].value_counts()


# In[6]:


#PREPROCESSING THE DATASET
df.isnull().sum()


# In[7]:


#EXPLORATORY DATA ANALYSIS
df['sepal_length'].hist()


# In[8]:


df['sepal_width'].hist()


# In[9]:


df['petal_length'].hist()


# In[10]:


df['petal_width'].hist()


# In[11]:


#SCATTER PLOT
colors = ['red','orange','blue']
species = ['setosa','versicolor','virginica']


# In[12]:


for i in range(3):
    x = df[df['species'] == species[i]]
    plt.scatter(x['sepal_length'],x['sepal_width'],c = colors[i], label = species[i])
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()


# In[13]:


for i in range(3):
    x = df[df['species'] == species[i]]
    plt.scatter(x['petal_length'],x['petal_width'],c = colors[i], label = species[i])
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend()


# In[14]:


for i in range(3):
    x = df[df['species'] == species[i]]
    plt.scatter(x['sepal_length'],x['petal_length'],c = colors[i], label = species[i])
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.legend()


# In[15]:


for i in range(3):
    x = df[df['species'] == species[i]]
    plt.scatter(x['sepal_width'],x['petal_width'],c = colors[i], label = species[i])
plt.xlabel('Sepal Width')
plt.ylabel('Petal Width')
plt.legend()


# In[16]:


#CORRELATION MATRIX
df.corr()


# In[17]:


corr = df.corr()
fig, ax = plt.subplots(figsize=(7,5))
sns.heatmap(corr, annot=True, ax=ax)


# In[18]:


#LABEL ENCODER
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[19]:


df['species'] = le.fit_transform(df['species'])
df.head()


# In[20]:


#MODEL TRAINING
from sklearn.model_selection import train_test_split
X = df.drop(columns=['species'])
Y = df['species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30)


# In[21]:


#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[22]:


model.fit(x_train, y_train)


# In[23]:


#PERFORMANCE METRICS
print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[24]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()


# In[25]:


model.fit(x_train, y_train)


# In[26]:


print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[27]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()


# In[28]:


model.fit(x_train, y_train)


# In[29]:


print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[ ]:




