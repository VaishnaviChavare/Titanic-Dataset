#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns


# In[3]:


df = pd.read_csv("titanic_train.csv")
df


# In[4]:


df.info()


# In[5]:


df.describe().T


# In[7]:


df.isnull().sum()


# In[8]:


df.head()


# In[19]:


# Drop the 'Cabin' column
df.drop(columns=["Cabin"], inplace=True)

# Fill missing Age with median
df["Age"].fillna(df["Age"].median(), inplace=True)

# Fill missing Embarked with mode
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# Fill missing Fare if exists (useful when testing with test set later)
df["Fare"].fillna(df["Fare"].median(), inplace=True)

# Convert Sex to numeric
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# One-hot encode Embarked
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

# Drop columns not needed for modeling
df.drop(columns=["Name", "Ticket", "PassengerId"], inplace=True)

# Final check
df.isnull().sum()


# In[20]:


df


# In[21]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[22]:


sns.countplot(data=df_cleaned, x="Survived"); plt.show()
sns.countplot(data=df, x="Sex", hue="Survived"); plt.show()
sns.countplot(data=df, x="Pclass", hue="Survived"); plt.show()
sns.kdeplot(data=df_cleaned, x="Age", hue="Survived", fill=True); plt.show()


# In[28]:


X = df.drop("Survived", axis=1)
y = df["Survived"]


# In[29]:


from sklearn.model_selection import train_test_split


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[31]:


from sklearn.linear_model import LogisticRegression


# In[32]:


model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)


# In[ ]:




