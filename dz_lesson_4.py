#!/usr/bin/env python
# coding: utf-8

# ### Задание_1

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


from sklearn.datasets import load_boston
boston = load_boston()


# In[3]:


X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = boston.target


# In[4]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[5]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)


# In[6]:


from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, learning_rate=250, random_state=42)
X_train_tsne = tsne.fit_transform(X_train_scaled)


# In[8]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1])
plt.grid()
plt.show()


# ### Задание_2

# In[9]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, max_iter=100, random_state=42)
kmeans_train = kmeans.fit_predict(X_train_scaled)


# In[14]:


plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c=kmeans_train)
plt.grid()
plt.show()


# In[13]:


print('average price:')
print('Кластер 0: {}'.format(y_train[kmeans_train == 0].mean()))
print('Кластер 1: {}'.format(y_train[kmeans_train == 1].mean()))
print('Кластер 2: {}'.format(y_train[kmeans_train == 2].mean()))
print('average CRIM:')
print('Кластер 0: {}'.format(X_train.loc[kmeans_train == 0, 'CRIM'].mean()))
print('Кластер 1: {}'.format(X_train.loc[kmeans_train == 1, 'CRIM'].mean()))
print('Кластер 2: {}'.format(X_train.loc[kmeans_train == 2, 'CRIM'].mean()))


# ### Задание_3

# In[15]:


kmeans_test = kmeans.predict(X_test_scaled)


# In[19]:


print('average price:')
print('Кластер 0: {}'.format(y_test[kmeans_test == 0].mean()))
print('Кластер 1: {}'.format(y_test[kmeans_test == 1].mean()))
print('Кластер 2: {}'.format(y_test[kmeans_test == 2].mean()))
print('average CRIM:')
print('Кластер 0: {}'.format(X_test.loc[kmeans_test == 0, 'CRIM'].mean()))
print('Кластер 1: {}'.format(X_test.loc[kmeans_test == 1, 'CRIM'].mean()))
print('Кластер 2: {}'.format(X_test.loc[kmeans_test == 2, 'CRIM'].mean()))


# In[ ]:




