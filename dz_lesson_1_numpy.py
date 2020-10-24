#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# ### Numpy. Задание_1

# In[3]:


a = np.array([[1,6], [2,8], [3,11], [3, 10], [1, 7]])


# In[5]:


a


# In[8]:


mean_a = np.mean(a, axis=0)
mean_a


# ### Numpy. Задание_2

# In[10]:


a_centered = np.subtract(a, mean_a)
a_centered


# ### Numpy. Задание_3

# In[13]:


a_centered_sp = np.dot(a_centered[:,0], a_centered[:,1])
a_centered_sp


# In[14]:


a_cov = a_centered_sp / 4
a_cov


# In[ ]:




