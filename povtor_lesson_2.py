#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# ### Задание_1

# In[2]:


a = np.arange(12, 24)
a


# ### Задание_2

# In[9]:


a1 = a.reshape(2, 6)
a2 = a.reshape(3, 4)
a3 = a.reshape(12, 1)
a4 = a.reshape(4, 3)
a5 = a.reshape(6, 2)
a1, a2, a3, a4, a5


# ### Задание_3

# In[16]:


a1 = a.reshape(2, -1)
a2 = a.reshape(3, -1)
a3 = a.reshape(6, -1)
a4 = a.reshape(-1, 4)
a5 = a.reshape(-1, 2)
a1, a2, a3, a4, a5


# ### Задание_4

# In[19]:


a = np.arange(10, 20)
a


# In[23]:


a1 = a.reshape(2,5)
a2 = a1.reshape(10, 1)
a2


# Одномерным назвать нельзя

# ### Задание_5

# In[24]:


a=np.random.randn(3,4)
a


# In[25]:


a.flatten()


# ### Задание_6

# In[29]:


a = np.arange(20, 0, -2)
a


# ### Задание_7

# In[30]:


b=np.arange(20, 1, -2).reshape(1,10)
b


# b - двумерный

# ### Задание_8

# In[34]:


a = np.zeros((2, 2))
a


# In[35]:


b = np.zeros((3, 2)) + 1
b


# In[39]:


v = np.vstack((a, b))
v


# In[40]:


v.size


# ### Задание_9

# In[44]:


a = np.arange(0, 12)
a


# In[45]:


A = a.reshape(4, 3)
A


# In[46]:


At = A.T
At


# In[49]:


B = np.dot(A, At)
B


# In[51]:


det_B = np.linalg.det(B)
det_B


# Обратной матрицы не существует, так как матрица вырожденная

# ### Задание_10

# In[52]:


np.random.seed(42)


# ### Задание_11

# In[54]:


c = np.random.randint(1, 16, 16)
c


# ### Задание_12

# In[62]:


C = c.reshape(4, 4)
C


# In[64]:


D = B + C * 10
D


# In[65]:


D_det = np.linalg.det(D)
D_det


# In[66]:


D_rank = np.linalg.matrix_rank(D)
D_rank


# In[70]:


D_inv = np.linalg.inv(D)
D_inv


# ### Задание_13

# In[71]:


D_inv=np.where(D_inv < 0, 0, 1)
D_inv


# In[72]:


E=np.where(D_inv==1, B, C)
E


# In[ ]:




