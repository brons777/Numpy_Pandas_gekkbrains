#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")


# ### Задание_1

# In[5]:


x = [1, 2, 3, 4, 5, 6, 7]
y = [3.5, 3.8, 4.2, 4.5, 5, 5.5, 7]


# In[6]:


plt.plot(x, y)
plt.show


# In[8]:


fig, ax = plt.subplots()
ax.scatter(x, y, c ='red')


# ### Задание_2

# In[9]:


t = np.linspace(0, 10, 51)
t


# In[11]:


f = np.cos(t)
f


# In[17]:


plt.plot(t, f, color = 'green')
plt.title('График f(t)')
plt.xlabel('Значения t')
plt.ylabel('Значения f')
plt.axis([0.5, 9.5, -2.5, 2.5])
plt.show


# ### Задание_3

# In[18]:


x = np.linspace(-3, 3, 51)
x


# In[21]:


y1 = x ** 2
y2 = 2 * x + 0.5
y3 = -3 * x -1.5
y4 = np.sin(x)


# In[41]:


fig, ax = plt.subplots(nrows = 2, ncols = 2)
ax[0, 0].plot(x, y1)
ax[0, 1].plot(x, y2)
ax[1, 0].plot(x, y3)
ax[1, 1].plot(x, y4)
ax[0, 0].set_title('График 1')
ax[0, 1].set_title('График 2')
ax[1, 0].set_title('График 3')
ax[1, 1].set_title('График 4')
fig.set_size_inches(8, 6)
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)

ax[0, 0].set_xlim([-5, 5])
plt.show


# ### Задание_4

# In[42]:


import pandas as pd


# In[43]:


plt.style.use('fivethirtyeight')


# In[44]:


df = pd.read_csv('creditcard.csv')
df.head()


# In[47]:


temp = df['Class'].value_counts()
temp


# In[53]:


temp.plot(kind = 'bar')
plt.show


# In[55]:


temp.plot(kind = 'bar', logy = True)
plt.show


# In[63]:


v1_1=df.set_index('Class')['V1'].filter(like='1', axis=0)
v1_1=v1_1.reset_index()
v1_1=v1_1.drop('Class', axis=1)

v1_0=df.set_index('Class')['V1'].filter(like='0', axis=0)
v1_0=v1_0.reset_index()
v1_0=v1_0.drop('Class', axis=1)
v1_0


# In[70]:


plt.hist(v1_1['V1'], bins=20, color='grey', edgecolor='black', density = 0.5, orientation = 'horizontal',alpha=0.5)
plt.hist(v1_0['V1'], bins=20, color='red', edgecolor='black', density = 0.5, orientation = 'horizontal',alpha=0.5)
plt.plot()
plt.xlabel('Class')
plt.legend(labels=['Class 0', 'Class 1'])

