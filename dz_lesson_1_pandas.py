#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# ### pandas. Задание_1

# In[3]:


d = {'author_id': [1, 2, 3], 'author_name': ['Тургенев', 'Чехов', 'Островский']}
authors = pd.DataFrame(data = d)
authors


# In[4]:


b = {'author_id': [1, 1, 1, 2, 2, 3, 3],
     'book_title': ['Отцы и дети', 'Рудин', 'Дворянское гнездо', 'Толстый и тонкий', 'Дама с собачкой', 'Гроза', 'Таланты и поклонники'],
     'price': [450, 300, 350, 500, 450, 370, 290]}
book = pd.DataFrame(data = b)
book


# ### pandas. Задание_2

# In[7]:


author_price = pd.merge(authors, book, on = 'author_id', how = 'inner')
author_price


# ### pandas. Задание_3

# In[18]:


top5 = author_price.nlargest(5, 'price')
top5.reset_index(inplace = True, drop=True)
top5


# ### pandas. Задание_4

# In[24]:


authors_stat = author_price.groupby('author_name').agg({'price': ['min', 'max', 'mean']})
authors_stat = authors_stat.rename(columns={'min':'min_price', 'max':'max_price', 'mean':'mean_price'})
authors_stat


# ### pandas. Задание_5

# In[28]:


author_price['cover'] = ['твердая', 'мягкая', 'мягкая', 'твердая', 'твердая', 'мягкая', 'мягкая']
author_price


# In[33]:


import numpy as np
book_info = pd.pivot_table(author_price, values='price', index=['author_name'], columns=['cover'], aggfunc=np.sum)
book_info['мягкая'].fillna(0, inplace = True)
book_info['твердая'].fillna(0, inplace = True)
print(book_info)


# In[34]:


book_info.to_pickle('book_info.pkl')


# In[35]:


book_info2 = pd.read_pickle('book_info.pkl')


# In[36]:


book_info.equals(book_info2)


# In[ ]:




