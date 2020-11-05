#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd


# In[16]:


from sklearn.datasets import load_boston


# ### Задание_1

# In[80]:


boston = load_boston()


# In[81]:


boston.keys()


# In[82]:


data = boston.data
data


# In[83]:


feature_names = boston.feature_names
feature_names


# In[84]:


target = boston.target


# In[85]:


X = pd.DataFrame(data, columns=feature_names)
X.head()


# In[86]:


X.shape


# In[87]:


X.info()


# In[88]:


Y = pd.DataFrame(target, columns=["price"])


# In[89]:


Y.head()


# In[90]:


Y.info()


# In[91]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state = 42)


# In[92]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()


# In[93]:


lr.fit(X_train, Y_train)


# In[94]:


Y_pred = lr.predict(X_test)


# In[95]:


check_test = pd.DataFrame({
    "Y_test": Y_test["price"],
    "Y_pred": Y_pred.flatten(),
}, columns = ['Y_test', 'Y_pred'])

check_test.head(10)


# In[96]:


check_test["error"] = check_test["Y_pred"] - check_test["Y_test"]
check_test.head()


# In[97]:


from sklearn.metrics import r2_score
r2_score(Y_test, Y_pred)


# ### Задание_2

# In[98]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=1000, max_depth=12, random_state=42)


# In[99]:


model.fit(X_train, Y_train.values[:, 0])


# In[100]:


Y_pred = model.predict(X_test)
Y_pred.shape


# In[101]:


check_test = pd.DataFrame({
    "Y_test": Y_test["price"],
    "Y_pred": Y_pred.flatten(),
}, columns = ['Y_test', 'Y_pred'])

check_test.head(10)


# In[102]:


check_test["error"] = check_test["Y_pred"] - check_test["Y_test"]
check_test.head()


# In[103]:


r2_score(Y_test, Y_pred)


# Модель RandomForestRegressor предпочтительнее

# ### Задание_3

# In[105]:


get_ipython().run_line_magic('pinfo', 'RandomForestRegressor')


# In[106]:


model.feature_importances_.sum()


# In[107]:


print(model.feature_importances_)


# Наибольшую важность показывают признаки RM и LSTAT

# ### Задание_4

# In[109]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


# In[110]:


df = pd.read_csv('creditcard.csv')
df.head()


# In[111]:


df["Class"].value_counts(normalize=True)


# In[112]:


df.info()


# In[116]:


df.isnull().value_counts()


# In[117]:


pd.options.display.max_columns = 100


# In[118]:


df.head(10)


# In[119]:


X = df.drop("Class", axis=1)
X.head()


# In[120]:


y = df["Class"]
y.head()


# In[121]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100, stratify=y)


# In[122]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[123]:


parameters = [{'n_estimators': [10, 15],
'max_features': np.arange(3, 5),
'max_depth': np.arange(4, 7)}]


# In[124]:


clf = GridSearchCV(
    estimator=RandomForestClassifier(random_state=100),
    param_grid=parameters,
    scoring='roc_auc',
    cv=3,
)


# In[125]:


clf.fit(X_train, y_train)


# In[126]:


clf.best_params_


# In[134]:


y_pred_proba = clf.predict_proba(X_test)
print(y_pred_proba)


# In[128]:


y_pred_proba = y_pred_proba[:, 1]


# In[129]:


from sklearn.metrics import roc_auc_score


# In[130]:


roc_auc_score(y_test, y_pred_proba)


# ### Дополнительные задания

# ### Задание_1

# In[135]:


from sklearn.datasets import load_wine
data = load_wine()
data.keys()


# ### Задание_2

# In[136]:


type(data)


# In[137]:


data_keys=data["feature_names"]
data_keys


# ### Задание_3

# In[138]:


for line in data.DESCR.split('\n'):
    print(line)


# ### Задание_4

# In[139]:


np.unique(data["target"]).shape


# In[140]:


data["target_names"]


# ### Задание_5

# In[141]:


X = pd.DataFrame(data.data, columns=feature_names)
X.head()


# ### Задание_6

# In[142]:


X.shape


# In[143]:


X.info()


# ### Задание_7

# In[144]:


X["target"]=data["target"].astype(np.int64)
X.info()


# ### Задание_8

# In[147]:


X_cor=X.corr()
X_cor


# ### Задание_9

# In[148]:


high_corr=X_corr["target"]
high_corr=high_corr[np.abs(high_corr)>0.5].drop("target", axis=0)
high_corr=list(high_corr.index)
high_corr


# ### Задание_10

# In[149]:


X=X.drop("target", axis=1)
X.head()


# In[150]:


for i in high_corr:
    X[i+"_2"]=X[i]**2
X.head()


# In[151]:


X.describe()


# In[ ]:




