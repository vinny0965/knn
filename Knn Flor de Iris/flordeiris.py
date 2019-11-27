#!/usr/bin/env python
# coding: utf-8

# # Carregando Bibliotecas

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn

#get_ipython().run_line_magic('matplotlib', 'inLine')


# # Carregando os dados do arquivo csv em um dataFrame

# In[2]:


df = pd.read_csv('iris.csv')


# # Coluna dos dados

# In[3]:


df.columns

# # vendo os dados

# In[4]:
print("Coluna de dados em CSV",df)

# # Descrevendo os Dados

# In[5]:

df.describe()

# ### Dispersão dos dados

# In[6]:

sb.pairplot(df, hue='target')

# ### Selecionando as "features" para classificação em um array NUMPY

# In[7]:

x = np.array(df.drop('target',1))
print(x)

# ### Selecionando as classes para uma classificação em um array NUMPY

# In[8]:

y = np.array(df.target)
y

# # Importando o KNN

# In[9]:

from sklearn.neighbors import KNeighborsClassifier

# # Criando um Classificador

# In[10]:

knn = KNeighborsClassifier(n_neighbors=3)

# # Treinando o classificador

# In[11]:

knn.fit(x,y)

# # Predizendo um tipo de flor iris

# ## Passar o vetor com as caracteristicas da flor

# In[12]:

print("O tipo de flor é uma",knn.predict([[3.0,3.1,3.0,2.0]]))

# # Acuracia

# In[13]:

print("A acurácia é",knn.score(x,y))


# In[ ]:




