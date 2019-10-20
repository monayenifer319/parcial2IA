#!/usr/bin/env python
# coding: utf-8

# In[6]:


from random import choice
from numpy import array, dot, random
import matplotlib.pyplot as plt


# In[7]:


#Funcion de Activación
activacion = lambda x:0 if x < 0 else 1


# In[8]:


#Set  Entrenamiento
entrenamiento = [
    (array([0,0,1]),0),
    (array([0,1,1]),1),
    (array([1,0,1]),1),
    (array([1,1,1]),1)
]


# In[21]:


w = random.rand(3)
errores = []
esperados = []
bahias = 0.5
n = 500


# In[22]:


#Entrenamiento
for i in range(n):
    x, esperado = choice(entrenamiento)
    resultado = dot(w, x)
    esperados.append(esperado)
    error = esperado - activacion(resultado)
    errores.append(error)
    #Ajuste
    w += bahias * error * x


# In[23]:


for x, _ in entrenamiento:
    resultado = dot(w, x)
    print("{}: {} -> {}".format(x[:3], resultado, activacion(resultado)))


# In[24]:


plt.plot(errores,'-',color='red')
plt.plot(esperados,'*', color='green')


# In[ ]:




