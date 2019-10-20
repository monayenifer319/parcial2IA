#!/usr/bin/env python
# coding: utf-8

# In[69]:


from numpy import exp, array, random, dot
from random import choice
import matplotlib.pyplot as plt


# In[74]:


class RedNeuronal():
    def __init__(self):
        self.pesos_signaticos = 2 * random.random((3,1)) - 1
        
    #funcion de activacion
    def __sigmoide(self, x):
        return 1 / (1 + exp(-x))
    
    def __sigmoide_derivado(self, x):
        return x * (1 - x)
    
    #entrenamiento
    def entrenamiento(self,entradas,salidas,numero_iteraciones):
        errores = []
        resulEsperados = []
        for i in range(numero_iteraciones):
            salida = self.pensar(entradas)
            error = salidas - salida
            ajuste = dot(entradas.T, error * self.__sigmoide_derivado(salida))
            self.pesos_signaticos += ajuste
            
            esperado = choice(salidas)  
            errores.append(error)
            resulEsperados.append(esperado)
            
        plt.plot(error,'-',color='red')
        plt.plot(resulEsperados,'*', color='green')
           # print(error)
            
    def pensar(self,entrada):
        return self.__sigmoide(dot(entrada, self.pesos_signaticos))
    
 # set de entrenamiento
if __name__ == '__main__':
    red_neuronal = RedNeuronal()
    entradas = array([[1,1,1],
                      [0,1,0], 
                      [0,1,1], 
                      [1,0,1]])
    
    salidas = array([[1,1,0,1]]).T
    
    red_neuronal.entrenamiento(entradas,salidas,500)
    
    print(red_neuronal.pesos_signaticos)
    print(red_neuronal.pensar(array([1,0,0])))
        


# In[ ]:





# In[ ]:




