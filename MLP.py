# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 21:43:43 2019

@author: Avell
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import timeit
import math
import random
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

def OHE(data, categorical_index):
    labelencoder_X = LabelEncoder()
    data[:, categorical_index] = labelencoder_X.fit_transform(data[:,categorical_index])
    onehotencoder = OneHotEncoder(categorical_features = [categorical_index])
    data = onehotencoder.fit_transform(data).toarray()
    return data

def Encoder(data): #função para codificar os dados em strings para numeros para poderem ser calculados no perceptron
    enc = LabelEncoder()
    label_encoder = enc.fit(data)
    y = label_encoder.transform(data) + 1
    return y

#X = np.array([
#    [0, 1],
#    [1, 0],
#    [1, 1],
#    [0, 0]
#])
#y = np.array([
#    [1],
#    [1],
#    [0],
#    [0]
#])

#X = np.array([
#    [0, 0, 0],
#    [0, 0, 1],
#    [0, 1, 0],
#    [0, 1, 1],
#    [1, 0, 0],
#    [1, 0, 1],
#    [1, 1, 0],
#    [1, 1, 1]
#])
#y = np.array([
#    [0],
#    [1],
#    [1],
#    [0],
#    [1],
#    [0],
#    [0],
#    [1]
#])

#le dados do dataset 
data = pd.read_table('Iris.txt', decimal  = ",")
x = OHE(np.asarray(data.iloc[:,:].values), 4)
y = x[:,:3]
X = x[:,3:]


# escala as unidades
X = X/np.amax(X, axis=0)
y = y 

class Neural_Network(object):
  def __init__(self):
    #parametros da rede
    self.inputSize = 4
    self.outputSize = 3
    self.hiddenSize = 5
    self.hiddenLayers = 1
    self.Ni = 0.1

    #Inicializa a rede com pesos aleatórios baseados no numero de entradas, camadas oculta, numero de neuronios e saidas
    self.W = [0]*(self.hiddenLayers+1)
    self.W[0] = np.random.randn(self.inputSize, self.hiddenSize)
    for i in range(1, self.hiddenLayers):
        self.W[i] = np.random.randn(self.hiddenSize, self.hiddenSize)
    self.W[self.hiddenLayers] = np.random.randn(self.hiddenSize, self.outputSize)
    

  def forward(self, X):
    #propaga a entrada para frente na rede.
    self.net = [0]*(len(self.W))
    self.fnet = [0]*(len(self.W))
    self.net[0] = np.dot(X, self.W[0])
    self.fnet[0] = self.sigmoid(self.net[0])
    for k in range(1, len(self.net)):
        self.net[k] = np.dot(self.fnet[k-1], self.W[k])
        self.fnet[k] = self.sigmoid(self.net[k])

    return self.fnet[k]

  def sigmoid(self, s):
    # função de ativação 
    return 1/(1+np.exp(-s))

  def sigmoidDer(self, s):
    #derivada da sigmoide
    return s * (1 - s) 

  def backward(self, X, y, o):
    # propaga os erros paratrás na rede
    self.o_error = (y - o)*self.sigmoidDer(o) # aplica a diferenciaçao da sigmoide no erro
    self.error = [0]*(len(self.fnet))
    self.error[len(self.error)-1] = (y - o)*self.sigmoidDer(o)


    for j in range(self.hiddenLayers-1, -1, -1): #calcula a propagação do erro para cada camada a partir da anterior
        self.error[j] = self.Ni * self.error[j+1].dot(self.W[len(self.W)-(self.hiddenLayers-j)].T) * self.sigmoidDer(self.fnet[j])

    
    self.W[0] += X.T.dot(self.error[0]) # ajusta o peso da primeira camada de acordo com a entrada
    for n in range(1, len(self.W)):
        self.W[n] += self.fnet[n-1].T.dot(self.error[n]) # ajusta os pesos de cada camada de acordo com o erro propagado
    

  def train (self, X, y):
    o = self.forward(X)
    self.backward(X, y, o)

NN = Neural_Network()
for i in range(500000): # treina a rede i vezes
  NN.train(X, y)
  
print("Input: " + str(X) )
print("Output Real: \n" + str(y) )
print("Output Predita: \n" + str(NN.forward(X)))
print("Erro médio: \n" + str(np.mean(np.square(y - NN.forward(X))))) # mean sum squared loss
print("\n")

y_pred = NN.forward(X)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred)