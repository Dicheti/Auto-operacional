#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 11:39:31 2018

@author: dicheti
"""

#### notas
##   requisitos
##   Theano
##   pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
##   Tensorflow
##   pip install tensorflow
##   Keras
#    pip install --upgrade keras

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense






## \\\\\\\\\ IMPORTAR BANCO DE DADOS  \\\\\\\\

dataset = pd.read_csv('BANCO_ROTA.csv')

## seleção das colunas de partida
x = dataset.iloc[:, 1:5].values

## seleção das colunas de treinamento (boole)
y = dataset.iloc[:, 5].values


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1,)

## \\\\\ \\\ SIMULAR SITUACAO PARA TREINAMENTO \\\\\\
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)



## \\\\ MONTAGEM DA REDE NEURAL


##inicialização
classifier = Sequential()
## camada de entrada e primeira camada oculta
classifier.add(Dense(units = 200, kernel_initializer = 'uniform', activation = 'relu', input_dim = 4))
## segunda camada oculta
classifier.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'relu'))
## camada de saida 
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
## formatacao do resultado
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
## criando modelo de aprendizagem
classifier.fit(x_train, y_train, batch_size = 35, nb_epoch = 80000)


## \\\\\\ FAZER PREVISÃO \\\\\\\\\

y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.8)
##  \\\\\\ CRIAR MATRIZ \\\\\\

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
