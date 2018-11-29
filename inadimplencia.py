# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""
#primeira rede, classificar inadimplentes e adimplentes 
##usar rede para perfilar pessoas com base no historico do banco

# / / / / /  P R É   P R O C E S S A M E N T O //  //  //  //

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# IMPORTAR O INTERVALO DE DADOS QUE IRÃO SERVIR PARA O TREINAMENTO
# matriz na variável X = DADOS DO PERFIL
# matriz na variável Y = INFORMAÇÃO À SER PREFISTA, OBJETIVO DO FORECAST!
# estrutura da seleção de dados:
### "[linha inicial:linha final, coluna inicial: coluna final ]"

dataset = pd.read_csv('NOME DO BANCO DE DADOS')
X = dataset.iloc[:, :].values
y = dataset.iloc[:, ].values

# CODIFICAÇÃO DAS VARIÁVEIS TEXTO EM NUMEROS 
# especificar dentro da variavel o intervalo a ser codificado
# enumerar as colunas em X1 X2 ETC... 

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# transformar as variaveis em intervalos binarios [coluna do binario]
# exemplo tres variaveis de texto codificadas em 0,1,2 viram tres colunas 0,1
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

#remover excesso de binarios exemplo abaixo remove a coluna 1
# se houverem tres colunas 0,1, uma variavel pode ser considerada 00
X = X[:, 1:]

## /  /  /  /  /  /  /  / OBJETO DE TREINAMENTO PADRONIZAÇÃO /  /  /
# criar variaveis/cenário de treinamento com os dados pré processados

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# ESCALAR E PADRONIZAR VARIAVEIS exemplo transformar valores de milhares para 
# intervalos proximos de 1 ou 0
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#  /  /  /  /  /   /  C O N F I G U R A R   R E D E / / / / / / / /

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

## OBJETO DE INICIALIZAÇÃO - COSTUMA DAR ERRO, REVISAR AS BIBLIOTECAS NECESSARIAS

classifier = Sequential()

# camada de entrada e primeira camada oculta da rede, unidades = neuronios,
# imput dim =numero de variaveis que entram na primeira camada 
# drop out é o numero de neuronios que serão desativados para evitar "viciar"a rede
# drop out = 0,1 significa 10% ignorados em cada rodada
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dropout(p = 0.1))

# segunda camada
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.1))

# camada de saida, o numero de neuronios "units" significa o numero de resultados
# no caso como é uma coluna deve ser uma camada de neuronios
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compilar a rede
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# definir parametros de treino (hiperparametros)
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

#  /  /  /  /  /  /  /  T E S T E  E  A V A L I A Ç Ã O / / / / 

# prever os resultados
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# fazer teste manual
"""Preencher o vetor abaixo com dados relativos a um perfil
fora da matriz de treinamento, seguir o modelo abaixo colunas separadas
por virgula"""
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40]])))
new_prediction = (new_prediction > 0.5)

# montar a "Confusion Matrix" 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# consultando se os hiperparametros estão corretos

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()

# melhorando e corrigindo hiperparametros
#alterar e testar tantos quantos der 

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_