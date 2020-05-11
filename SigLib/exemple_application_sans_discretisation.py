import discretisation
import clusterisation
import signature
import misc

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelEncoder
import pandas as pd 
import json

print("\n\n\n========== INITIALISATION DES VARIABLES DU PROJET ==========")

project_name = 'mnist_3layers_012_345_nodisc'
train_values = [0,1,2]
test_values=[3,4,5]


print("\n\n\n========== PREPARATION DES DONNEES ==========")

#utilisation du jeu de données mnist
(X_train_base, y_train_base), (X_test_base, y_test_base) = mnist.load_data()

X_train = X_train_base
y_train = y_train_base

X_test = X_test_base
y_test = y_test_base

X_train = X_train.reshape(len(X_train), 784)
X_train = X_train.astype('float32')

X_test = X_test.reshape(len(X_test), 784)
X_test = X_test.astype('float32')

#met les valeurs des pixels entre 0 et 1
X_train /= 255
X_test /= 255
     
X_train_values, y_train_values = misc.get_dataset_values(train_values, X_train, y_train, limit=[2000,2000,2000])  
X_test_values, y_test_values = misc.get_dataset_values(test_values, X_test, y_test, limit=[100,100,100])

train_X=np.asarray(X_train_values)
test_X = np.asarray(X_test_values)

train_y=y_train_values

encoder = LabelEncoder()

train_y=encoder.fit_transform(train_y)

train_y=keras.utils.to_categorical(train_y, len(train_values))


print("\n\n\n========== CREATION DU MODELE ==========")

model = Sequential()
model.add(Dense(512, input_dim = 784 , activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(len(train_values), activation='softmax'))

model.compile(loss = 'binary_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )


print("\n\n\n========== APPRENTISSAGE ==========")

model.fit(train_X, train_y, epochs = 15, batch_size = 100)


print("\n\n\n============ GENERATION DES SIGNATURES D'APPRENTISSAGE ========")

#extraction et sauvegarde des signatures d'apprentissage
train_signatures = signature.generate_train_signatures(train_X, train_y, model, train_values, project_name)

#à décommenter si les signatures d'apprentissage ont déja été sauvegardées : pas besoin de réapprendre le modèle
#train_signatures = signature.get_train_signatures(model, project_name)


print("\n\n\n============ CLUSTERISATION DES SIGNATURES D'APPRENTISSAGE ========")

clusterized_training_signatures, layers_kmeans = clusterisation.clusterize_training_signatures(5, train_signatures, train_values)


print("\n\n\n============ GENERATION DES SIGNATURES DE TEST ========")

test_signatures = signature.generate_test_signatures(model, project_name, test_X, y_test_values, train_values, test_values)

#à décommenter si les signatures de test ont déja été sauvegardées
#test_signatures = signature.get_test_signatures(model, project_name, test_values)


print("\n\n\n============ CLUSTERISATION DES SIGNATURES DE TEST ========")

clusterisation.clusterize_test_signatures(test_signatures, layers_kmeans, test_values, project_name)


print("\n\n\n============ GENERATION DES DONNEES POUR L'INTERFACE ========")

misc.generate_interface_data(project_name, model, clusterized_training_signatures)
    