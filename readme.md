# Que se passe-t-il dans mon réseau de neurones profond ?
 
## Auteurs
 
DIDIER Adrien - MARTIN Loïc - PORTAL Pierre - TROUCHE Aurélien
 
## Description
 
Ceci est une librairie permettant de faciliter l'analyse des signatures des fonctions d'activation des neurones dans une réseau de neurones profond. 

Elle inclut :

- des fonctions Python de sauvegarde, de discrétisation et de clusterisation des valeurs des fonctions d'activation des données d'apprentissage et des données à prédire

- une interface web utilisant Node/HTML/Javascript permettant de visualiser les données générées

Cette librairie est le fruit d'un projet de TER en Master 1 Informatique à l'université de Montpellier, elle est censée fonctionner pour tout type de données en entrée. 

Nous avons essentiellement utilisé la base de données MNIST comme jeu de données et nous fournissons quelques fonctions permettant de faciliter l'utilisation de celle-ci avec notre librairie.
 
## Installation

### Dépendances Python 

Spyder 3.3.6

Python 3.7.4

Pandas 0.25.1

Scikit-learn 0.21.3

Tensorflow 1.15.0

Keras 2.2.4

Numpy 1.16.5


### Dépendances NodeJS

Google Chrome 81 (81.0.4044.129)

express 4.17.1

csv-parse 4.9.1

mustache 4.0.1


### Pour faire fonctionner l'interface
- Installer node et les dépendances nécessaires
	
- Executer la commande "node index.js" dans le dossier de l'interface
	
- Accéder à l'URL "http://localhost:2400/"

 
## Mode d’emploi

Deux fichiers d'exemple sont présents dans la librairie.

Dépendances
```python
import clusterisation
import signature
import misc
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelEncoder
```

Initialisation des variables du projet

```python
project_name = 'mnist_complet_3layers'
train_values = [0,1,2,3,4,5,6,7,8,9]
test_values=[0,1,2,3,4,5,6,7,8,9]
```

Préparation des données

```python
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
     
X_train_values, y_train_values = misc.get_dataset_values(train_values, X_train, y_train)  
X_test_values, y_test_values = misc.get_dataset_values(test_values, X_test, y_test)

train_X=np.asarray(X_train_values)
test_X = np.asarray(X_test_values)

train_y=y_train_values

encoder = LabelEncoder()

train_y=encoder.fit_transform(train_y)

train_y=keras.utils.to_categorical(train_y, len(train_values))
```

Création et apprentissage du modèle

```python
model = Sequential()
model.add(Dense(512, input_dim = 784 , activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(len(train_values), activation='softmax'))

model.compile(loss = 'binary_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )

model.fit(train_X, train_y, epochs = 15, batch_size = 100)
```

Generation des signatures d'apprentissage
```python
train_signatures = signature.generate_train_signatures(train_X, train_y, model, train_values, project_name)
```
Clusterisation des signatures d'apprentissage
```python
clusterized_training_signatures, layers_kmeans = clusterisation.clusterize_training_signatures(10, train_signatures, train_values)
```

Generation des signatures de test
```python
test_signatures = signature.generate_test_signatures(model, project_name, test_X, y_test_values, train_values, test_values)
```

Clusterisation des signatures de test
```python
clusterisation.clusterize_test_signatures(test_signatures, layers_kmeans, test_values, project_name)
```

Generation des données pour l'interface
```python
misc.generate_interface_data(project_name, model, clusterized_training_signatures)
```
On glisse le dossier ${project_name}_DATA dans le dossier data de l'interface. On peut maintenant visualiser les résultats à l'aide de l'interface :
```cmd
node index.js
 ```
 
## Démonstration
 
Une présentation de notre projet est disponible à l'adresse suivante https://www.youtube.com/watch?v=Jitaq5fWRUA
 
[![Watch the video](https://github.com/AdrienL3/TER_RNP_M1/blob/master/interface.png)](https://www.youtube.com/watch?v=Jitaq5fWRUA)
 
## Rapport
 
Vous pouvez accéder au rapport de ce projet depuis ce github.
 
 

 
 
 
 
 
 
 





