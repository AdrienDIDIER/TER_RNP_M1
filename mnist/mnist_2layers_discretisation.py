# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 15:24:09 2020

@author: pierr
"""

#Que fait ce code ?
#créer et train le reseau de neurone avec les classes 0 et 1 (2 layers)
#extraire les fonctions d'activation dans le dossier mnist_512_512
#discrétiser les fonctions d'activation dans le dossier mnist_512_512
#clusteriser les fonctions d'activation discrétisées pour chaque layer et sauvegarde des nb_bins

#charger les classes 2, 3, 4, 5, 6, 7, 8, 9
#predict sur ces classes et sauvegarder les fonctions d'activation dans le dossier mnist_512X['numéro de la classe']_512X_datava


#Ce que je veux qu'il fasse
#créer et train le reseau de neurone avec des classes quelconque et un nombre de layer quelconque
#extraire les fonctions d'activation et les sauvegarder
#discrétiser les fonctions d'activation et les sauvegarder
#clusteriser les fonctions d'activation discrétisées pour chaque layer et sauvegarde des nb_bins

#ce programme permet GET PREDICTION SIGNATURES



import numpy as np
import pandas as pd 
import keras
import os
import re
import matplotlib.pyplot as plt
import collections as c

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical

from tensorflow.keras.datasets import mnist

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing, model_selection
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn import datasets
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.decomposition import PCA
#from sklearn.datasets import make_blob
from sklearn.preprocessing import StandardScaler

from scipy.cluster.hierarchy import dendrogram, linkage

from Levenshtein import distance

#Fonction permettant de récupérer les points qui ont été bien prédits

def get_goodXy (X,y,z):
        ynew = model.predict_classes(X)
        X_good =[]
        y_good=[]
        for i in range(len(X)):
            if z==0:
                #if (ynew[i]==[0] and y[i]==1) or (ynew[i]==[1] and y[i]==0):
                #print("ynew :",ynew[i][0])
                if (ynew[i][0]==0 and y[i]==1) or (ynew[i][0]==1 and y[i]==0):
                    print ("error prediction for X=%s, Predicted=%s, Real=%s"% (X[i], ynew[i][0], y[i]))
                else :
                    X_good.append(X[i])
                    y_good.append(y[i])
            else:
                if (ynew[i][0]!=0 and y[i]==0):
                    print ("error prediction for X=%s, Predicted=%s, Real=%s"% (X[i], ynew[i][0], y[i]))
                else :
                    X_good.append(X[i])
                    y_good.append(y[i])
        return X_good,y_good    
        

#Fonction permettant d'obtenir les fonctions d'activation de chaque couches
def get_result_layers(model,X):
    result_layers=[]
    for i in range (len(model.layers)-1):
        hidden_layers= keras.backend.function(
                [model.layers[0].input],   
                [model.layers[i].output,] 
                )    
        result_layers.append(hidden_layers([X])[0])  
    return result_layers

#def save_result_layers(filename,X,y,result_layers):
#    f = open(filename, "w")
#    for nb_X in range (len(X)):
#        #my_string=""
#        my_string=str(y[nb_X])+','
#        for nb_layers in range (len(model.layers)-1):
#            my_string+="<b>,"
#            for j in range (len(result_layers[nb_layers][nb_X])):
#                my_string+=str(result_layers[nb_layers][nb_X][j])+','
#            my_string+="</b>,"    
#        my_string=my_string [0:-1]
#        my_string+='\n'
#        f.write(my_string)    
#    f.close()
#    
    
#Fonction permettant de sauvegarder les fonctions d'activation de chaque couche dans un répertoire
def save_result_layers(project_name,X,y,result_layers):
    os.makedirs(project_name, exist_ok=True)
    for i in range(0, len(result_layers)):
        f = open(project_name + '/' + project_name + '_result_l' + str(i+1) + '.csv', "w")
        for nb_X in range (len(X)):
           my_string=str(y[nb_X])+','
           for j in range (len(result_layers[i][nb_X])):
               my_string+=str(result_layers[i][nb_X][j])+','
           my_string=my_string [0:-1]
           my_string+='\n'
           f.write(my_string)
        f.close()
    
#    f = open(filename, "w")
#    for nb_X in range (len(X)):
#        #my_string=""
#        my_string=str(y[nb_X])+','
#        for nb_layers in range (len(model.layers)-1):
#            my_string+="<b>,"
#            for j in range (len(result_layers[nb_layers][nb_X])):
#                my_string+=str(result_layers[nb_layers][nb_X][j])+','
#            my_string+="</b>,"    
#        my_string=my_string [0:-1]
#        my_string+='\n'
#        f.write(my_string)    
#    f.close()
     
     
    
#Fonction permettant de créer un csv pour chaque couches dans un répertoire   
#def get_directory_layers_from_csv(filename):
#    tokens=filename.split("_")
#    df = pd.read_csv(filename, sep = ',', header = None) 
#    
#    # creation d'un répertoire pour sauver tous les fichiers
#    repertoire=filename[0]
#    os.makedirs(repertoire, exist_ok=True)
#    string = repertoire+'/'+tokens[0]+'_'
#    f=[]
#    filenames=[]
#    for nb_tokens in range (1,len(tokens)-1):
#        name_file=string+'l'+str(nb_tokens)+'_'+tokens[nb_tokens]+'.csv'
#        f.append(open(name_file, "w"))
#        filenames.append(name_file)
#        
#        
#    # sauvegarde du dataframe dans une chaîne de caracteres
#    ch = df.to_string(header=False,
#                  index=False,
#                  index_names=False).split('\n')
#    vals = [','.join(ele.split()) for ele in ch]
#    
#    # sauvegarde dans des fichiers spécifiques par layer
#    for nb_exemples in range (len(vals)):
#        deb=str(df[0][nb_exemples])+','
#        # 1 ligne correspond à une chaine
#        s=vals[nb_exemples]
#        listoftokens=re.findall(r'<b>,(.+?),</b>', s)
#        nb_layers=len(listoftokens)
#        
#        for nb_token in range (nb_layers):
#            save_token=''
#            save_token=deb+str(listoftokens[nb_token])+'\n'
#            
#            f[nb_token].write(save_token)


#Fonction permettant de discrétiser nos valeurs présentes dans le fichier csv
def discretise_dataset(filename,bins):
    df = pd.read_csv(filename, sep = ',', header = None)
    #df = df.drop(df.columns[0], axis=1)
    #print(df.values)
    oneColumn = np.array(df[1])
    for i in range(2,df.shape[1]):
        oneColumn=np.append(oneColumn,np.array(df[i]))
    dfoneColumn=pd.DataFrame(oneColumn)
    nb_bins=bins
    dftemp=pd.DataFrame()
    dftemp[0]=pd.cut(dfoneColumn[0], bins=nb_bins, labels=np.arange(512), right=False)
    df_new=pd.DataFrame(df[0])
    nb_tuples=df.shape[0]
    j=0
    for i in range(1,df.shape[1]):
        df_new[i]=np.copy(dftemp[0][j:j+nb_tuples])
        j+=nb_tuples
    return df_new

def discretise_dataset_predict_true(filename,bins):
    df = pd.read_csv(filename, sep = ',', header = None)
    oneColumn = np.array(df[1])
    for i in range(2,df.shape[1]):
        oneColumn=np.append(oneColumn,np.array(df[i]))
    dfoneColumn=pd.DataFrame(oneColumn)
    nb_bins=bins
    dftemp=pd.DataFrame()
    dftemp[0], return_bins =pd.cut(dfoneColumn[0], bins=nb_bins, retbins=True, labels=np.arange(nb_bins), right=False)
    df_new=pd.DataFrame(df[0])
    nb_tuples=df.shape[0]
    j=0
    for i in range(1,df.shape[1]):
        df_new[i]=np.copy(dftemp[0][j:j+nb_tuples])
        j+=nb_tuples
    return df_new, return_bins

def recup_labels(df_cluster, val):
    cluster = []
    for i in range(len(df.values[:,0])):
        if df.values[:,0][i] == val:
            cluster.append(df_cluster.labels_[i])
    
    return cluster 


print("\n\n\n========== Apprentissage ==========")

#Utilisation du jeu de données mnist
(X_train_base, y_train_base), (X_test_base, y_test_base) = mnist.load_data()

X_train = X_train_base
y_train = y_train_base

X_test = X_test_base
y_test = y_test_base

X_train_sample=X_train[0:7840]
y_train_sample=y_train[0:7840]

X_train=X_train_sample
y_train=y_train_sample
X_train = X_train.reshape(7840, 784)
X_train = X_train.astype('float32')

X_train /= 255

#On récupère les objets appartenants aux classes 0 et 1
X_01=[]
y_01=[]
nb_X=0
for i in range(X_train.shape[0]):
    if (y_train[i]==0 or y_train[i]==1):
        
        nb_X+=1
        X_01.append(X_train[i])
        y_01.append(y_train[i])

       
train_X=np.asarray(X_01)

train_y=y_01

encoder = LabelEncoder()
train_y=encoder.fit_transform(train_y)


#Création de notre reseau de neurones
#La première couche possède 512 neurones, correspondant aux 512 pixels d'une image
input_dim = 784

model = Sequential()
model.add(Dense(512, input_dim = input_dim , activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )

model.fit(train_X, train_y, epochs = 40, batch_size = 100)


X_good,y_good=get_goodXy (train_X, train_y,0)

# Récupération des valeurs de tous les layers sauf le dernier
result_layers=get_result_layers(model,X_good)

project_name = 'neural_test'
    
# Sauvegarde du fichier
save_result_layers(project_name,X_good,y_good,result_layers)


# tri du fichier puis conversion en csv
#os.system (project_name+'_tmp > ' + project_name +'.csv')
# effacer le fichier intermédiaire
#os.system ('rm mnist_512_tmp')

#Création du(des) csv à partir du fichier tmp 
  
#Création de la matrice avec les données discrétisées
#df, df_bins = discretise_dataset_predict_true('mnist_512/mnist_l1_512.csv',512)
#df,nb_bins_l1 = discretise_dataset_predict_true('mnist_512_512/mnist_l1_512.csv',512)

#df1_save = df
#
#df_cluster_l1 = KMeans(n_clusters=5, random_state=0).fit(df.values)
#
#clusters_l1=[]
#for i in range (0,2):
#    dict_cluster={}
#    dict_cluster['class']=i
#    dict_cluster['clusters']=recup_labels(df_cluster_l1, i)
#    clusters_l1.append(dict_cluster)
#    
##Enregistrement de la matrice en csv
#string="mnist_512_512/mnist_l1_512_disc.csv"
#df.to_csv(string)

#df,nb_bins_l2 = discretise_dataset_predict_true('mnist_512_512/mnist_l2_512.csv',512)

#df_cluster_l2 = KMeans(n_clusters=5, random_state=0).fit(df.values)
#
#clusters_l2=[]
#for i in range (0,2):
#    dict_cluster={}
#    dict_cluster['class']=i
#    dict_cluster['clusters']=recup_labels(df_cluster_l2, i)
#    clusters_l2.append(dict_cluster)
#
##Enregistrement de la matrice en csv
#string="mnist_512_512/mnist_l2_512_disc.csv"
#df.to_csv(string)

#%%


def predict_real_value(value,X_train_digits,y_train_labels):
    #print ("computing ",value)
    X_value=[]
    for i in range(X_train_digits.shape[0]):
        if (y_train_labels[i]==value):
            X_value.append(X_train_digits[i])
    X_value=np.array(X_value)
    train_digitsvalue=X_value
    
    # re-shape the images data
    train_datavalue = np.reshape(train_digitsvalue,(train_digitsvalue.shape[0],784))

    # re-scale the image data to values between (0.0,1.0]
    train_datavalue = train_datavalue.astype('float32') / 255.  
    
    #print(train_datavalue[0]) 
    Y_pred_classes = model.predict_classes(train_datavalue)
    return np.array(Y_pred_classes)[:,0], train_datavalue

# load the data 
(X_train_digits, y_train_labels), (X_test_digits, y_test_labels) = mnist.load_data()
values=[2, 3, 4, 5, 6, 7, 8, 9]

real_values_predict=[]
for i in range (len(values)):
    dict_predict={}
    dict_predict['value']=values[i]
    dict_predict['prediction'], train_datavalue = predict_real_value(values[i],X_train_digits,y_train_labels)
    real_values_predict.append(dict_predict)
    result_layers_datavalue = get_result_layers(model, train_datavalue)
    filename = project_name + "_predict_" + str(values[i])
    save_result_layers(filename,train_datavalue,dict_predict['prediction'],result_layers_datavalue)
    
    print(i)






















