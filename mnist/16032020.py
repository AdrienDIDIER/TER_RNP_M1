import numpy as np
import pandas as pd 
import keras
import os
import re
import matplotlib.pyplot as plt

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
                if (ynew[i]==0 and y[i]==1) or (ynew[i]==1 and y[i]==0):
                    print ("error prediction for X=%s, Predicted=%s, Real=%s"% (X[i], ynew[i], y[i]))
                else :
                    X_good.append(X[i])
                    y_good.append(y[i])
            else:
                if (ynew[i]==5 and y[i]!=5):
                    print ("error prediction for X=%s, Predicted=%s, Real=%s"% (X[i], ynew[i], y[i]))
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

def save_result_layers(filename,X,y,result_layers):
    f = open(filename, "w")
    for nb_X in range (len(X)):
        #my_string=""
        my_string=str(y[nb_X])+','
        for nb_layers in range (len(model.layers)-1):
            my_string+="<b>,"
            for j in range (len(result_layers[nb_layers][nb_X])):
                my_string+=str(result_layers[nb_layers][nb_X][j])+','
            my_string+="</b>,"    
        my_string=my_string [0:-1]
        my_string+='\n'
        f.write(my_string)    
    f.close()
    
    
#Fonction permettant de créer un csv pour chaque couches dans un répertoire   
def get_directory_layers_from_csv(filename):
    tokens=filename.split("_")
    df = pd.read_csv(filename, sep = ',', header = None) 
    
    # creation d'un répertoire pour sauver tous les fichiers
    repertoire=filename[0:-4]
    os.makedirs(repertoire, exist_ok=True)
    string = repertoire+'/'+tokens[0]+'_'
    f=[]
    filenames=[]
    for nb_tokens in range (1,len(tokens)-1):
        name_file=string+'l'+str(nb_tokens)+'_'+tokens[nb_tokens]+'.csv'
        f.append(open(name_file, "w"))
        filenames.append(name_file)
        
        
    # sauvegarde du dataframe dans une chaîne de caracteres
    ch = df.to_string(header=False,
                  index=False,
                  index_names=False).split('\n')
    vals = [','.join(ele.split()) for ele in ch]
    
    # sauvegarde dans des fichiers spécifiques par layer
    for nb_exemples in range (len(vals)):
        deb=str(df[0][nb_exemples])+','
        # 1 ligne correspond à une chaine
        s=vals[nb_exemples]
        listoftokens=re.findall(r'<b>,(.+?),</b>', s)
        nb_layers=len(listoftokens)
        
        for nb_token in range (nb_layers):
            save_token=''
            save_token=deb+str(listoftokens[nb_token])+'\n'
            
            f[nb_token].write(save_token)


#Fonction permettant de discrétiser nos valeurs présentes dans le fichier csv
def discretise_dataset(filename,bins):
    df = pd.read_csv(filename, sep = ',', header = None) 
    print(df.values)
    oneColumn = np.array(df[1])
    for i in range(2,df.shape[1]):
        oneColumn=np.append(oneColumn,np.array(df[i]))
    dfoneColumn=pd.DataFrame(oneColumn)
    nb_bins=bins
    dftemp=pd.DataFrame()
    dftemp[0]=pd.cut(dfoneColumn[0], bins=nb_bins, labels=np.arange(nb_bins), right=False)
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
    dftemp[0], bins =pd.cut(dfoneColumn[0], bins=nb_bins, retbins=True, labels=np.arange(nb_bins), right=False)
    df_new=pd.DataFrame(df[0])
    nb_tuples=df.shape[0]
    j=0
    for i in range(1,df.shape[1]):
        df_new[i]=np.copy(dftemp[0][j:j+nb_tuples])
        j+=nb_tuples
    return df_new, bins


print("\n\n\n========== Apprentissage ==========")

#Utilisation du jeu de données mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train_sample=X_train[0:784]
y_train_sample=y_train[0:784]

X_train=X_train_sample
y_train=y_train_sample
X_train = X_train.reshape(784, 784)
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
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )

model.fit(train_X, train_y, epochs = 40, batch_size = 100)

X_good,y_good=get_goodXy (train_X, train_y,0)



# Récupération des valeurs de tous les layers sauf le dernier
result_layers=get_result_layers(model,X_good)



# Sauvegarde du fichier
# Structure :
# 0/1 = valeur de la classe
save_result_layers("mnist_512_tmp",X_good,y_good,result_layers)
# tri du fichier puis conversion en csv
os.system ('sort mnist_512_tmp > mnist_512_.csv')
# effacer le fichier intermédiaire
#os.system ('rm mnist_512_tmp')



#Création du(des) csv à partir du fichier tmp 
filename='mnist_512_tmp'
get_directory_layers_from_csv(filename)

  
#Création de la matrice avec les données discrétisées
#df, df_bins = discretise_dataset_predict_true('mnist_512/mnist_l1_512.csv',512)
df = discretise_dataset('mnist_512/mnist_l1_512.csv',512)

print(df.values)
df_cluster = KMeans(n_clusters=2, random_state=0).fit(df.values)
print(df_cluster.labels_)


#Enregistrement de la matrice en csv
string="mnist_512/mnist_l1_512_disc.csv"
df.to_csv(string)

# =============================================================================
# 
# #Recuperation de la matrice sous la forme de chaine
# df_mat=pd.read_csv(string, sep = ',',header = None)
# 
# ch = df_mat.to_string(header=False,index=False,index_names=False).split('\n')
# 
# vals = []
# X_kmeans = []
# 
# for v in df_mat.values[1:]:
#     string = ""
#     for s in v[1:]:
#         string = string + str(s) + ","
#     new_string = string[:-1]
#     vals.append(new_string)
#     
# List1 = vals
# List2 = vals
# 
# #Création de la matrice de distance à l'aide de la distance de Levenshtein
# Matrix = np.zeros((len(List1),len(List2)),dtype=np.int)
# 
# for i in range(0,len(List1)):
#     for j in range(0,len(List2)):
#         Matrix[i,j] = distance(List1[i],List2[j])
# 
# 
# 
# #Génération des clusters
# 
# 
# #X = StandardScaler().fit_transform(Matrix)
# print("\n\n\n========== MATRIX ==========")
# print(Matrix)
# db = DBSCAN(eps=250, min_samples=5).fit(Matrix)
# 
# =============================================================================



print("\n\n\n========== KMEANS ==========")
#
#X_kmeans = df_mat.values[1:]
#    
#df_kmeans = KMeans(n_clusters=5).fit(X_kmeans)
#print("Label : ",df_kmeans.labels_)

#Utilisation du jeu de données mnist
(X_train_5, y_train_5), (X_test_5, y_test_5) = mnist.load_data()
X_train_sample_5=X_train_5[0:100]
y_train_sample_5=y_train_5[0:100]

X_train_5=X_train_sample_5
y_train_5=y_train_sample_5
X_train_5 = X_train_5.reshape(100, 784)
X_train_5 = X_train_5.astype('float32')

X_train_5 /= 255

#On récupère les objets appartenants aux classes 5
X_5=[]
y_5=[]
nb_X5=0

for i in range(X_train_5.shape[0]):
    if (y_train_5[i]==1):
        
        nb_X5+=1
        X_5.append(X_train_5[i])
        y_5.append(y_train_5[i])

       
train_X_5=np.asarray(X_5)

train_y_5=y_5

encoder_5 = LabelEncoder()
train_y_5=encoder_5.fit_transform(train_y_5)



#Création de notre reseau de neurones
#La première couche possède 512 neurones, correspondant aux 512 pixels d'une image
input_dim = 784

model_5 = Sequential()
model_5.add(Dense(512, input_dim = input_dim , activation = 'relu'))
model_5.add(Dense(1, activation = 'sigmoid'))

model_5.compile(loss = 'binary_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )

model_5.fit(train_X_5, train_y_5, epochs = 40, batch_size = 100)

X_good_5,y_good_5=get_goodXy (train_X_5, train_y_5,1)



# Récupération des valeurs de tous les layers sauf le dernier
result_layers_5=get_result_layers(model_5,X_good_5)



# Sauvegarde du fichier
# Structure :
# 0/1 = valeur de la classe
save_result_layers("mnist_512_tmp_5",X_good_5,y_good_5,result_layers_5)
# tri du fichier puis conversion en csv
os.system ('sort mnist_512_tmp_5 > mnist_512_5.csv')
# effacer le fichier intermédiaire
#os.system ('rm mnist_512_tmp')

#Création du(des) csv à partir du fichier tmp 
filename5='mnist_512_tmp_5'
get_directory_layers_from_csv(filename5)


#df_5_compared = discretise_dataset("mnist_512_t/mnist_l1_512.csv", df_bins)
#
#print("df5 comparé :" , df_5_compared)
#

df_5 = discretise_dataset("mnist_512_t/mnist_l1_512.csv", 512)

#df = KMeans(n_clusters=1).fit(df_5.values)
#print(df.labels_)

print("df: ", df.values)
print("df_5 :", df_5.values)
#print(df_5.values)
print("clust: ", df_cluster)
print("labels: ", df_cluster.labels_)

y_result = df_cluster.predict(df_5.values)
print("Result : ", y_result)

# =============================================================================
# print("\n\n\n========== DBSCAN ==========")
# print(db)
# core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# core_samples_mask[db.core_sample_indices_] = True
# labels = db.labels_
# 
# # Number of clusters in labels, ignoring noise if present.
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# n_noise_ = list(labels).count(-1)
# 
# 
# print('Estimated number of clusters: %d' % n_clusters_)
# print('Estimated number of noise points: %d' % n_noise_)
# 
# 
# print("\n\n\n========== DBSCAN Clustering ==========")
# unique_labels = set(labels)
# colors = [plt.cm.Spectral(each)
#           for each in np.linspace(0, 1, len(unique_labels))]
# for k, col in zip(unique_labels, colors):
#     if k == -1:
#         # Black used for noise.
#         col = [0, 0, 0, 1]
# 
#     class_member_mask = (labels == k)
# 
#     xy = Matrix[class_member_mask & core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#              markeredgecolor='k', markersize=14)
# 
#     xy = Matrix[class_member_mask & ~core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#              markeredgecolor='k', markersize=6)
# 
# plt.title('Estimated number of clusters: %d' % n_clusters_)
# plt.show()
# 
# 
# print("\n\n\n========== Dendrogramme ==========")
# #Création du dendrogramme
# Z = linkage(Matrix, 'ward')
# fig = plt.figure(figsize=(25, 10))
# dn = dendrogram(Z)
# plt.show()
# 
# =============================================================================

