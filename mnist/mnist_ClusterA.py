import numpy as np
import pandas as pd 
import keras
import os
import re
from sklearn.cluster import KMeans
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn import preprocessing, model_selection
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn import datasets
from tensorflow.keras.datasets import mnist
from Levenshtein import distance

def get_goodXy (X,y):
    ynew = model.predict_classes(X)
    X_good =[]
    y_good=[]
    for i in range(len(X)):
        if (ynew[i]==0 and y[i]==1) or (ynew[i]==1 and y[i]==0):
            print ("error prediction for X=%s, Predicted=%s, Real=%s"% (X[i], ynew[i], y[i]))
        else :
            X_good.append(X[i])
            y_good.append(y[i])
    return X_good,y_good        

def get_result_layers(model,X):
    result_layers=[]
    for i in range (len(model.layers)-1):
        hidden_layers= keras.backend.function(
                [model.layers[0].input],   
                [model.layers[i].output,] 
                )    
        result_layers.append(hidden_layers([X_good])[0])  
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
    

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train_sample=X_train[0:100]
y_train_sample=y_train[0:100]

X_train=X_train_sample
y_train=y_train_sample
X_train = X_train.reshape(100, 784)
X_train = X_train.astype('float32')

X_train /= 255



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


1
input_dim = 784

model = Sequential()
model.add(Dense(5, input_dim = input_dim , activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )

model.fit(train_X, train_y, epochs = 40, batch_size = 32)

X_good,y_good=get_goodXy (train_X, train_y)

# Récupération des valeurs de tous les layers sauf le dernier
result_layers=get_result_layers(model,X_good)

# Sauvegarde du fichier
# structure :
# 0/1 = valeur de la classe
# chaque valeur de layer est entourée par une étoile *
save_result_layers("mnist_512_tmp",X_good,y_good,result_layers)
# tri du fichier
os.system ('sort mnist_512_tmp > mnist_512_.csv')
# effacer le fichier intermédiaire
#os.system ('rm mnist_512_tmp')

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
    token_layer=[]
    token_exemples=[]
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

        
filename='mnist_512_tmp'
get_directory_layers_from_csv(filename)


def discretise_dataset(filename,bins):
    df = pd.read_csv(filename, sep = ',', header = None) 
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
    
# exemple d'utilisation  
df=discretise_dataset('mnist_512/mnist_l1_512.csv',5)
#print (df.head())
#print (df.tail())

#exemple d'utilisation de la mesure de levenshtein
#print (distance('3,10,0,4,0,0,9,0', '4,8,0,2,0,0,8,0'))

#Creation d'une matrice de distance
#df_mat=pd.DataFrame(X)

df.to_csv("mnist_512/mnist_l1_512_disc.csv")

string="mnist_512/mnist_l1_512_disc.csv"

#df_mat.to_csv(string, sep=',', encoding='utf-8',index=False, header=False)
df_mat=pd.read_csv(string, sep = ',',header = None)

#recuperation de la matrice sous la forme de chaine
ch = df_mat.to_string(header=False,index=False,index_names=False).split('\n')

vals = []

for v in df_mat.values[1:]:
    string = ""
    for s in v[1:]:
        string = string + str(s) + ","
    new_string = string[:-1]
    vals.append(new_string)

#print(vals)

List1 = vals
List2 = vals

Matrix = np.zeros((len(List1),len(List2)),dtype=np.int)

for i in range(0,len(List1)):
    for j in range(0,len(List2)):
        Matrix[i,j] = distance(List1[i],List2[j])

#print (Matrix)

import numpy as np
import sklearn.cluster

words = vals
words = np.asarray(words) #So that indexing with a list will work

lev_similarity = -1*np.array([[distance(w1,w2)for w1 in words] for w2 in words])

affprop = sklearn.cluster.AffinityPropagation(affinity="precomputed", damping=0.5)
affprop.fit(lev_similarity)

#Affichage
for cluster_id in np.unique(affprop.labels_):
    exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
    cluster = np.unique(words[np.nonzero(affprop.labels_==cluster_id)])
    cluster_str = ", ".join(cluster)
    print(" - *%s:* %s" % (exemplar, cluster_str))


    

