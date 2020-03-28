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
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )

model.fit(train_X, train_y, epochs = 40, batch_size = 100)


X_good,y_good=get_goodXy (train_X, train_y,0)

# Récupération des valeurs de tous les layers sauf le dernier
result_layers=get_result_layers(model,X_good)



# Sauvegarde du fichier
# Structure :
# 0/1 = valeur de la classe
# =============================================================================
# print("format 1")
# print(X_good)
# print(y_good)
# print(result_layers)
# =============================================================================
save_result_layers("mnist_512_tmp",X_good,y_good,result_layers)
# tri du fichier puis conversion en csv
os.system ('mnist_512_tmp > mnist_512_.csv')
# effacer le fichier intermédiaire
#os.system ('rm mnist_512_tmp')



#Création du(des) csv à partir du fichier tmp 
filename='mnist_512_tmp'
get_directory_layers_from_csv(filename)

  
#Création de la matrice avec les données discrétisées
#df, df_bins = discretise_dataset_predict_true('mnist_512/mnist_l1_512.csv',512)
df,nb_bins = discretise_dataset_predict_true('mnist_512/mnist_l1_512.csv',512)

#%%
print("\nClusters :")

#import sys
#import numpy
#numpy.set_printoptions(threshold=sys.maxsize)

df_cluster = KMeans(n_clusters=5, random_state=0).fit(df.values)

#for i in range(500):
#    print("DF values : ",df.values[:,0][i], "Labels : ",df_cluster.labels_[i])



def recup_labels(val):
    cluster = []
    for i in range(len(df.values[:,0])):
        if df.values[:,0][i] == val:
            cluster.append(df_cluster.labels_[i])
    
    return cluster 
    
clusters=[]
for i in range (0,2):
    dict_cluster={}
    dict_cluster['class']=i
    dict_cluster['clusters']=recup_labels(i)
    clusters.append(dict_cluster)
    
#print(clusters[0])

def affichage_Clusters():
    for i in range (len(clusters)):
        print ("\nClusters classe #",clusters[i]['class'])
        cpt = c.Counter(clusters[i]['clusters'])
        print ('\n'.join(['({}) objets pour cluster {}'.format(cpt[m],m) for m in cpt]))   

print("Premier affichage clusters :")
affichage_Clusters()
#%%
    
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

#%%

print("\n\n\n========== Clusters ==========\n")
#
#X_kmeans = df_mat.values[1:]
#    
#df_kmeans = KMeans(n_clusters=5).fit(X_kmeans)
#print("Label : ",df_kmeans.labels_)

#Utilisation du jeu de données mnist
#(X_train_5, y_train_5), (X_test_5, y_test_5) = mnist.load_data()
X_train_5 = X_train_base
y_train_5 = y_train_base

X_test_5 = X_test_base
y_test_5 = y_test_base

X_train_sample_5=X_train_5[0:7840]
y_train_sample_5=y_train_5[0:7840]

X_train_5=X_train_sample_5
y_train_5=y_train_sample_5
X_train_5 = X_train_5.reshape(7840, 784)
X_train_5 = X_train_5.astype('float32')

X_train_5 /= 255

#On récupère les objets appartenants aux classes 5
X_5=[]
y_5=[]
nb_X5=0

classe = int(input("Entrez la classe à train : "))

for i in range(X_train_5.shape[0]):
    if (y_train_5[i]==classe):
        
        nb_X5+=1
        X_5.append(X_train_5[i])
        y_5.append(y_train_5[i])

       
train_X_5=np.asarray(X_5)

train_y_5=y_5

encoder_5 = LabelEncoder()
train_y_5=encoder_5.fit_transform(train_y_5)

predicted_y5 = model.predict_classes(train_X_5)
predicted_y5_noarray = []
for i in range(0, len(predicted_y5)):
    predicted_y5_noarray.append(predicted_y5[i][0])

result_layers_5 = get_result_layers(model, train_X_5)


# =============================================================================

save_result_layers("mnist_512_tmp_5",train_X_5,predicted_y5_noarray,result_layers_5)
os.system ('mnist_512_tmp_5 > mnist_512_5.csv')

#Création du(des) csv à partir du fichier tmp 
filename5='mnist_512_tmp_5'
get_directory_layers_from_csv(filename5)


#df_5_compared = discretise_dataset("mnist_512_t/mnist_l1_512.csv", df_bins)
#
#print("df5 comparé :" , df_5_compared)

df_5 = discretise_dataset("mnist_512_t/mnist_l1_512.csv", nb_bins)
string="mnist_512_t/mnist_l1_512_disc.csv"
df_5.to_csv(string)
#df = KMeans(n_clusters=1).fit(df_5.values)
#print(df.labels_)
#print("\ndf : ", df)
#print("\ndf.values : ", df.values)
#print("\ndf_5 :", df_5)
#print("\ndf_5.values :", df_5.values)
#print(df_5.values)

#print("\nclust: ", df_cluster)
#print("\nlabels: ", df_cluster.labels_)

y_result = df_cluster.predict(df_5.values)
#print("Result : ", y_result)

dict_cluster_5={}
dict_cluster_5['class']= classe
dict_cluster_5['clusters']=y_result
clusters.append(dict_cluster_5)

def affichage_Clusters_ajout():
    affich = ""
    cpt = c.Counter(clusters[2]['clusters']) 
   #/print(cpt)
    for i in range(0,2):
        cptclasse = 0
        cpt_temp = list(c.Counter(clusters[i]['clusters']))
        cpt_c = []
        
        for y in range(len(cpt_temp)):
            cptclasse += cpt[cpt_temp[y]]
            cpt_c.append(str(cpt_temp[y])+":"+str(cpt[cpt_temp[y]]))
        
        #print(cpt_c)
        
        affich += str(cptclasse) + " signatures de #"+str(clusters[i]['class'])+"\n"
        for y in range(len(cpt_c)):          
            affich += "   "+ str(cpt_c[y].split(":")[1])+" objets pour cluster " + str(cpt_c[y].split(":")[0]) +"\n"
       
    print(affich)
        
    
#%%

# ============================================================================
print("\n\n\n========== Vérité terrain ==========\n")


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
    return np.array(Y_pred_classes)

# load the data 
(X_train_digits, y_train_labels), (X_test_digits, y_test_labels) = mnist.load_data()
values=[2, 3, 4, 5, 6, 7, 8, 9]

real_values_predict=[]
for i in range (len(values)):
    dict_predict={}
    dict_predict['value']=values[i]
    dict_predict['prediction']=predict_real_value(values[i],X_train_digits,y_train_labels)[:,0]
    real_values_predict.append(dict_predict)

#real_values_predict contient les valeurs obtenues en application du modèle initial
#il s'agit de la vérité terrain -> le classifieur en sortie donne une classe
#ces données vont servir à vérifier que lorsqu'on recherche les signatures dans les layers
# on a bien ce qui est bon
for i in range (len(real_values_predict)):
    print ("\nPrediction pour classe #",real_values_predict[i]['value'])
    cpt = c.Counter(real_values_predict[i]['prediction'])
    print ('\n'.join(['({}) objets pour classe {}'.format(cpt[m],m) for m in cpt]))

# ============================================================================
print("\n\n\n========== Predict Clusters classe "+str(classe)+" ==========\n")

affichage_Clusters_ajout() 

#%%
    
# ============================================================================
print("\n\n\n========== Affichage Clusters ==========")

def plot_clusters_2D (X,y_predict, nb_clusters, pca_done=False):
    if pca_done==False:
        pca = PCA(n_components=2) 
        X_r = pca.fit(X).transform(X)
    else: X_r = X    
    data = pd.DataFrame(X_r, columns=['x','y'])
    data['label']=y_predict                             
    list_clusters=list(set(y_predict))
    #create a new figure
    plt.figure(figsize=(5,5))

    #loop through labels and plot each cluster
    for i, label in enumerate(list_clusters):

        #add data points 
        plt.scatter(x=data.loc[data['label']==label, 'x'], 
                y=data.loc[data['label']==label,'y'], 
                alpha=0.20)
    
        #add label
        plt.annotate(label, 
                 data.loc[data['label']==label,['x','y']].mean(),
                 horizontalalignment='center',
                 verticalalignment='center',
                 size=20, weight='bold') 



plot_clusters_2D(df.values,df_cluster.labels_,5)


#%%























