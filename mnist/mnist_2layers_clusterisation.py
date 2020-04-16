import numpy as np
import pandas as pd 
import keras
import os
import re
import matplotlib.pyplot as plt
import collections as c
import csv

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

def discretise_dataset(filename,bins, LIMIT):
    df = pd.read_csv(filename, sep = ',', header = None)[:LIMIT]
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

project_name = 'neural_test'

def discretize_result_layers(project_name, nb_layers):
    discretized_result_layers = []
    nbs_bins = []
    for i in range(0, nb_layers):
        df, nb_bins = discretise_dataset_predict_true(project_name+'/'+project_name+'_result_l'+ str(i+1) +'.csv',512)
        discretized_result_layers.append(df)
        nbs_bins.append(nb_bins)
    return discretized_result_layers, nbs_bins


def recup_labels(df_cluster, val, df):
    cluster = []
    for i in range(len(df.values[:,0])):
        if df.values[:,0][i] == val:
            cluster.append(df_cluster.labels_[i])
    
    return cluster 

def clusterize_discretized_result_layers(discretized_result_layers):
    clusterized_discretized_result_layers = []
    for df in discretized_result_layers:
          df_cluster = KMeans(n_clusters=5, random_state=0).fit(df.values) 
          clusters = []
          for i in range (0,2):
              dict_cluster={}
              dict_cluster['class']=i
              dict_cluster['clusters']=recup_labels(df_cluster, i, df)
              clusters.append(dict_cluster)
          clusterized_discretized_result_layers.append(clusters)
    return clusterized_discretized_result_layers


discretized_result_layers, nbs_bins = discretize_result_layers(project_name, 3)

clusterized_discretized_result_layers = clusterize_discretized_result_layers(discretized_result_layers)



def getClustersData(clusters):
    csv_data = []
    for i in range (len(clusters)):
        print ("\nClusters classe #",clusters[i]['class'])
        cpt = c.Counter(clusters[i]['clusters'])
        for m in cpt:
            csv_data.append([clusters[i]['class'], cpt[m], m])
        print ('\n'.join(['({}) objets pour cluster {}'.format(cpt[m],m) for m in cpt]))  
    return csv_data


def save_clusters_data(clusterized_discretized_result_layers):
    f = open('clusters_data.csv', 'w')
    header = 'layer,class,count,cluster' + '\n'
    f.write(header)
    for df in clusterized_discretized_result_layers:
        csv_data = getClustersData(df)
        for d in csv_data:
            clean =  str(1) + ',' + str(d[0]) + ',' + str(d[1]) + ',' + str(d[2]) + '\n'
            f.write(clean)
    f.close()
    
    
save_clusters_data(clusterized_discretized_result_layers)

values=[2, 3, 4, 5, 6, 7, 8, 9]

def discretize_values_predictions(values, nbs_bins):
    discretized_values_predictions = []
    for i in values:
        directory_name = project_name + '_predict_' + str(i) + '/'
        for j in range(0, len(nbs_bins)):
            file_name = directory_name + project_name + '_predict_' + str(i) +  "_result_l" + str(j+1) + ".csv"
#            file = pd.read_csv(file_name, sep = ',', header = None)
            df = discretise_dataset(file_name, nbs_bins[j])
            df.to_csv(directory_name + project_name + '_predict_' + str(i) +  "_result_l" + str(j+1) + "_disc.csv")
            discretized_values_predictions.append(df)
            print('saved ' + directory_name + project_name + '_predict_' + str(i) +  "_result_l" + str(j+1) + "_disc.csv")
    return discretized_values_predictions
    
    
#discretized_values_predictions = discretize_values_predictions(values, nbs_bins)

discretized_values_predictions = []
for i in values:
    for j in range(0, 3):
        directory_name = project_name + '_predict_' + str(i) + '/'
        df = pd.read_csv(directory_name + project_name + '_predict_' + str(i) +  "_result_l" + str(j+1) + "_disc.csv")
        discretized_values_predictions.append(df)

def clusterize_discretized_values_predictions(discretized_values_predictions, clusterized_discretized_result_layers):   
    f = open('clusterized_values.csv', 'w')
    header = 'input_class,'
    for i in range(0, len(clusterized_discretized_result_layers)):
        header += 'cluster_in_layer' + str(i+1) + ','
    header += 'output_class' + '\n'
    f.write(header)
    for j in range(0, len(values)): 
        y_results =[]
        for i in range(0, len(clusterized_discretized_result_layers)):     
            y_results.append(clusterized_discretized_result_layers[j].predict(discretized_values_predictions[j * len(values) + i].values))
        output_class = -1
        clean = str(values[j]) + ','    
        for k in range(len(y_results)):
            for l in range(len(y_results[k])):
                clean += str(y_results[k][l]) + ','
                output_class = discretized_values_predictions.iloc[l, 0]
        clean += str(output_class)
        f.write(clean)
        print(values[j])
    f.close()
            

clusterize_discretized_values_predictions(discretized_values_predictions, clusterized_discretized_result_layers)




#f = open('clusterized_values.csv', 'w')
#header = 'input_class,layer1,layer2,output' + '\n'
#f.write(header)
#for i in values:    
#    directory_name = project_name + '_predict_' + str(i) + '/'
#    
#    l1_file = directory_name + project_name + '_predict_' + str(i) +  "_result.csv"
#    l2_file = directory_name + "mnist_l2_512X.csv"
#    
#    df_l2_file = pd.read_csv(l2_file, sep = ',', header = None)[:100]
#    
#    df_disc_l1 = discretise_dataset(l1_file, nb_bins_l1, 100)
#    y_result_l1 = df_cluster_l1.predict(df_disc_l1.values)
#    y_results_l1.append(y_result_l1)
#        
#    df_disc_l2 = discretise_dataset(l2_file, nb_bins_l2, 100)
#    y_result_l2 = df_cluster_l2.predict(df_disc_l2.values)
#    y_results_2.append(y_result_l2)
#    
#    for j in range(len(y_result_l1)):
#        clean = str(i) + ',' + str(y_result_l1[j]) + ',' + str(y_result_l2[j]) + ',' + str(df_l2_file.iloc[j,0]) + '\n'
#        f.write(clean)
#    
#    print(i)
#
#f.close()
#    


