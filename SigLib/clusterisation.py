import collections as c
from sklearn.cluster import KMeans
import json

import numpy as np

import os

def recup_labels(df_cluster, val, output_classes):
    cluster = []
    for i in range(0, len(output_classes)):
        if output_classes[i] == val:
            cluster.append(df_cluster.labels_[i])    
    return cluster 


#rename clusterize_signatures(signatures, training_class_values)
def clusterize_training_signatures(training_signatures, train_values):
    clusterized_training_signatures = []
    layers_kmeans = []
    for df in training_signatures:
          output_classes = df.copy().iloc[:, 0].to_numpy()
          df_ = df.drop(df.columns[0], axis=1)
          kmeans = KMeans(n_clusters=5, random_state=0).fit(df_.values) 
          layers_kmeans.append(kmeans)      
          clusters = []
          for i in range (0,len(train_values)):
              dict_cluster={}
              dict_cluster['class']=train_values[i]
              dict_cluster['clusters']=recup_labels(kmeans, train_values[i], output_classes)
              clusters.append(dict_cluster)
          clusterized_training_signatures.append(clusters)
    return clusterized_training_signatures, layers_kmeans

def getClustersData(clusters):
    csv_data = []
    for i in range (len(clusters)):
        cpt = c.Counter(clusters[i]['clusters'])
        for m in cpt:
            csv_data.append([clusters[i]['class'], cpt[m], m])  
    return csv_data

#save_clusters_data(clusterized_signatures)
def save_clusters_data(dir_, clusterized_discretized_result_layers):
    with open(dir_ + 'clusters_info.json','w') as json_file:
        clusters_info = []
        layer_number = 1
        for df in clusterized_discretized_result_layers:
            clusters_layer_data = getClustersData(df)
            clusters_info.append([])
            for d in clusters_layer_data:
                cluster_dict = {'layer': layer_number, 'class': int(d[0]), 'instances_count': int(d[1]), 'cluster_index': int(d[2])}
                clusters_info[layer_number - 1].append(cluster_dict)
            layer_number += 1
        json.dump(clusters_info, json_file)
        print('saved clusters_info.json')
        
#rename predict_cluster(signatures, layers_kmeans, class_values)
def clusterize_test_signatures(test_signatures, layers_kmeans, values, directory_name): 
    if not(os.path.isdir(directory_name + '_DATA')):
        os.mkdir(directory_name + '_DATA')
    f = open(directory_name + '_DATA/clusterized_values.csv', 'w')
    header = 'input_class,'
    for ll in range(0, len(layers_kmeans)):
        header += 'cluster_in_layer' + str(ll+1) + ','
    header += 'output_class' + '\n'
    f.write(header)
    for j in range(0, len(values)): 
        y_results =[]
        for i in range(0, len(layers_kmeans)):
            index = j * len(layers_kmeans) + i
            output_classes = test_signatures[index].copy().iloc[:, 0].to_numpy()
            df = test_signatures[index].copy()
            df_ = df.drop(df.columns[0], axis=1)
            if np.any(np.isnan(df_)):
                mat = np.isnan(df_)
                for k in range(0, len(mat.values)):
                    for l in range(0, len(mat.values[i])):
                        if mat.values[k][l] == True:
                            df_.iat[k, l] = 0
            y_results.append(layers_kmeans[i].predict(df_.values))
        for k in range(len(y_results[0])):
            clean = str(values[j]) + ','
            for l in range(len(y_results)):
                clean += str(y_results[l][k]) + ','
            output_class = output_classes[k]
            clean += str(output_class) + '\n'
            f.write(clean)
    print('saved clusterized_values.csv in ' + directory_name + '_DATA/')
    f.close()
    
    
    

            