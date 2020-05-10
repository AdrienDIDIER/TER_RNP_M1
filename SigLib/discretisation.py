import numpy as np
import pandas as pd 
import re
import os

#on lui donne tout le temps un df
#bins = nombre de discretisation
#boolean retbins si on veut retourner nb_bins
    
def discretise_dataset(df, bins, retbinsBool = False):
    oneColumn = np.array(df[1])
    for i in range(2,df.shape[1]):
        oneColumn=np.append(oneColumn,np.array(df[i]))
    dfoneColumn=pd.DataFrame(oneColumn)
    nb_bins=bins
    dftemp=pd.DataFrame()
    if(retbinsBool):
        dftemp[0], return_bins =pd.cut(dfoneColumn[0], nb_bins, right=False, labels=np.arange(nb_bins), retbins=True)
    else:
        dftemp[0]=pd.cut(dfoneColumn[0], nb_bins, right=False, labels=np.arange(len(nb_bins) - 1))
    df_new=pd.DataFrame(df[0])
    nb_tuples=df.shape[0]
    j=0
    for i in range(1,df.shape[1]):
        df_new[i]=np.copy(dftemp[0][j:j+nb_tuples])
        j+=nb_tuples
    if(retbinsBool):
        return df_new, return_bins
    else:
        return df_new


#soit on lui donne le project_name et il va chercher les fichiers dans le dossier du projet 
#ou alors on lui donne une liste de df (un df pour chaque layer) et il sauvegarde dans un output_file_name
#ou alors on lui donne une liste de filepath et il enregistre à côté

#rename discretize_training_signatures(bins, )
def discretize_training_signatures(bins, project_name = None, filepaths = None):
    discretized_result_layers = []
    nbs_bins = []
    if isinstance(project_name, str):
        regex = re.compile('('+project_name+'_result_l)([0-9]*)(.csv)')
        for root, dirs, files in os.walk(project_name):
            for filename in files:
                if regex.match(filename):
                    df_dataset = pd.read_csv(project_name + '/' + filename, sep = ',', header = None)
                    df, nb_bins = discretise_dataset(df_dataset, bins, retbinsBool = True)
                    discretized_result_layers.append(df)
                    nbs_bins.append(nb_bins)
    elif isinstance(filepaths, list):
        for filepath in filepaths:
            df_dataset = pd.read_csv(filepath, sep = ',', header = None)
            df, nb_bins = discretise_dataset(df_dataset, bins, retbinsBool = True)
            discretized_result_layers.append(df)
            nbs_bins.append(nb_bins)
    return discretized_result_layers, nbs_bins

#rename discretize_test_signatures(bins, )
def discretize_test_signatures(project_name, test_values, nbs_bins):
    discretized_values_predictions = []
    for i in test_values:
        directory_name = project_name + '_predict_' + str(i) + '/'
        for j in range(0, len(nbs_bins)):
            file_name = directory_name + project_name + '_predict_' + str(i) +  "_result_l" + str(j+1) + ".csv"
            file = pd.read_csv(file_name, sep = ',', header = None)
            df = discretise_dataset(file, nbs_bins[j])
            df.to_csv(directory_name + project_name + '_predict_' + str(i) +  "_result_l" + str(j+1) + "_disc.csv")
            discretized_values_predictions.append(df)
            print('saved ' + directory_name + project_name + '_predict_' + str(i) +  "_result_l" + str(j+1) + "_disc.csv")
    return discretized_values_predictions
    
def get_saved_discretized_test_signatures(model, project_name, test_values):
    discretized_values_predictions = []
    for i in test_values:
        for j in range(0, len(model.layers) - 1):
            directory_name = project_name + '_predict_' + str(i) + '/'
            df = pd.read_csv(directory_name + project_name + '_predict_' + str(i) +  "_result_l" + str(j+1) + "_disc.csv", index_col=0)
            discretized_values_predictions.append(df.copy())
    return discretized_values_predictions
