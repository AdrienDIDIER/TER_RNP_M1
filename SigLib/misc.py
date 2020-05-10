import json
import clusterisation


#Fonction pour récupérer les objets dans X_train appartenants aux classes données dans train_values
#2 types possible pour l'argument limit :
#   - un int pour donner le nombre d'objets total à récupérer
#   - une liste de int de même taille que train_values pour récupérer le nombre d'objets total à récupérer pour chaque classe

def get_dataset_values(values, X, y, limit = None):
    X_values=[]
    y_values=[]
    if isinstance(limit, int):    
        nb_X = 0
        for i in range(X.shape[0]):
            if nb_X >= limit:
                break
            for index in range(0, len(values)):
                if y[i]==values[index]:
                    X_values.append(X[i])
                    y_values.append(y[i])
                    nb_X+=1
    elif isinstance(limit, list):
        if len(limit) == len(values):
            limit_count = [0] * len(values)
            for i in range(X.shape[0]):
                for index in range(0, len(values)):
                    if limit_count[index] < limit[index]:
                        if y[i]==values[index]:
                            X_values.append(X[i])
                            y_values.append(y[i])
                            limit_count[index]+=1 
    else:
        for i in range(X.shape[0]):
            for index in range(0, len(values)):
                if y[i]==values[index]:
                    X_values.append(X[i])
                    y_values.append(y[i])
                    
    return X_values, y_values

def save_network_info(dir_, model):
    layers_info = []
    for layer in model.layers:
        layers_info.append(layer.get_config())
    with open(dir_ + 'network_info.json','w') as json_file:
        json.dump(layers_info, json_file)

def generate_interface_data(directory_name, model, clusterized_training_signatures):
    dir_ = directory_name + '_DATA/'
    save_network_info(dir_, model)
    clusterisation.save_clusters_data(dir_, clusterized_training_signatures)
    print('saved interface data in ' + directory_name + '_DATA/')
    