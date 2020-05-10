import keras
import os
import numpy as np
import pandas as pd 


def get_result_layers(model,X):
    result_layers=[]
    for i in range (len(model.layers)-1):
        hidden_layers= keras.backend.function(
                [model.layers[0].input],   
                [model.layers[i].output,] 
                )    
        result_layers.append(hidden_layers([X])[0])  
    return result_layers

#permet de sélectionner uniquement les signatures dont la prédiction a été bonne
def get_goodXy (X,y, model, train_values):
        ynew = model.predict_classes(X)
        X_good =[]
        y_good=[]
        for i in range(len(X)):
            if train_values[ynew[i]] == y[i]:
                X_good.append(X[i])
                y_good.append(y[i])
        return X_good,y_good   

def get_train_y_class_number(y, train_values):
    for i in range(0, len(y)):
            if y[i] == 1:
                return train_values[i]
    
def get_train_y_classes_number(train_y, train_values):
    classes_number = []
    for y in train_y:
        classes_number.append(get_train_y_class_number(y, train_values))
    return classes_number



#Fonction permettant de sauvegarder les fonctions d'activation de chaque couche dans un répertoire
def save_result_layers(directory_name,X,y,result_layers):
    os.makedirs(directory_name, exist_ok=True)
    for i in range(0, len(result_layers)):
        f = open(directory_name + '/' + directory_name + '_result_l' + str(i+1) + '.csv', "w")
        for nb_X in range (len(X)):
           my_string=str(y[nb_X])+','
           for j in range (len(result_layers[i][nb_X])):
               my_string+=str(result_layers[i][nb_X][j])+','
           my_string=my_string [0:-1]
           my_string+='\n'
           f.write(my_string)
        print('saved ' + directory_name + '/' + directory_name + '_result_l' + str(i+1) + '.csv')
        f.close()
        
        
def predict_real_value(model, value, X_test_values,y_test_values, train_values):
    #print ("computing ",value)
    X_value=[]
    for i in range(0, len(X_test_values)):
        if (y_test_values[i]==value):
            X_value.append(X_test_values[i])
    X_value=np.array(X_value)
    train_datavalue=X_value
    Y_pred_classes = model.predict_classes(train_datavalue)
    y_pred_true_classes = []
    for class_index in Y_pred_classes:
        y_pred_true_classes.append(train_values[class_index])
    return y_pred_true_classes, train_datavalue

def generate_train_signatures(train_X, train_y, model, train_values, project_name):
    train_y_classes_number = get_train_y_classes_number(train_y, train_values)    
    X_good,y_good=get_goodXy(train_X, train_y_classes_number, model, train_values)   
    # Récupération des valeurs de tous les layers sauf le dernier
    result_layers= get_result_layers(model,X_good)        
    # Sauvegarde du fichier
    save_result_layers(project_name,X_good,y_good,result_layers)
    return get_train_signatures(model, project_name)


def generate_test_signatures(model, project_name, X_test_values, y_test_values, train_values, test_values):
    real_values_predict=[]
    for i in range (len(test_values)):
        dict_predict={}
        dict_predict['value']=test_values[i]
        dict_predict['prediction'], train_datavalue = predict_real_value(model, test_values[i],X_test_values,y_test_values, train_values)
        real_values_predict.append(dict_predict)
        result_layers_datavalue = get_result_layers(model, train_datavalue)
        filename = project_name + "_predict_" + str(test_values[i])
        save_result_layers(filename,train_datavalue,dict_predict['prediction'],result_layers_datavalue)
    return get_test_signatures(model, project_name, test_values)
        

def get_test_signatures(model, project_name, test_values):
    test_signatures = []
    for i in test_values:
        for j in range(0, len(model.layers) - 1):
            directory_name = project_name + '_predict_' + str(i) + '/'
            df = pd.read_csv(directory_name + project_name + '_predict_' + str(i) +  "_result_l" + str(j+1) + ".csv")
            test_signatures.append(df.copy())
    return test_signatures

def get_train_signatures(model, project_name):
    train_signatures = []
    for j in range(0, len(model.layers) - 1):
        directory_name = project_name + '/'
        df = pd.read_csv(directory_name + project_name + "_result_l" + str(j+1) + ".csv")
        train_signatures.append(df.copy())
    return train_signatures