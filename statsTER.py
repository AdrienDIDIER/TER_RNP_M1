import numpy as np
import matplotlib.pyplot as plt
import csv

nbClasses = 2
nbNeurones = 3

classe = []
n1_0 = []
n2_0 = []
n3_0 = []
n1_1 = []
n2_1 = []
n3_1 = []

name = "makemoons_3_10_10_3_.csv"
with open(name,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        if row[0]=="0": #ou "0.0" si utilisation code_v2
            n1_0.append(float(row[2]))
            n2_0.append(float(row[3]))
            n3_0.append(float(row[4]))
        else:
            n1_1.append(float(row[2]))
            n2_1.append(float(row[3]))
            n3_1.append(float(row[4]))


print(n1_0)


print("Statistiques :")
for i in range(nbClasses):
    for j in range(1, nbNeurones+1):
        #print(j,i)
        print("Moyennes n°%d couche %d : %.16f" % (j,i, np.mean(eval("n"+"%s_%s" % (j,i)))))


     




plt.plot(classe, label='Classe')
plt.ylabel('')
plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.show()