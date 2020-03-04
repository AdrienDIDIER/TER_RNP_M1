import pandas as pd


df = pd.read_csv (r'layers_to_csv_2_3_1_.csv')
print("\nMoyenne du résultat de la fonction d'activation pour chaque neurones sans prise en compte classes : ")
print(df.mean(axis = 0)[1:len(df)])   

print("\nMédiane du résultat de la fonction d'activation pour chaque neurones sans prise en compte classes : ")
print(df.median(axis = 0)[1:len(df)])   

print("\nMoyenne Classe/Neurones : ")
print(df.groupby('Classe').mean())

print("\nMédiane Classe/Neurones : ")
print(df.groupby('Classe').median())

print("\nMin Classe/Neurones : ")
print(df.groupby('Classe').min())

print("\nMax Classe/Neurones : ")
print(df.groupby('Classe').max())

data = pd.DataFrame({
    "Histogramme":df.iloc[:,0]
    })
hist = data.hist(bins=2)
df.plot.box()


dataN2 = pd.DataFrame({
        "Histogramme N2":df.iloc[:,3]
        })
hist = dataN2.hist(bins=3)
df.plot.box()


df["valQ"]=pd.qcut(df["N2"],2,labels=["0","1"])
df["valQ"].describe()
print("test :",df["valQ"])