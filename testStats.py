import pandas as pd


df = pd.read_csv (r'makemoons_3_10_10_3_.csv')
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