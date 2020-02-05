# coding: utf-8
 
from tkinter import * 
from scipy import *

nbCouche = 3
nbNeuronneEntree = 2
nbNeuronneHidden = 3
nbNeuronneSortie = 2

listNeu = [nbNeuronneEntree,nbNeuronneHidden,nbNeuronneSortie]

def maximum(liste):
    maxi = liste[0]
    for i in liste:
        if i >= maxi:
            maxi = i
    return maxi

maxN = maximum(listNeu)

fenetre = Tk()

label = Label(fenetre, text="DL")
label.pack()

longueurCanvas = nbCouche * 200
largueurCanvas = maxN * 100
# canvas
canvas = Canvas(fenetre, width=longueurCanvas, height=largueurCanvas, background='white')

x0 = 50
x1 = 130
y0 = largueurCanvas/(listNeu[0]+1)
y1 = y0 + 70

for variable in range(len(listNeu)) :

    for variable1 in range(listNeu[variable]):
        #print(gauche +" "+ droite)
        neuronne = canvas.create_oval(x0,y0,x1,y1,width=2,outline="black")
        y0 = largueurCanvas/(listNeu[variable]+1)
        y1 = y0 + 70
    
    x0 = x0 +200
    x1 = x1 +200
    
                    
canvas.pack()

fenetre.mainloop()
