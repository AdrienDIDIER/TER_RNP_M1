# coding: utf-8
 
from tkinter import * 
from scipy import *

nbCouche = 3
nbNeuronneCouche = 3


fenetre = Tk()

label = Label(fenetre, text="DL")
label.pack()

longueurCanvas = nbCouche * 200
largueurCanvas = nbNeuronneCouche * 100
# canvas
canvas = Canvas(fenetre, width=longueurCanvas, height=largueurCanvas, background='white')

x0 =50
x1 =130
y0 = 10
y1 = 90
for variable in range(nbCouche) :
    if variable != 0 :
        x0 = x0 +200
        x1 = x1 +200
        y0 = 10
        y1 = 90
    for variable1 in range(nbNeuronneCouche) :
        gauche = str(x0) +"|"+str(y0+(y1-y0)/2)
        droite = str(x1) +"|"+str(y0+(y1-y0)/2)
        #print(gauche +" "+ droite)
        neuronne = canvas.create_oval(x0,y0,x1,y1,width=2,outline="black",tag=gauche+"$"+droite)
        y0 = y0 + 90
        y1 = y1 + 90

nbNeuronnesT = nbCouche * nbNeuronneCouche;

for var in range(nbNeuronnesT) :
    if var!=0:
        if var!=3:
            if var!=6:
                if var!=9:
                    recupTags = canvas.gettags(var)
                    sepCoord = str(recupTags).split("$");
                    recupCoord = sepCoord[-1][:-3]
                    splitCoord = recupCoord.split("|")
                    
              #à poursuivre
                    
canvas.pack()

fenetre.mainloop()
