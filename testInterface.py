# -*- coding: utf-8 -*-

from tkinter import Tk, Canvas, Label
from scipy import *

def center(coords):
    return [(coords[2] + coords[0]) / 2, (coords[3] + coords[1]) / 2]
    

nbCouche = 3
nbNeuronneCouche = 4

fenetre = Tk()


label = Label(fenetre, text="DL")
label.pack()

longueurCanvas = nbCouche * 200
largueurCanvas = nbNeuronneCouche * 100
# canvas
canvas = Canvas(fenetre, width=longueurCanvas, height=largueurCanvas, background='white')

neurones = []


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
        neurones.append(canvas.create_oval(x0,y0,x1,y1,width=2,fill='white',outline="black",tag=gauche+"$"+droite))
        y0 = y0 + 90
        y1 = y1 + 90
        
arretes = []

print(neurones)
        
for i in range(nbCouche - 1) :
    for j in range(nbNeuronneCouche) :
        for k in range(nbNeuronneCouche) :
            source_coords = canvas.bbox(neurones[j + i*nbNeuronneCouche])
            dest_coords = canvas.bbox(neurones[k + i*nbNeuronneCouche + nbNeuronneCouche])
            source_center = center(source_coords)
            dest_center = center(dest_coords)
            arrete_id = canvas.create_line(source_center[0], source_center[1], dest_center[0], dest_center[1])
            canvas.tag_lower(arrete_id)
            arretes.append(arrete_id)


    
    
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
