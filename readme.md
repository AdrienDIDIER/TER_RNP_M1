# Que se passe-t-il dans mon réseau de neurones profond ?
 
## Auteurs
 
DIDIER Adrien - MARTIN Loïc - PORTAL Pierre - TROUCHE Aurélien
 
## Description
 
Ceci est une librairie permettant de faciliter l'analyse des signatures des fonctions d'activation des neurones dans une réseau de neurones profond. 

Elle inclut :

- des fonctions Python de sauvegarde, de discrétisation et de clusterisation des valeurs des fonctions d'activation des données d'apprentissage et des données à prédire

- une interface web utilisant Node/HTML/Javascript permettant de visualiser les données générées

Cette librairie est le fruit d'un projet de TER en Master 1 Informatique à l'université de Montpellier, elle est censée fonctionner pour tout type de données en entrée. 

Nous avons essentiellement utilisé la base de données MNIST comme jeu de données et nous fournissons quelques fonctions permettant de faciliter l'utilisation de celle-ci avec notre librairie.
 
 
 
## Mode d’emploi (le mode d'emploi de votre programme (un exemple pour pouvoir lancer votre programme pour quelqu'un qui ne connaît pas puisse facilement l'utiliser.)
 
Programme python
 
Programme js avec node

Création du modèle

Apprentissage du modèle

Extraction des signatures

Discrétisation des signatures

Clusterisation des signatures

Affichage des résultats
 
 
Pour faire fonctionner l'interface :

- Installer node
	
- Accéder au dossier mnist_web avec le terminal
	
- Executer la commande "npm install package.json"
	
- Executer les commandes "npm install express", "npm install csv-parse", "npm install mustache" (installation des packages nécessaires)
	
- Executer la commande "node index.js"
	
- Accéder à l'URL "http://localhost:2400/"

Pour adapter l'interface :

	- Il est nécessaire que les fichiers csv aient les mêmes en-têtes

	- Il est nécesaire que le fichier all_clusters.csv ait exactement le même nombre de lignes et de colonnes

	- Il est nécessaire que le fichier clusterized_values.csv ait le même nombre de colonnes
 
## Dépendances
 
Spyder 3.3.6

Python 3.7.4

Pandas 0.25.1

Scikit-learn 0.21.3

Tensorflow 1.15.0

Keras 2.2.4

Numpy 1.16.5
 
Google Chrome 81 (81.0.4044.129)

express 4.17.1

csv-parse 4.9.1

mustache 4.0.1

 
## Démonstration
 
si vous avez des vidéos sur youtube -> mettre un lien vers la vidéo pour que les personnes puissent avoir une démonstration. Attention une vidéo est plus facile qu'une démonstration en ligne via un serveur car ce dernier ne pourra pas être utilisé.
 OU INSERER DIRECTEMENT LA VIDEO DANS LE GIT
 
## Rapport
 
Vous pouvez accéder au rapport de ce projet depuis ce github.
 
 

 
 
 
 
 
 
 





