#!/usr/bin/env python
# coding: utf-8

# In[20]:


#coding:utf8

import pandas as pd
import math
import scipy
import scipy.stats

#C'est la partie la plus importante dans l'analyse de données. D'une part, elle n'est pas simple à comprendre tant mathématiquement que pratiquement. D'autre, elle constitue une application des probabilités. L'idée consiste à comparer une distribution de probabilité (théorique) avec des observations concrètes. De fait, il faut bien connaître les distributions vues dans la séance précédente afin de bien pratiquer cette comparaison. Les probabilités permettent de définir une probabilité critique à partir de laquelle les résultats ne sont pas conformes à la théorie probabiliste.
#Il n'est pas facile de proposer des analyses de données uniquement dans un cadre univarié. Vous utiliserez la statistique inférentielle principalement dans le cadre d'analyses multivariées. La statistique univariée est une statistique descriptive. Bien que les tests y soient possibles, comprendre leur intérêt et leur puissance d'analyse dans un tel cadre peut être déroutant.
#Peu importe dans quelle théorie vous êtes, l'idée de la statistique inférentielle est de vérifier si ce que vous avez trouvé par une méthode de calcul est intelligent ou stupide. Est-ce que l'on peut valider le résultat obtenu ou est-ce que l'incertitude qu'il présente ne permet pas de conclure ? Peu importe également l'outil, à chaque mesure statistique, on vous proposera un test pour vous aider à prendre une décision sur vos résultats. Il faut juste être capable de le lire.

#Par convention, on place les fonctions locales au début du code après les bibliothèques.
def ouvrirUnFichier(nom):
    with open(nom, "r") as fichier:
        contenu = pd.read_csv(fichier)
    return contenu

#Théorie de l'échantillonnage (intervalles de fluctuation)
#L'échantillonnage se base sur la répétitivité.
#print("Résultat sur le calcul d'un intervalle de fluctuation")

donnees = pd.read_csv("./data/Echantillonnage-100-Echantillons.csv", encoding="utf-8")#Théorie de l'estimation (intervalles de confiance)
#L'estimation se base sur l'effectif.
#print("Résultat sur le calcul d'un intervalle de confiance")

#Théorie de la décision (tests d'hypothèse)
#La décision se base sur la notion de risques alpha et bêta.
#Comme à la séance précédente, l'ensemble des tests se trouve au lien : https://docs.scipy.org/doc/scipy/reference/stats.html
#print("Théorie de la décision")


# In[21]:


donnees = pd.read_csv(r"C:\Users\jeann\OneDrive\Documents\Master 1 - Geoint\Python\Seance_05\Exercice\src\data\Echantillonnage-100-Echantillons.csv")
print(donnees)


# In[22]:


#Calculer les moyennes
moyennes = donnees.mean()
print("1) Moyennes brutes :")
print(moyennes)


# In[23]:


#Arrondir chaque moyenne à l'entier avec la fonction native round()
moyennes_arrondies = moyennes.apply(lambda x: int(round(x, 0)))
print("\n2) Moyennes arrondies (entiers) :")
print(moyennes_arrondies)


# In[24]:


#Calculer la somme des trois moyennes arrondies
somme_moyennes = moyennes_arrondies.sum()
print("\n3) Somme des moyennes arrondies :", somme_moyennes)


# In[25]:


#Calculer les fréquences de l'échantillon : chaque moyenne / somme totale
frequences_echantillon = moyennes_arrondies / somme_moyennes


# In[26]:


#Arrondir les fréquences à 2 décimales avec la fonction native round()
frequences_echantillon_arrondies = frequences_echantillon.apply(lambda x: round(x, 2))
print("\n4) Fréquences de l'échantillon (arrondies à 2 décimales) :")
print(frequences_echantillon_arrondies)


# In[27]:


# Calculer l’intervalle de fluctuation à 95 % (zC = 1.96)
zC = 1.96
n = 1000  # taille totale de l’échantillon (somme des trois moyennes arrondies)

bornes_inf = frequences_echantillon_arrondies.apply(lambda p: round(p - zC * math.sqrt((p * (1 - p)) / n), 3))
bornes_sup = frequences_echantillon_arrondies.apply(lambda p: round(p + zC * math.sqrt((p * (1 - p)) / n), 3))

print("\n5) Intervalle de fluctuation à 95 % (zC = 1.96) :")
print("Borne inférieure :")
print(bornes_inf)
print("Borne supérieure :")
print(bornes_sup)


# In[28]:


#Prendre le premier échantillon et convertir en liste native
premier_echantillon = [frequences_echantillon_arrondies.iloc[0]]


# In[29]:


#Calculer la somme de la ligne (effectif total de l'échantillon)
n_echantillon = sum(premier_echantillon)
print("="*70)
print("ANALYSE DES INTERVALLES DE CONFIANCE - PREMIER ÉCHANTILLON")
print("="*70)
print(f"\n1) Échantillon isolé : {premier_echantillon}")
print(f"2) Effectif total de l'échantillon (n) : {n_echantillon}")


# In[30]:


#Calculer les fréquences pour chaque modalité
frequences = [round(x / n_echantillon, 3) for x in premier_echantillon]
print(f"3) Fréquences observées : {frequences}")


# In[31]:


import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro
import matplotlib.pyplot as plt


# In[32]:


#THÉORIE DE LA DÉCISION - TEST DE SHAPIRO-WILK
print("="*80)
print("3) THÉORIE DE LA DÉCISION - TEST DE NORMALITÉ (SHAPIRO-WILK)")
print("="*80)


# In[33]:


# Charger les deux fichiers CSV
d1 = pd.read_csv("data/Loi-normale-Test-1.csv")
d2 = pd.read_csv("data/Loi-normale-Test-2.csv")

# Convertir en listes natives (première colonne)
data1 = list(d1.iloc[:, 0])
data2 = list(d2.iloc[:, 0])

print(f"\nNombre de valeurs - Test 1 : {len(data1)}")
print(f"Nombre de valeurs - Test 2 : {len(data2)}")


# In[34]:


#APPLICATION DU TEST DE SHAPIRO-WILK
# Test pour la distribution 1
stat1, p_value1 = scipy.stats.shapiro(data1)

print("\n" + "-"*80)
print("TEST 1 (Loi-normale-Test-1.csv)")
print("-"*80)
print(f"Statistique W : {round(stat1, 6)}")
print(f"P-value : {round(p_value1, 6)}")


# In[35]:


#Test pour la distribution 2
stat2, p_value2 = scipy.stats.shapiro(data2)

print("\n" + "-"*80)
print("TEST 2 (Loi-normale-Test-2.csv)")
print("-"*80)
print(f"Statistique W : {round(stat2, 6)}")
print(f"P-value : {round(p_value2, 6)}")


# In[36]:


# INTERPRÉTATION DU TEST (seuil alpha = 0.05)
alpha = 0.05

print("\n" + "="*80)
print("INTERPRÉTATION DES RÉSULTATS (seuil α = 0.05)")
print("="*80)

print("\nRAPPEL DE LA RÈGLE DE DÉCISION :")
print("• Si p-value > α (0.05) → On NE rejette PAS H0")
print("  → La distribution PEUT être considérée comme normale")
print("• Si p-value ≤ α (0.05) → On REJETTE H0")
print("  → La distribution NE suit PAS une loi normale")

print("\n" + "-"*80)
print("RÉSULTAT POUR TEST 1 :")
print("-"*80)
if p_value1 > alpha:
    print(f"✓ P-value ({round(p_value1, 6)}) > α ({alpha})")
    print("✓ Décision : On NE REJETTE PAS H0")
    print("✓ Conclusion : La distribution TEST 1 suit une LOI NORMALE")
    resultat1 = "NORMALE"
else:
    print(f"✗ P-value ({round(p_value1, 6)}) ≤ α ({alpha})")
    print("✗ Décision : On REJETTE H0")
    print("✗ Conclusion : La distribution TEST 1 NE suit PAS une loi normale")
    resultat1 = "NON NORMALE"

print("\n" + "-"*80)
print("RÉSULTAT POUR TEST 2 :")
print("-"*80)
if p_value2 > alpha:
    print(f"✓ P-value ({round(p_value2, 6)}) > α ({alpha})")
    print("✓ Décision : On NE REJETTE PAS H0")
    print("✓ Conclusion : La distribution TEST 2 suit une LOI NORMALE")
    resultat2 = "NORMALE"
else:
    print(f"✗ P-value ({round(p_value2, 6)}) ≤ α ({alpha})")
    print("✗ Décision : On REJETTE H0")
    print("✗ Conclusion : La distribution TEST 2 NE suit PAS une loi normale")
    resultat2 = "NON NORMALE"


# In[37]:


#CONCLUSION FINALE

print("\n" + "="*80)
print("CONCLUSION FINALE")
print("="*80)

if resultat1 == "NORMALE" and resultat2 != "NORMALE":
    print("\n>>> La distribution qui suit une LOI NORMALE est : TEST 1")
elif resultat2 == "NORMALE" and resultat1 != "NORMALE":
    print("\n>>> La distribution qui suit une LOI NORMALE est : TEST 2")
elif resultat1 == "NORMALE" and resultat2 == "NORMALE":
    print("\n>>> Les DEUX distributions suivent une loi normale")
else:
    print("\n>>> AUCUNE des deux distributions ne suit une loi normale")


# In[38]:


# STATISTIQUES DESCRIPTIVES

print("\n" + "="*80)
print("STATISTIQUES DESCRIPTIVES COMPLÉMENTAIRES")
print("="*80)

print("\nTEST 1 :")
print(f"  Moyenne : {round(np.mean(data1), 3)}")
print(f"  Écart-type : {round(np.std(data1, ddof=1), 3)}")
print(f"  Médiane : {round(np.median(data1), 3)}")
print(f"  Min : {round(min(data1), 3)} | Max : {round(max(data1), 3)}")

print("\nTEST 2 :")
print(f"  Moyenne : {round(np.mean(data2), 3)}")
print(f"  Écart-type : {round(np.std(data2, ddof=1), 3)}")
print(f"  Médiane : {round(np.median(data2), 3)}")
print(f"  Min : {round(min(data2), 3)} | Max : {round(max(data2), 3)}")

