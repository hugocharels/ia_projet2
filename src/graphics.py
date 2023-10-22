import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Lire les données CSV
data = pd.read_csv('results_newworld1.csv')

# Filtrer les données par algorithme et type (WMDP ou BetterValueFunction)
minimax_data = data[(data['Algorithm'] == 'minimax') & (data['WMDP'] == 'WorldMDP')]
alpha_beta_data_wmdp = data[(data['Algorithm'] == 'alpha_beta') & (data['WMDP'] == 'WorldMDP')]
alpha_beta_data_better_value = data[(data['Algorithm'] == 'alpha_beta') & (data['WMDP'] == 'BetterValueFunction')]

# Ajouter un petit décalage aux valeurs de l'axe y pour les rendre positives
offset = 1e-6
minimax_y = minimax_data['Expanded States'] + offset
alpha_beta_y_wmdp = alpha_beta_data_wmdp['Expanded States'] + offset
alpha_beta_y_better_value = alpha_beta_data_better_value['Expanded States'] + offset

# Créer les courbes
plt.plot(minimax_data['Depth'], minimax_y, label='Minimax (WMDP)')
plt.plot(alpha_beta_data_wmdp['Depth'], alpha_beta_y_wmdp, label='Alpha-Beta (WMDP)')
plt.plot(alpha_beta_data_better_value['Depth'], alpha_beta_y_better_value, label='Alpha-Beta (BetterValueFunction)')

# Mettre l'axe des ordonnées (y) en échelle logarithmique
plt.yscale('log')

# Définir les étiquettes des axes et le titre
plt.xlabel('Profondeur')
plt.ylabel('Nombre de nœuds étendus (échelle logarithmique)')
plt.title('Nombre de nœuds étendus par profondeur')

# Afficher la légende
plt.legend()

# Enregistrer le graphique dans un fichier image
plt.savefig("graph_newworld2.png")
