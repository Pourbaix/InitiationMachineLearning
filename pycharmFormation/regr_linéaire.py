# Import numpy for numeric calculation.
import numpy as np
# Import pandas for better data sorting and display.
import pandas as pd
# Import pyplot to process data and create graphs.
import matplotlib.pyplot as plt
import pygments.lexers.console

house_data = pd.read_csv("house.csv")
house_data = house_data[house_data["loyer"] < 10000]
# On print les 10 première données de notre jeux. On va donc essayer de prédire avec
# une surface le loyer que l'on devra payer
print("\nFirst 10 datas of the data set:")
print(house_data[:10])
# Donc Input => Surface et Output => Loyer
plt.plot(house_data["surface"], house_data["loyer"], "ro", markersize=4)
plt.show()
# On observe quelques données abérantes que l'on va simplement supprimées du
# graphe au chargement des données en ajoutant la ligne 10

X = np.matrix([np.ones(house_data.shape[0]), house_data["surface"]]).T
Y = np.matrix(house_data["loyer"]).T
# print(X[:10])
# print(Y[:10])
theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
# Avec ce theta on va pouvoir déja faire des prédictions avec un
# certain nombre de mètres carrés
print(f"Loyer estimé pour 30 mètre carré: {round(theta.item(0) + theta.item(1) * 30)}€")
# On peut également afficher la droite de prédiction
plt.plot([0, 250], [theta.item(0), theta.item(1) + 250 * theta.item(1)], linestyle="--", c="#000000")
plt.show()
# On peut alors afficher les données et la droite dde prédiction
plt.plot([0, 250], [theta.item(0), theta.item(1) + 250 * theta.item(1)], linestyle="--", c="#000000")
plt.plot(house_data["surface"], house_data["loyer"], "ro", markersize=4)
plt.show()
