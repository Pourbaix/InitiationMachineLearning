# Import numpy for numeric calculation.
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
# on charge le jeux de données
mnist = fetch_openml('mnist_784', version=1)

print(mnist.data.shape)
print(mnist.target.shape)

print(mnist.target[65000])

sample = np.random.randint(70000, size=5000)
data = mnist.data[sample]
target = mnist.target[sample]

print(data.shape)

# On va séparer données d'entrainement et données de test
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.8)

