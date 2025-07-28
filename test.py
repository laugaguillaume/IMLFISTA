import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Matrice diagonale D à coefficients strictement positifs
D = np.diag([2.0, 1.0, 0.5])  # Exemple : D = diag(2, 1, 0.5)

# Échantillonnage de points dans [-1, 1]^3
n_points = 10000000
points = 2 * np.random.rand(n_points, 3) - 1  # Uniforme dans [-1, 1]^3

# Calcul de la norme \|D x\|_1
norms = np.sum(np.abs((D @ points.T).T), axis=1)

# Garde les points tels que \|D x\|_1 <= 1
inside = points[norms <= 1]

# Affichage
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(inside[:, 0], inside[:, 1], inside[:, 2], s=0.5, alpha=0.3)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Boule unité de la norme ||D·x||₁')

plt.show()
