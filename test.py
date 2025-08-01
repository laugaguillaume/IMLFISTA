'''
Rank estimation using Lanczos randomization method.
'''

import numpy as np
from scipy.linalg import qr, svd
import warnings

def estimate_rank(linear_operator, input_dim, output_dim=None,
                  oversampling=10, tolerance=1e-12, max_iterations=None):
    """
    Estime le rang d'un opérateur linéaire en utilisant la méthode de Lanczos randomisé.

    Parameters:
    -----------
    linear_operator : callable
        Fonction qui prend un vecteur x et retourne Ax
    input_dim : int
        Dimension de l'espace d'entrée
    output_dim : int, optional
        Dimension de l'espace de sortie (si None, supposé égal à input_dim)
    oversampling : int, default=10
        Nombre de vecteurs supplémentaires pour améliorer la précision
    tolerance : float, default=1e-12
        Seuil en dessous duquel une valeur singulière est considérée comme nulle
    max_iterations : int, optional
        Nombre maximum d'itérations (si None, utilise input_dim)

    Returns:
    --------
    rank : int
        Rang estimé de l'opérateur
    singular_values : array
        Valeurs singulières de la matrice échantillonnée
    """

    if output_dim is None:
        output_dim = input_dim

    if max_iterations is None:
        max_iterations = min(input_dim, output_dim)

    # Estimation initiale du rang avec échantillonnage aléatoire
    initial_samples = min(20, input_dim)

    # Générer des vecteurs aléatoires orthonormés
    np.random.seed(42)  # Pour la reproductibilité
    Omega = np.random.randn(input_dim, initial_samples)
    Q, _ = qr(Omega, mode='economic')

    # Appliquer l'opérateur linéaire
    Y = np.zeros((output_dim, initial_samples))
    for i in range(initial_samples):
        Y[:, i] = linear_operator(Q[:, i])

    # Décomposition SVD pour estimer le rang initial
    try:
        _, s, _ = svd(Y, full_matrices=False)
        initial_rank_estimate = np.sum(s > tolerance)
    except:
        initial_rank_estimate = min(initial_samples, min(input_dim, output_dim))

    # Ajuster le nombre d'échantillons basé sur l'estimation initiale
    target_samples = min(initial_rank_estimate + oversampling,
                        min(input_dim, output_dim),
                        max_iterations)

    if target_samples <= initial_samples:
        rank = initial_rank_estimate
        return rank, s

    # Générer plus d'échantillons si nécessaire
    Omega_full = np.random.randn(input_dim, target_samples)
    Q_full, _ = qr(Omega_full, mode='economic')

    # Appliquer l'opérateur à tous les échantillons
    Y_full = np.zeros((output_dim, target_samples))
    for i in range(target_samples):
        Y_full[:, i] = linear_operator(Q_full[:, i])

    # Décomposition SVD finale
    try:
        _, s_full, _ = svd(Y_full, full_matrices=False)
        rank = np.sum(s_full > tolerance)
    except Exception as e:
        warnings.warn(f"Erreur dans la SVD: {e}")
        rank = target_samples
        s_full = np.ones(target_samples)

    return rank, s_full

def adaptive_rank_estimation(linear_operator, input_dim, output_dim=None,
                           tolerance=1e-12, max_samples=None):
    """
    Version adaptative qui ajoute progressivement des échantillons
    jusqu'à stabilisation du rang.
    """
    if output_dim is None:
        output_dim = input_dim

    if max_samples is None:
        max_samples = min(input_dim, output_dim)

    batch_size = 5
    current_samples = 0
    previous_rank = 0
    stable_count = 0

    np.random.seed(42)
    all_samples = np.random.randn(input_dim, max_samples)
    Q, _ = qr(all_samples, mode='economic')

    Y = np.zeros((output_dim, max_samples))

    while current_samples < max_samples:
        # Ajouter un nouveau batch d'échantillons
        next_batch = min(batch_size, max_samples - current_samples)

        for i in range(next_batch):
            Y[:, current_samples + i] = linear_operator(Q[:, current_samples + i])

        current_samples += next_batch

        # Calculer le rang avec les échantillons actuels
        try:
            _, s, _ = svd(Y[:, :current_samples], full_matrices=False)
            current_rank = np.sum(s > tolerance)
        except:
            current_rank = current_samples
            s = np.ones(current_samples)

        # Vérifier la stabilité
        if current_rank == previous_rank:
            stable_count += 1
        else:
            stable_count = 0

        # Arrêter si le rang est stable ou si on a atteint le maximum théorique
        if stable_count >= 3 or current_rank == min(current_samples, min(input_dim, output_dim)):
            break

        previous_rank = current_rank

    return current_rank, s

# Exemple d'utilisation
def example_usage():
    import deepinv as dinv
    """Exemple avec une matrice de rang 3"""

    # Créer une matrice de test de rang 3
    np.random.seed(123)
    U = np.random.randn(100, 3)
    V = np.random.randn(3, 80)
    true_matrix = U @ V  # Matrice 100x80 de rang 3

    def my_operator(x):
        return true_matrix @ x

    # Estimer le rang
    estimated_rank, singular_values = estimate_rank(
        my_operator,
        input_dim=80,
        output_dim=100,
        tolerance=1e-10
    )

    print(f"Rang réel: 3")
    print(f"Rang estimé: {estimated_rank}")
    print(f"Premières valeurs singulières: {singular_values[:10]}")

    # Version adaptative
    adaptive_rank, _ = adaptive_rank_estimation(
        my_operator,
        input_dim=80,
        output_dim=100,
        tolerance=1e-10
    )

    print(f"Rang estimé (adaptatif): {adaptive_rank}")

if __name__ == "__main__":
    example_usage()