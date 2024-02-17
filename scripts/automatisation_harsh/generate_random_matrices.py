import numpy as np
import pandas as pd


def generate_random_adjacency_matrix(num_nodes, max_weight=10):
    """
    Génère une matrice d'adjacence symétrique avec des poids aléatoires.

    Args:
        num_nodes (int): Nombre de nœuds dans le graphe.
        max_weight (int): Poids maximum autorisé pour les arêtes.

    Returns:
        np.ndarray: Matrice d'adjacence générée.
    """
    # Matrice vide de taille num_nodes x num_nodes:
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    # Remplir la matrice avec des poids aléatoires pour les arêtes
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            weight = np.random.randint(
                1, max_weight + 1
            )  # Poids aléatoire entre 1 et max_weight
            adjacency_matrix[i, j] = weight
            adjacency_matrix[j, i] = weight  # La matrice est symétrique

    return adjacency_matrix


def save_adjacency_matrix_to_csv(adjacency_matrix, filename="adjacency_matrix.csv"):
    """
    Sauvegarde la matrice d'adjacence dans un fichier CSV.

    Args:
        adjacency_matrix (np.ndarray): Matrice d'adjacence.
        filename (str): Nom du fichier CSV.
    """
    df = pd.DataFrame(adjacency_matrix)
    df.to_csv(filename, index=False)
