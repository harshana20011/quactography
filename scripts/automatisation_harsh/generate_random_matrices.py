import numpy as np
import pandas as pd

# todo: use saved to npz matrix


def generate_random_adjacency_matrix(num_nodes, num_zeros_to_add, max_weight=0.5):
    """
    Generates a symetric adjacency matrix with random weights, and adds num_zeros_to_add zeros to the matrix to break the complete graph

    Args:
        num_zeros_to_add (int): Number of zeros to add to the adjacency matrix
        num_nodes (int): Number of nodes in the graph
        max_weight (float): Maximum weight for the edges

    Returns:
        adjacency_matrix (np.ndarray): Adjacency matrix of the graph with first row as number of nodes and matrix starts at second row
    """
    # Matrice vide de taille num_nodes x num_nodes:
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=float)

    # Remplir la matrice avec des poids aléatoires pour les arêtes
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            weight = np.random.uniform(
                0, max_weight
            )  # Poids aléatoire entre 0 et max_weight
            adjacency_matrix[i, j] = weight
            adjacency_matrix[j, i] = weight  # La matrice est symétrique
            # Mettez à zéro au moins la moitié des connexions supplémentaires

    for _ in range(num_zeros_to_add):
        i, j = np.random.randint(0, num_nodes, size=2)
        adjacency_matrix[i, j] = 0
        adjacency_matrix[j, i] = 0

    return adjacency_matrix


def generate_random_adjacency_matrix_from_zeros(num_nodes, probability):
    """Generates a random matrix from all zeros then adds edges based on probability.

    Args:
        num_nodes (int): number of nodes in the graph
        probability (float): probability of adding an edge between two nodes

    Returns:
         adjacency_matrix (np.ndarray): Adjacency matrix of the graph with first row as number of nodes and matrix starts at second row
    """
    # Initialize a null matrix with zeros :
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=float)

    # Fill the matrix with random edges based probability :
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and np.random.rand() < probability:
                # Set an edge between nodes i and j :
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1

    # Assign random weights to existing edges :
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adjacency_matrix[i, j] == 1:
                weight = np.random.uniform(0.01, 0.5)
                adjacency_matrix[i, j] = weight
                adjacency_matrix[j, i] = weight

    return adjacency_matrix


def save_adjacency_matrix_to_csv(adjacency_matrix, filename="adjacency_matrix.csv"):
    """
    Save adjacency matrix to a CSV file.

    Args:
        adjacency_matrix (np.ndarray): adjacency matrix to save
        filename (str): name of the file to save the adjacency matrix to
    """
    adj_matrix_csv = pd.DataFrame(adjacency_matrix)
    adj_matrix_csv.to_csv(filename, index=False)


def save_adjacency_matrix_to_npz(adjacency_matrix, filename="adjacency_matrix.npz"):
    """
    Save adjacency matrix to a npz file.

    Args:
        adjacency_matrix (np.ndarray): adjacency matrix to save
        filename (str): name of the file to save the adjacency matrix to
    """
    np.savez(filename, adjacency_matrix=adjacency_matrix)
