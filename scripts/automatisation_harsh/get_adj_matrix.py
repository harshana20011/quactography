import pandas as pd
import numpy as np
import sympy as smp
import numpy as np
import rustworkx as rx
from rustworkx.visualization import mpl_draw as draw
import matplotlib.pyplot as plt

from generate_random_matrices import generate_random_adj_matrix_given_nodes_edges
from generate_random_matrices import generate_random_adjacency_matrix_from_zeros
from generate_random_matrices import save_adjacency_matrix_to_csv


def get_adj_matrix(csv_path: str):
    """Read adjacency matrix from csv file and convert to numpy array."""
    adj_matrix_from_csv = pd.read_csv(csv_path)
    mat_adj = np.array(adj_matrix_from_csv)
    return adj_matrix_from_csv, mat_adj


# New method to get random adjacency matrix from given number of nodes and edges:
def get_random_adj_matrix_given_nodes_edges(
    num_nodes: int, num_edges: int, edges_matter: bool
):
    path = generate_random_adj_matrix_given_nodes_edges(
        num_nodes, num_edges, edges_matter
    )
    adj_matrix_from_csv, mat_adj = get_adj_matrix(path)
    return adj_matrix_from_csv, mat_adj


# Old method to get random adjacency matrix from zeros with probabilities:
def get_random_adj_matrix_with_probability(num_nodes: int, probability_of_edge: float):
    """Generate a random adjacency matrix with probability of edge.

    Args:
        num_nodes (int): number of nodes in the graph
        probability_of_edge (float): probability of adding an edge between two nodes

    Returns:
         Returns:
        mat_adj (np.ndarray) : adjacency matrix of the graph
        adj_matrix_from_csv (pd.DataFrame) : adjacency matrix of the graph in pandas DataFrame format
    """
    random_adj_matrix = generate_random_adjacency_matrix_from_zeros(
        num_nodes, probability=probability_of_edge
    )
    mat_adj = np.array(random_adj_matrix)
    file_name = "matrices/random_adjacency_matrix.csv"

    save_adjacency_matrix_to_csv(
        random_adj_matrix,
        filename=file_name,
    )

    adj_matrix_from_csv, mat_adj = get_adj_matrix(file_name)
    return adj_matrix_from_csv, mat_adj


# Test:
# get_random_adj_matrix_given_nodes_edges(4, 1000, True)
