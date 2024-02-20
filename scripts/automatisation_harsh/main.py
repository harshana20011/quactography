# from qiskit.visualization import plot_distribution
# from qiskit.primitives import Estimator, Sampler
# from qiskit.quantum_info import SparsePauliOp
# from qiskit.circuit.library import QAOAAnsatz
# from scipy.optimize import minimize

# import matplotlib.pyplot as plt
# import networkx as nx
import pandas as pd
import numpy as np

from visualisation_entry_graph import visualize_num_nodes
from connexions_qubits import connexions_edges
from generate_random_matrices import generate_random_adjacency_matrix
from generate_random_matrices import save_adjacency_matrix_to_csv
from hamiltonian import hc

# Code pour matrice d'adjacence déjà existante:
# df = pd.read_csv(r"scripts\automatisation_harsh\matrices\mat_adj.csv")
# mat_adj = np.array(df)


# Code pour une matrice générée aléatoirement:
num_nodes = 8
random_adj_matrix = generate_random_adjacency_matrix(num_nodes, num_zeros_to_add=20)
mat_adj = np.array(random_adj_matrix)
save_adjacency_matrix_to_csv(random_adj_matrix, filename="random_adjacency_matrix.csv")

# Lire la matrice :
df = pd.read_csv(r"random_adjacency_matrix.csv")

# Visualisation et détermination du nombre de noeuds dans le graphe:
num_nodes = visualize_num_nodes(df, mat_adj)

# Détermination des connexions possibles entre chaque noeud voisin, en considérant le dédoublement des connexions et détermination du nom des qubits:
connexions_edges(mat_adj, num_nodes)

# Le coût obligatoire (premier terme de Hamiltonien):
hc(mat_adj, num_nodes)
