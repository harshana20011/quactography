from qiskit.visualization import plot_distribution
from qiskit.primitives import Estimator, Sampler
from qiskit.circuit.library import QAOAAnsatz
from scipy.optimize import minimize
import pandas as pd
import numpy as np

from visualisation_entry_graph import visualize_num_nodes
from connexions_qubits import connexions_edges
from generate_random_matrices import generate_random_adjacency_matrix
from generate_random_matrices import save_adjacency_matrix_to_csv
from hamiltonian import hc, hdep, hfin, hint

# Code pour matrice d'adjacence déjà existante :
# df = pd.read_csv(r"scripts\automatisation_harsh\matrices\mat_adj.csv")
# mat_adj = np.array(df)

# Code pour une matrice générée aléatoirement :
num_nodes = 8
random_adj_matrix = generate_random_adjacency_matrix(num_nodes, num_zeros_to_add=20)
mat_adj = np.array(random_adj_matrix)
save_adjacency_matrix_to_csv(random_adj_matrix, filename="random_adjacency_matrix.csv")

# Lire la matrice :
df = pd.read_csv(r"random_adjacency_matrix.csv")

# Visualisation et détermination du nombre de noeuds dans le graphe :
num_nodes = visualize_num_nodes(df, mat_adj)

# Déterminer les différentes caractéristiques du graphe :
number_of_edges, weights, depart, destination, q_indices, all_weights_sum = (
    connexions_edges(mat_adj, num_nodes)
)

# Le coût obligatoire (premier terme de Hamiltonien) :
hc = hc(number_of_edges, weights, all_weights_sum)

# Le terme associé au respect de la contrainte de départ :
# Déterminer le noeud de départ :
noeud_de_depart = 0
hdep = hdep(noeud_de_depart, depart, q_indices, destination, number_of_edges)


# Le terme associé au respect de la contrainte de fin :
# Déterminer le noeud de fin :
noeud_de_fin = 5
hfin = hfin(noeud_de_fin, depart, q_indices, destination, number_of_edges)

hint = hint(
    noeud_de_depart, noeud_de_fin, depart, q_indices, destination, number_of_edges
)

# Alpha : coût associé aux contraintes:
alpha = 0.8
# Fonction coût en représentation QUBO:
h = -hc + alpha * ((hdep**2) + (hfin**2) + hint)


# Create QAOA circuit.
ansatz = QAOAAnsatz(h, reps=1)
ansatz.decompose(reps=3).draw()

# Run on local estimator and sampler. Fix seeds for results reproducibility.
estimator = Estimator(options={"shots": 1000000, "seed": 42})
sampler = Sampler(options={"shots": 1000000, "seed": 42})


# Cost function for the minimizer.
# Returns the expectation value of circuit with Hamiltonian as an observable.
def cost_func(params, estimator, ansatz, hamiltonian):
    cost = (
        estimator.run(ansatz, hamiltonian, parameter_values=params).result().values[0]
    )
    return cost


# Generate starting point. Fixed to zeros for results reproducibility.
# x0 = 2 * np.pi * np.random.rand(ansatz.num_parameters)
x0 = np.zeros(ansatz.num_parameters)

res = minimize(cost_func, x0, args=(estimator, ansatz, h), method="COBYLA")
print(res)

# cost_func([ 8.250e-01  ,2.622e-01  ,5.591e-02 ,-2.605e-01],estimator, ansatz, h)

# Get probability distribution associated with optimized parameters.
circ = ansatz.copy()
circ.measure_all()
dist = sampler.run(circ, res.x).result().quasi_dists[0]


plot_distribution(dist.binary_probabilities(), figsize=(8, 8))

print(max(dist.binary_probabilities(), key=dist.binary_probabilities().get))  # type: ignore
print(sorted(dist.binary_probabilities(), key=dist.binary_probabilities().get))  # type: ignore
