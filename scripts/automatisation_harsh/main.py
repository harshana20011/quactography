from qiskit.primitives import Estimator, Sampler
from qiskit.circuit.library import QAOAAnsatz
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import multiprocessing
import networkx as nx
import pandas as pd
import numpy as np
import itertools


from visualisation_entry_graph import visualize_num_nodes
from connexions_qubits import connexions_edges
from generate_random_matrices import generate_random_adjacency_matrix
from generate_random_matrices import save_adjacency_matrix_to_csv
from hamiltonian import hc, hdep, hfin, hint


# Fonction à faire en parallèle :
def _find_shortest_path_parallel(args):
    hc1 = args[0]
    hdep1 = args[1]
    hfin1 = args[2]
    hint1 = args[3]
    alpha = args[4]

    # Fonction coût en représentation QUBO:
    h = -hc1 + alpha * ((hdep1**2) + (hfin1**2) + hint1)

    # Create QAOA circuit.
    ansatz = QAOAAnsatz(h, reps=1)

    # print(ansatz.decompose(reps=1).draw())

    # Run on local estimator and sampler. Fix seeds for results reproducibility.
    estimator = Estimator(options={"shots": 1000000, "seed": 42})
    sampler = Sampler(options={"shots": 1000000, "seed": 42})

    # Cost function for the minimizer.
    # Returns the expectation value of circuit with Hamiltonian as an observable.
    def cost_func(params, estimator, ansatz, hamiltonian):
        cost = (
            estimator.run(ansatz, hamiltonian, parameter_values=params)
            .result()
            .values[0]
        )
        return cost

    # Generate starting point. Fixed to zeros for results reproducibility.
    # x0 = 2 * np.pi * np.random.rand(ansatz.num_parameters)
    x0 = np.zeros(ansatz.num_parameters)

    res = minimize(cost_func, x0, args=(estimator, ansatz, h), method="COBYLA")
    # print(res)

    min_cost = cost_func(res.x, estimator, ansatz, h)

    # print(f"Minimum cost: {min_cost}")

    # Get probability distribution associated with optimized parameters.
    circ = ansatz.copy()
    circ.measure_all()
    dist = sampler.run(circ, res.x).result().quasi_dists[0]

    # plot_distribution(dist.binary_probabilities(), figsize=(7, 5))

    # print(max(dist.binary_probabilities(), key=dist.binary_probabilities().get))  # type: ignore
    bin_str = list(map(int, max(dist.binary_probabilities(), key=dist.binary_probabilities().get)))  # type: ignore
    bin_str.reverse()
    bin_str = np.array(bin_str)

    # Concaténer chaque liste en une seule chaîne de caractères
    str_path = ["".join(map(str, bin_str))]  # type: ignore
    str_path = str_path[0]  # type: ignore

    # Save parameters alpha and min_cost with path in csv file:
    alpha_min_cost = [alpha, min_cost, str_path]

    # print(sorted(dist.binary_probabilities(), key=dist.bina
    # ry_probabilities().get))  # type: ignore
    print("Finished with alpha : ", alpha)

    return (res, min_cost, alpha_min_cost)


def main():

    # # Code pour matrice d'adjacence déjà existante :
    # df = pd.read_csv(r"scripts\automatisation_harsh\matrices\mat_adj.csv")
    # mat_adj = np.array(df)

    # Code pour une matrice générée aléatoirement :
    num_nodes = 7
    random_adj_matrix = generate_random_adjacency_matrix(num_nodes, num_zeros_to_add=0)
    mat_adj = np.array(random_adj_matrix)
    save_adjacency_matrix_to_csv(
        random_adj_matrix, filename="random_adjacency_matrix.csv"
    )

    # Lire la matrice :
    df = pd.read_csv(r"random_adjacency_matrix.csv")

    # Visualisation et détermination du nombre de noeuds dans le graphe :
    num_nodes = visualize_num_nodes(df, mat_adj)

    # Déterminer les différentes caractéristiques du graphe :
    number_of_edges, weights, depart, destination, q_indices, all_weights_sum = (
        connexions_edges(mat_adj, num_nodes)
    )

    # Le coût obligatoire (premier terme de Hamiltonien) :
    hc1 = hc(number_of_edges, weights, all_weights_sum)

    # Le terme associé au respect de la contrainte de départ :
    # Déterminer le noeud de départ :
    noeud_de_depart = 0
    hdep1 = hdep(noeud_de_depart, depart, q_indices, destination, number_of_edges)

    # Le terme associé au respect de la contrainte de fin :
    # Déterminer le noeud de fin :
    noeud_de_fin = 5
    hfin1 = hfin(noeud_de_fin, depart, q_indices, destination, number_of_edges)

    hint1 = hint(
        noeud_de_depart, noeud_de_fin, depart, q_indices, destination, number_of_edges
    )

    # Alpha : coût associé aux contraintes:
    alphas = [
        0.5 * all_weights_sum,
        all_weights_sum,
        2 * all_weights_sum,
        3 * all_weights_sum,
        4 * all_weights_sum,
    ]

    alpha_min_costs = []

    # Nombre de processeurs :
    nbr_processes = multiprocessing.cpu_count()

    pool = multiprocessing.Pool(nbr_processes)
    results = pool.map(
        _find_shortest_path_parallel,
        zip(
            itertools.repeat(hc1),
            itertools.repeat(hdep1),
            itertools.repeat(hfin1),
            itertools.repeat(hint1),
            alphas,
        ),
    )
    pool.close()
    pool.join()

    for i, alpha in enumerate(alphas):
        print("Alpha : ", alpha, " ({})".format(i))
        print(results[i][0])
        print(f"Minimum cost: {results[i][1]}")
        print()

        alpha_min_costs.append(results[i][2])
        visualize(depart, destination, mat_adj, alpha_min_costs[i][2])

    # Assuming alpha_min_costs is your list of arrays
    alpha_min_costs = np.array(alpha_min_costs, dtype="str")

    # Save to file :

    np.savetxt("alpha_min_cost.txt", alpha_min_costs, delimiter=",", fmt="%s")


def visualize(depart, destination, mat_adj, bin_str):
    G = nx.Graph()
    for _, value in enumerate(depart):
        G.add_edge(
            depart[value],
            destination[value],
            weight=(mat_adj[depart[value], destination[value]]),
        )

    pos = nx.spring_layout(G)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_nodes(G, pos, node_size=400, node_color="#797EF6")
    nx.draw_networkx_edges(G, pos, width=2, edge_color="#797EF6")
    nx.draw_networkx_labels(
        G, pos, font_size=10, font_family="sans-serif", font_color="w"
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels)

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # Display graph
    # bin_str = list(map(int, max(dist.binary_probabilities(), key=dist.binary_probabilities().get)))  # type: ignore
    # bin_str.reverse()
    # print(bin_str)

    pos = nx.spring_layout(G, seed=7)
    edge_labels = nx.get_edge_attributes(G, "weight")

    e_in = [(u, v) for i, (u, v, d) in enumerate(G.edges(data=True)) if bin_str[i]]
    e_out = [(u, v) for i, (u, v, d) in enumerate(G.edges(data=True)) if not bin_str[i]]

    print(e_in)

    color_map = np.array(["#D3D3D3"] * G.number_of_nodes())
    print(list(sum(e_in, ())))
    color_map[list(sum(e_in, ()))] = "#EE6B6E"

    nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=400)  # type: ignore
    nx.draw_networkx_edges(
        G, pos, edgelist=e_in, width=2, alpha=1, edge_color="#EE6B6E", style="dashed"
    )
    nx.draw_networkx_edges(G, pos, edgelist=e_out, width=2, edge_color="#D3D3D3")
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    nx.draw_networkx_labels(
        G, pos, font_size=10, font_family="sans-serif", font_color="w"
    )

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
