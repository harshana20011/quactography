import multiprocessing
import pandas as pd
import numpy as np
import itertools
from numpy import load

from visualisation_entry_graph import visualize_num_nodes
from connexions_qubits import connexions_edges
from generate_random_matrices import generate_random_adjacency_matrix
from generate_random_matrices import save_adjacency_matrix_to_csv
from hamiltonian import hc, hdep, hfin, hint
from alphas_min_graph import plot_alpha_cost
from visualize_paths_opt import visualize
from qaoa_path import _find_shortest_path_parallel


def main():
    """This script is the main entry point for the automatisation of the QAOA algorithm:
    It...
    - reads an adjacency matrix from a csv or npz file or creates one
    - determines the possible connexions between nodes and the number of edges
    - calculates the cost of the Hamiltonian
    - calculates the cost of the constraints
    - calculates the minimum cost for different values of alpha
    - visualizes the paths and the minimum cost for different values of alpha
    - saves the minimum cost for different values of alpha to a file
    - plots the minimum cost for different values of alpha
    """
    # First method: visualize a graph from a csv file, existing in matrices automatisation_harsh/folder:
    df = pd.read_csv(r"scripts\automatisation_harsh\matrices\mat_adj2.csv")
    mat_adj = np.array(df)

    # # Second method: generate a random adjacency matrix
    # # and save it to a csv file in automatisation_harsh/matrices
    # # First choose the number of nodes you wish to have in the graph:
    # num_nodes = 4
    # random_adj_matrix = generate_random_adjacency_matrix(num_nodes, num_zeros_to_add=3)
    # mat_adj = np.array(random_adj_matrix)
    # save_adjacency_matrix_to_csv(
    #     random_adj_matrix,
    #     filename=r"scripts\automatisation_harsh\matrices\random_adjacency_matrix.csv",
    # )
    # # Read matrix from csv file and convert to numpy array:
    # df = pd.read_csv(
    #     r"scripts\automatisation_harsh\matrices\random_adjacency_matrix.csv"
    # )
    # mat_adj = np.array(df)

    # # Third method: visualize a graph from a npz file
    # # Open and read data from npz file: (fibercup data output from build_graph.py)
    # # todo: from build_graph.py, save npz file to matrices file in automatisation_harsh
    # matrix_from_npz = load(r"scripts\automatisation_harsh\matrices\outgraph.npz")
    # adjmat, node_indices, vol_dims = matrix_from_npz.files
    # print(adjmat, "\n", matrix_from_npz[adjmat])
    # print(node_indices, "\n", matrix_from_npz[node_indices])
    # print(vol_dims, "\n", matrix_from_npz[vol_dims])

    # Visualisation and determination of the number of nodes in the graph:
    num_nodes = visualize_num_nodes(df, mat_adj)

    # Determination of different connexions between nodes and the number of edges in the graph:
    number_of_edges, weights, depart, destination, q_indices, all_weights_sum = (
        connexions_edges(mat_adj, num_nodes)
    )

    # Calculation the cost of the first term in the Hamiltonian
    # which makes sure there is only one edge connected to the starting node:
    hc1 = hc(number_of_edges, weights, all_weights_sum)

    # Fix a starting node:
    noeud_de_depart = 2
    # Calculates the cost of the first term in the Hamiltonian
    # which when equal zero, makes sure there is only one edge connected to the starting node:
    hdep1 = hdep(noeud_de_depart, depart, q_indices, destination, number_of_edges)

    # Fix an ending node:
    noeud_de_fin = 0
    # Calculates the cost of the last term in the Hamiltonian
    # which when equal zero, makes sure there is only one edge connected to the ending node:
    hfin1 = hfin(noeud_de_fin, depart, q_indices, destination, number_of_edges)

    # Calculates the cost of the intermediate nodes in the Hamiltonian
    # which when equal zero, makes sure there are only two edges connected to the intermediate nodes:
    hint1 = hint(
        noeud_de_depart, noeud_de_fin, depart, q_indices, destination, number_of_edges
    )

    # Different values of alpha where alpha is a coefficient
    # that amplifies the cost of breaking constraints in the Hamiltonian:
    alphas = [
        0.0,
        0.5 * all_weights_sum,
        all_weights_sum,
        1.2 * all_weights_sum,
        1.5 * all_weights_sum,
    ]

    alpha_min_costs = []

    # Number of processes to use for parallel processing:
    nbr_processes = multiprocessing.cpu_count()
    # Number of repetitions for the QAOA algorithm (equal to number of
    # layers in the quantum circuit HC, HB with different parameters gamma and beta):
    reps = 5
    pool = multiprocessing.Pool(nbr_processes)
    results = pool.map(
        _find_shortest_path_parallel,
        zip(
            itertools.repeat(hc1),
            itertools.repeat(hdep1),
            itertools.repeat(hfin1),
            itertools.repeat(hint1),
            alphas,
            itertools.repeat(reps),
        ),
    )
    pool.close()
    pool.join()

    # Loop through the results and print the minimum cost for each value of alpha:
    for i, alpha in enumerate(alphas):
        print("Alpha : ", alpha, " ({})".format(i))
        print(results[i][0])
        print(f"Minimum cost: {results[i][1]}")
        print()

        alpha_min_costs.append(results[i][2])
        visualize(
            depart,
            destination,
            mat_adj,
            list(map(int, (alpha_min_costs[i][2]))),
            alpha,
            results[i][1],
            noeud_de_depart,
            noeud_de_fin,
            reps,
            all_weights_sum,
        )
        print(str(alpha_min_costs[i][2]))

    # Save the minimum cost for different values of alpha to a file with the corresponding binary path in txt file:
    alpha_min_costs = np.array(alpha_min_costs, dtype="str")
    np.savetxt(r"output\alpha_min_cost.txt", alpha_min_costs, delimiter=",", fmt="%s")

    # Save a plot of the minimum cost for different values of alpha:
    plot_alpha_cost()


if __name__ == "__main__":
    main()
