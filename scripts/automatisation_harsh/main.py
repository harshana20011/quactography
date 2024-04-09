import multiprocessing
import pandas as pd
import numpy as np
import itertools
from numpy import load

from visualisation_entry_graph import visualize_num_nodes
from connexions_qubits import connexions_edges
from generate_random_matrices import generate_random_adjacency_matrix
from generate_random_matrices import generate_random_adjacency_matrix_from_zeros
from generate_random_matrices import save_adjacency_matrix_to_csv
from hamiltonian import (
    mandatory_cost,
    starting_node_cost,
    ending_node_cost,
    intermediate_cost_h_term,
)
from alphas_min_graph import plot_alpha_cost
from visualize_paths_opt import visualize
from qaoa_path import _find_shortest_path_parallel


# todo: add args to main: start node/end node, nb reps, adj matrix file or random generation (edges + nodes params), alpha values (list), verbose (bool), parallel (bool)
# todo: add argparse to main
# todo: Output: paths, costs, alpha values, plot graph general, plot optimal path, hist of all paths and hist of 50% best paths,
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
    adj_matrix_from_csv = pd.read_csv(
        r"scripts\automatisation_harsh\matrices\mat_adj .csv"
    )
    mat_adj = np.array(adj_matrix_from_csv)

    # # Second method: generate a random adjacency matrix
    # # and save it to a csv file in automatisation_harsh/matrices
    # # First choose the number of nodes you wish to have in the graph:
    # num_nodes = 4
    # probability_of_edge = 0.6
    # random_adj_matrix = generate_random_adjacency_matrix_from_zeros(
    #     num_nodes, probability=probability_of_edge
    # )
    # mat_adj = np.array(random_adj_matrix)
    # save_adjacency_matrix_to_csv(
    #     random_adj_matrix,
    #     filename=r"scripts\automatisation_harsh\matrices\random_adjacency_matrix_f_z.csv",
    # )
    # # Read matrix from csv file and convert to numpy array:
    # adj_matrix_from_csv = pd.read_csv(
    #     r"scripts\automatisation_harsh\matrices\random_adjacency_matrix_f_z.csv"
    # )
    # # When using existing csv file for random matrix, convert to numpy array:
    # mat_adj = np.array(adj_matrix_from_csv)

    # # Third method: visualize a graph from a npz file
    # # Open and read data from npz file: (fibercup data output from build_graph.py)
    # # todo: from build_graph.py, save npz file to matrices file in automatisation_harsh
    # matrix_from_npz = load(r"scripts\automatisation_harsh\matrices\outgraph.npz")
    # adjmat, node_indices, vol_dims = matrix_from_npz.files
    # print(adjmat, "\n", matrix_from_npz[adjmat])
    # print(node_indices, "\n", matrix_from_npz[node_indices])
    # print(vol_dims, "\n", matrix_from_npz[vol_dims])

    # Visualisation and determination of the number of nodes in the graph:
    num_nodes = visualize_num_nodes(adj_matrix_from_csv, mat_adj)

    # Determination of different connexions between nodes and the number of edges in the graph:
    (
        number_of_edges,
        weights,
        starting_nodes,
        ending_nodes,
        q_indices,
        all_weights_sum,
        max_weight,
    ) = connexions_edges(mat_adj, num_nodes)

    # Calculation the cost of the first term in the Hamiltonian
    # which is the mandatory cost for taking a given path:
    mandatory_cost_hamiltonian = mandatory_cost(
        number_of_edges, weights, all_weights_sum
    )

    # Fix a starting node:
    starting_node = 15
    # Calculates the cost of the second term in the Hamiltonian
    # which when equal zero, makes sure there is only one edge connected to the starting node:
    starting_node_constraint_hamiltonian = starting_node_cost(
        starting_node,
        starting_nodes,
        q_indices,
        ending_nodes,
        number_of_edges,
    )

    # Fix an ending node:
    ending_node = 0
    # Calculates the cost of the third term in the Hamiltonian
    # which when equal zero, makes sure there is only one edge connected to the ending node:
    ending_node_constraint_hamiltonian = ending_node_cost(
        ending_node, starting_nodes, q_indices, ending_nodes, number_of_edges
    )

    # Calculates the cost of the intermediate nodes in the Hamiltonian (last term)
    # which when equal zero, makes sure there are only two edges connected to the intermediate nodes:
    intermediate_nodes_constraint_hamiltonian = intermediate_cost_h_term(
        starting_node,
        ending_node,
        starting_nodes,
        q_indices,
        ending_nodes,
        number_of_edges,
    )

    # Different values of alpha where alpha is a coefficient
    # that amplifies the cost of breaking constraints in the Hamiltonian:
    alphas = [
        # 0.0,
        # 0.50 * max_weight,
        # 1.00 * max_weight,
        # 3.0 * all_weights_sum,
        3 * all_weights_sum,
        # 3.2 * max_weight,
        # 3.32 * max_weight,
        # 3.39 * max_weight,
        # 3.3 * all_weights_sum,
        # 0.1 * all_weights_sum,
        # 3.6 * max_weight,
        # 3.7 * all_weights_sum,
        # 4.0 * all_weights_sum,
        # 3.85 * all_weights_sum,
        # 4.0 * all_weights_sum,
        # 4.1 * all_weights_sum,
        # 4.5 * all_weights_sum,
        # 5.0 * all_weights_sum,
        # 4.2 * all_weights_sum,
        # 20 * all_weights_sum,
        # 1.50 * all_weights_sum,
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
            itertools.repeat(mandatory_cost_hamiltonian),
            itertools.repeat(starting_node_constraint_hamiltonian),
            itertools.repeat(ending_node_constraint_hamiltonian),
            itertools.repeat(intermediate_nodes_constraint_hamiltonian),
            alphas,
            itertools.repeat(reps),
            itertools.repeat([min(weights[i]) for i in range(len(weights))][0]),
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
            starting_nodes,
            ending_nodes,
            mat_adj,
            list(map(int, (alpha_min_costs[i][2]))),
            alpha,
            results[i][1],
            starting_node,
            ending_node,
            reps,
            all_weights_sum,
        )
        print(str(alpha_min_costs[i][2]))

    # Save the minimum cost for different values of alpha to a file with the corresponding binary path in txt file:
    alpha_min_costs = np.array(alpha_min_costs, dtype="str")
    np.savetxt(
        r"output\alpha_min_cost_classical_read_leftq0.txt",
        alpha_min_costs,
        delimiter=",",
        fmt="%s",
    )

    # Save a plot of the minimum cost for different values of alpha:
    # plot_alpha_cost()
    print("------------------------PROCESS FINISHED-------------------------------")


if __name__ == "__main__":
    main()
