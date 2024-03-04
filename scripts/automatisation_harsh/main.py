import multiprocessing
import pandas as pd
import numpy as np
import itertools

from visualisation_entry_graph import visualize_num_nodes
from connexions_qubits import connexions_edges
from generate_random_matrices import generate_random_adjacency_matrix
from generate_random_matrices import save_adjacency_matrix_to_csv
from hamiltonian import hc, hdep, hfin, hint
from alphas_min_graph import plot_alpha_cost
from visualize_paths_opt import visualize
from qaoa_path import _find_shortest_path_parallel


def main():

    # Code pour matrice d'adjacence déjà existante :
    df = pd.read_csv(r"scripts\automatisation_harsh\matrices\mat_adj2.csv")
    mat_adj = np.array(df)  # type ignore

    # # Code pour une matrice générée aléatoirement :
    # num_nodes = 8
    # random_adj_matrix = generate_random_adjacency_matrix(num_nodes, num_zeros_to_add=29)
    # mat_adj = np.array(random_adj_matrix)
    # save_adjacency_matrix_to_csv(
    #     random_adj_matrix,
    #     filename=r"scripts\automatisation_harsh\matrices\random_adjacency_matrix.csv",
    # )

    # # Lire la matrice :
    # df = pd.read_csv(
    #     r"scripts\automatisation_harsh\matrices\random_adjacency_matrix.csv"
    # )
    # mat_adj = np.array(df)

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
    noeud_de_depart = 4
    hdep1 = hdep(noeud_de_depart, depart, q_indices, destination, number_of_edges)

    # Le terme associé au respect de la contrainte de fin :
    # Déterminer le noeud de fin :
    noeud_de_fin = 1
    hfin1 = hfin(noeud_de_fin, depart, q_indices, destination, number_of_edges)

    hint1 = hint(
        noeud_de_depart, noeud_de_fin, depart, q_indices, destination, number_of_edges
    )

    # Alpha : coût associé aux contraintes:
    alphas = [
        0.0,
        1.0,
        0.5 * all_weights_sum,
        all_weights_sum,
        2 * all_weights_sum,
        3 * all_weights_sum,
        4 * all_weights_sum,
    ]

    alpha_min_costs = []

    # Nombre de processeurs :
    nbr_processes = multiprocessing.cpu_count()
    reps = 4  # utilisateur peut changer la valeur de reps
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
        )
        print(str(alpha_min_costs[i][2]))
    # Assuming alpha_min_costs is your list of arrays
    alpha_min_costs = np.array(alpha_min_costs, dtype="str")

    # Save to file :

    np.savetxt(r"output\alpha_min_cost.txt", alpha_min_costs, delimiter=",", fmt="%s")
    plot_alpha_cost()


if __name__ == "__main__":
    main()
