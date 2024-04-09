import argparse

from get_adj_matrix import get_adj_matrix
from get_adj_matrix import get_random_adj_matrix_given_nodes_edges
from connexions_qubits import connexions_edges
from visualisation_entry_graph import visualize_num_nodes


class Graph:
    def __init__(self, mat_adj, adj_matrix_from_csv):
        self.num_nodes = visualize_num_nodes(adj_matrix_from_csv, mat_adj)
        (
            self.number_of_edges,
            self.weights,
            self.starting_nodes,
            self.ending_nodes,
            self.q_indices,
            self.all_weights_sum,
            self.max_weight,
        ) = connexions_edges(mat_adj, self.num_nodes)


def main():

    parser = argparse.ArgumentParser()

    # Optional arguments:
    parser.add_argument(
        "-v", "--verbose", help="increase output verbosity", action="store_true"
    )  # pour éviter de devoir écrire un int après --verbose : on met une action "store_true "
    parser.add_argument("-s", "--starting_node", help="start node", type=int, default=0)
    parser.add_argument("-t", "--target_node", help="end node", type=int, default=1)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", "--filename", type=str, help="filename for adj matrix")
    group.add_argument(
        "-n",
        "--number_nodes_edges",
        help="number of nodes and edges and importance of edges (1 if edges matter, 0 otherwise)",
        type=int,
        nargs="+",
        default=[4, 4, 1],
    )
    parser.add_argument("-r", "--reps", type=int, required=True, help="")
    # Add a list:
    parser.add_argument(
        "-a", "--alphas", nargs="+", type=int, help="List of alphas", default=[1.1]
    )

    # parse_args method : returns data from options
    args = parser.parse_args()

    print(args.alphas)
    print(type(args.alphas))
    if args.verbose:
        print("verbose")
    else:
        print("nothing")
    if args.filename:
        adj_matrix_from_csv, mat_adj = get_adj_matrix(args.filename)
    else:
        adj_matrix_from_csv, mat_adj = get_random_adj_matrix_given_nodes_edges(
            args.number_nodes_edges[0],
            args.number_nodes_edges[1],
            args.number_nodes_edges[2],
        )
    Graph(adj_matrix_from_csv, mat_adj)


if __name__ == "__main__":
    main()
