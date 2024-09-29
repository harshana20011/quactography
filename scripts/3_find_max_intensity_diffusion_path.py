import argparse
import sys

sys.path.append(r"C:\Users\harsh\quactography")

from quactography.graph.undirected_graph import Graph
from quactography.adj_matrix.io import load_graph

from quactography.hamiltonian.hamiltonian_qubit_edge import Hamiltonian_qubit_edge
from quactography.hamiltonian.hamiltonian_qubit_node import Hamiltonian_qubit_node
from quactography.solver.qaoa_multiprocess_solver import multiprocess_qaoa_solver


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument(
        "in_graph",
        help="Adjacency matrix which graph we want path that maximizes weights in graph, npz file",
        type=str,
    )
    p.add_argument("starting_node", help="Starting node of the graph", type=int)
    p.add_argument("ending_node", help="Ending node of the graph", type=int)
    p.add_argument("output_file", help="Output file name", type=str)

    p.add_argument(
        "--hamiltonian",
        help="Hamiltonian qubit representation to use for QAOA, either 'node' or 'edge' ",
        default="node",
        choices=["node", "edge"],
        type=str,
    )
    p.add_argument(
        "--alphas", nargs="+", type=int, help="List of alphas", default=[1.1]
    )

    p.add_argument(
        "--reps",
        help="Number of repetitions for the QAOA algorithm",
        type=int,
        default=1,
    )
    # Voir avec scilpy :
    p.add_argument(
        "-npr",
        "--number_processors",
        help="number of cpu to use for multiprocessing",
        default=1,
        type=int,
    )

    return p


def main():
    """
    Uses QAOA with multiprocess as an option to find shortest path, with a given Graph, starting, ending node and Hamiltonian associated
    to the graph.
    """
    parser = _build_arg_parser()
    args = parser.parse_args()

    weighted_graph, _, _ = load_graph(args.in_graph + ".npz")

    graph = Graph(weighted_graph, args.starting_node, args.ending_node)
    if args.hamiltonian == "edge":
        hamiltonians = [Hamiltonian_qubit_edge(graph, alpha) for alpha in args.alphas]
        print("Calculating qubits as edges")
    else:
        hamiltonians = [Hamiltonian_qubit_node(graph, alpha) for alpha in args.alphas]
        print("Calculating qubits as nodes")

    multiprocess_qaoa_solver(
        hamiltonians, args.reps, args.number_processors, args.output_file
    )


if __name__ == "__main__":
    main()