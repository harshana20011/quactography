from connexions_qubits import connexions_edges
from qiskit.quantum_info import SparsePauliOp
import numpy as np


def hc(mat_adj, num_nodes):
    """_summary_

    Args:
        mat_adj (_type_): _description_
        num_nodes (_type_): _description_
    """
    number_of_edges = connexions_edges(mat_adj, num_nodes)[0]
    weights = connexions_edges(mat_adj, num_nodes)[1]

    all_weights_sum = sum(np.tril(mat_adj).flatten())

    pauli_weight_first_term = []
    # Terme "IIIIII...I":
    pauli_weight_first_term.append(("I" * number_of_edges, all_weights_sum / 2))

    # Z à la bonne position:
    for i in range(number_of_edges):
        str = ("I" * (number_of_edges - i - 1) + "Z" + "I" * i, -weights[0][i] / 2)
        pauli_weight_first_term.append(str)

    h_c = SparsePauliOp.from_list(pauli_weight_first_term)
    print(f"\n Coût obligatoire = {h_c}")
    return h_c
