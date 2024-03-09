from qiskit.quantum_info import SparsePauliOp


# todo: change name of variables
def hc(number_of_edges, weights, all_weights_sum):
    """Cost of going through a path

    Args:
        number_of_edges (int): Number of edges in the graph
        weights (list int): The weights of the edges
        all_weights_sum (int): Sum of all weights in the graph

    Returns:
        Sparse pauli op (str):  Pauli string representing the cost of going through a path
    """

    pauli_weight_first_term = [("I" * number_of_edges, all_weights_sum / 2)]

    # Z à la bonne position:
    for i in range(number_of_edges):
        str1 = ("I" * (number_of_edges - i - 1) + "Z" + "I" * i, -weights[0][i] / 2)
        pauli_weight_first_term.append(str1)

    h_c = SparsePauliOp.from_list(pauli_weight_first_term)
    print(f"\n Coût obligatoire = {h_c}")
    return h_c


def hdep(noeud_de_depart, depart, q_indices, destination, number_of_edges):
    """Cost term of having a single departure connection (one edge connected to the starting node)

    Args:
        noeud_de_depart (int): Starting node decided by the user
        depart (list int):  List of nodes in departure (according to the adjacency matrix to avoid doublets)
        q_indices (list int): Index associated with each qubit according to the adjacency matrix
        destination (list int):  List of nodes in end (according to the adjacency matrix to avoid doublets)
        number_of_edges (int): Number of edges which is the same as the number of qubits in the graph

    Returns:
        Sparse pauli op (str): Pauli string representing the cost associated with the constraint of having a single departure connection
    """

    qubit_depart = []
    for node, value in enumerate(depart):
        if value == noeud_de_depart:
            qubit_depart.append(q_indices[node])
    for node, value in enumerate(destination):
        if value == noeud_de_depart:
            qubit_depart.append(q_indices[node])
    print(f"\n Qubit à sommer sur les x_i de départ: q({qubit_depart}) - I ")

    pauli_departure_term = [("I" * number_of_edges, len(qubit_depart) * 0.5 - 1)]

    # Z à la bonne position:
    for i, value in enumerate(qubit_depart):
        str2 = ("I" * (number_of_edges - (value + 1)) + "Z" + "I" * value, -0.5)
        pauli_departure_term.append(str2)
    h_dep = SparsePauliOp.from_list(pauli_departure_term)

    print(f"\n Contrainte de départ = {h_dep}")
    return h_dep


def hfin(noeud_de_fin, depart, q_indices, destination, number_of_edges):
    """Cost term of having a single end connection (one edge connected to the ending node)

    Args:
        noeud_de_fin (int): Ending node decided by the user
        depart (list int): List of nodes in departure (according to the adjacency matrix to avoid doublets)
        q_indices (list int): Index associated with each qubit according to the adjacency matrix
        destination (list int): List of nodes in end (according to the adjacency matrix to avoid doublets)
        number_of_edges (int): Number of edges which is the same as the number of qubits in the graph

    Returns:
       Sparse pauli op (str): Pauli string representing the cost associated with the constraint of having a single end connection
    """
    qubit_end = []
    for node, value in enumerate(destination):
        if value == noeud_de_fin:
            qubit_end.append(q_indices[node])
    for node, value in enumerate(depart):
        if value == noeud_de_fin:
            qubit_end.append(q_indices[node])
    print(f"\n Qubit à sommer sur les x_i de fin: q({qubit_end}) - I ")

    pauli_end_term = [("I" * number_of_edges, len(qubit_end) * 0.5 - 1)]

    # Z à la bonne position:
    for i, value in enumerate(qubit_end):
        str2 = ("I" * (number_of_edges - (value + 1)) + "Z" + "I" * value, -0.5)
        pauli_end_term.append(str2)
    h_fin = SparsePauliOp.from_list(pauli_end_term)

    print(f"\n Contrainte de fin = {h_fin}")
    return h_fin


def hint(
    noeud_de_depart, noeud_de_fin, depart, q_indices, destination, number_of_edges
):
    """Cost term of having an even number of intermediate connections (two edges connected to the intermediate nodes)

    Args:
        noeud_de_depart (int):  Starting node decided by the user
        noeud_de_fin (int): Ending node decided by the user
        depart (list int): List of nodes in departure (according to the adjacency matrix to avoid doublets)
        q_indices (list int): Index associated with each qubit according to the adjacency matrix
        destination (list int): List of nodes in end (according to the adjacency matrix to avoid doublets)
        number_of_edges (int): Number of edges which is the same as the number of qubits in the graph

    Returns:
        Sparse pauli op (str): Pauli string representing the cost associated with the constraint of having an even number of intermediate connections
    """
    # Intermediate connections, constraints:
    noeuds_int = []
    for node, value in enumerate(depart):
        if (value != noeud_de_depart) and (value != noeud_de_fin):
            if value not in noeuds_int:
                noeuds_int.append(depart[node])
    for node, value in enumerate(destination):
        if (value != noeud_de_depart) and (value != noeud_de_fin):
            if value not in noeuds_int:
                noeuds_int.append(destination[node])

    print(f"Liste de noeuds intermédiaires: {noeuds_int} \n")

    liste_qubits_int = [[] for _ in range(len(noeuds_int))]
    for i, node in enumerate(noeuds_int):
        for node, value in enumerate(destination):
            if value == noeuds_int[i]:
                liste_qubits_int[i].append(q_indices[node])
        for node, value in enumerate(depart):
            if value == noeuds_int[i]:
                liste_qubits_int[i].append(q_indices[node])

    for i in range(len(liste_qubits_int)):
        a = liste_qubits_int[i]
        print(f"Qubit à multiplier sur les x_i intermédiaires: q({a}) ")

    hint = []
    prod_terms = []
    for list_q in liste_qubits_int:
        prod_term = "I" * number_of_edges
        for qubit in list_q:
            prod_term = prod_term[:qubit] + "Z" + prod_term[qubit + 1 :]
        prod_terms.append(prod_term[::-1])

    for i in range(len(liste_qubits_int)):
        hint.append([])

        hint[i] = SparsePauliOp.from_list(
            [("I" * number_of_edges, -1.0), (prod_terms[i], 1.0)]
        )

    print(f"\n Contrainte intermédiaire = {hint}")

    hints = []
    for i in range(len(hint)):
        hints.append(hint[i] ** 2)
    print(f"Somme des termes intermédiaires au carré: {sum(hints)}")
    return sum(hints)
