from qiskit.quantum_info import SparsePauliOp


def hc(number_of_edges, weights, all_weights_sum):
    """_summary_

    Args:
        number_of_edges (int): Nombre de qubits soit le nombre de edges dans le graphe
        weights (list int): le poids associé à chaque qubit (edge) du graphe
        all_weights_sum (int): Le poids total sommé de tous les edges

    Returns:
        Sparse pauli op (str): Chaîne de Pauli représentant le coût obligatoire pour passer par un chemin
    """

    pauli_weight_first_term = [("I" * number_of_edges, all_weights_sum / 2)]
    # Terme "IIIIII...I":

    # Z à la bonne position:
    for i in range(number_of_edges):
        str1 = ("I" * (number_of_edges - i - 1) + "Z" + "I" * i, -weights[0][i] / 2)
        pauli_weight_first_term.append(str1)

    h_c = SparsePauliOp.from_list(pauli_weight_first_term)
    print(f"\n Coût obligatoire = {h_c}")
    return h_c


def hdep(noeud_de_depart, depart, q_indices, destination, number_of_edges):
    """_summary_

    Args:
        noeud_de_depart (int): entrée décicé par l'utilisateur : noeud de départ
        depart (list int): Liste de noeuds en départ (en fonction de la matrice d'adjacence pour
         éviter les doublements)
        q_indices (list int): Indice associé à chaque qubit en fonction de la matrice d'adjacence
        destination (list int):  Liste de noeuds en fin (en fonction de la matrice d'adjacence pour
        éviter les doublements)
        number_of_edges (int): nombre de edges qui est le même que le nombre de qubits dans le graphe

    Returns:
        Sparse pauli op (str): Chaîne de Pauli représentant le coût associé à la contrainte
        d'avoir une seule connexion de départ
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
    """_summary_

    Args:
        noeud_de_fin (int):fin décicé par l'utilisateur : noeud de fin
        depart (list int): Liste de noeuds en départ (en fonction de la matrice
        d'adjacence pour éviter les doublements)
        q_indices (list int): Indice associé à chaque qubit en
        fonction de la matrice d'adjacence
        destination (list int):  Liste de noeuds en fin (en fonction de la

        matrice d'adjacence pour éviter les doublements)
        number_of_edges (int): nombre de edges qui est le même que le nombre de qubits dans le graphe

    Returns:
       Sparse pauli op (str): Chaîne de Pauli représentant le coût associé
       à la contrainte d'avoir une seule connexion de fin
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
    """Générer tous les termes intermédiaires et les sommer

    Args:
        noeud_de_depart (int): entrée décicé par l'utilisateur : noeud de départ
        noeud_de_fin (int):fin décicé par l'utilisateur : noeud de fin
        depart (list int): Liste de noeuds en départ (en fonction de la matrice d'adjacence pour éviter les doublements)
        q_indices (list int): Indice associé à chaque qubit en fonction de la matrice d'adjacence
        destination (list int):  Liste de noeuds en fin (en fonction de la
        matrice d'adjacence pour éviter les doublements)
        number_of_edges (int): nombre de edges qui est le même que le nombre de qubits dans le graphe

    Returns:
        Somme sur les contraintes intermédiaires de chaque noeud au carré (somme finale)
    """
    # Contrainte intermédiaire:
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
