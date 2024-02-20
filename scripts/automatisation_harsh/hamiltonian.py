from connexions_qubits import connexions_edges
from qiskit.quantum_info import SparsePauliOp
import numpy as np


def hc(number_of_edges, weights, all_weights_sum):
    """_summary_

    Args:
        number_of_edges (int): Nombre de qubits soit le nombre de edges dans le graphe
        weights (list int): le poids associé à chaque qubit (edge) du graphe
        all_weights_sum (int): Le poids total sommé de tous les edges

    Returns:
        Sparse pauli op (str): Chaîne de Pauli représentant le coût obligatoire pour passer par un chemin
    """

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


def hdep(noeud_de_depart, depart, q_indices, destination, number_of_edges):
    """_summary_

    Args:
        noeud_de_depart (int): entrée décicé par l'utilisateur : noeud de départ
        depart (list int): Liste de noeuds en départ (en fonction de la matrice d'adjacence pour éviter les doublements)
        q_indices (list int): Indice associé à chaque qubit en fonction de la matrice d'adjacence
        destination (list int):  Liste de noeuds en fin (en fonction de la matrice d'adjacence pour éviter les doublements)
        number_of_edges (int): nombre de edges qui est le même que le nombre de qubits dans le graphe

    Returns:
        Sparse pauli op (str): Chaîne de Pauli représentant le coût associé à la contrainte d'avoir une seule connexion de départ
    """

    qubit_depart = []
    for node, value in enumerate(depart):
        if value == noeud_de_depart:
            qubit_depart.append(q_indices[node])
    for node, value in enumerate(destination):
        if value == noeud_de_depart:
            qubit_depart.append(q_indices[node])
    print(f"\n Qubit à sommer sur les x_i de départ: q({qubit_depart}) - I ")

    pauli_departure_term = []
    # Terme "IIIIII...I":
    pauli_departure_term.append(("I" * number_of_edges, len(qubit_depart) * 0.5 - 1))

    # Z à la bonne position:
    for i, value in enumerate(qubit_depart):
        str2 = ("I" * (number_of_edges - (value + 1)) + "Z" + "I" * value, -0.5)
        pauli_departure_term.append(str2)
    h_dep = SparsePauliOp.from_list(pauli_departure_term)

    print(f"\n Contrainte de départ = {h_dep}")
    return h_dep


def hfin(noeud_de_fin, depart, q_indices, destination, number_of_edges):
    qubit_end = []
    for node, value in enumerate(destination):
        if value == noeud_de_fin:
            qubit_end.append(q_indices[node])
    for node, value in enumerate(depart):
        if value == noeud_de_fin:
            qubit_end.append(q_indices[node])
    print(f"\n Qubit à sommer sur les x_i de fin: q({qubit_end}) - I ")

    pauli_end_term = []
    # Terme "IIIIII...I":
    pauli_end_term.append(("I" * number_of_edges, len(qubit_end) * 0.5 - 1))

    # Z à la bonne position:
    for i, value in enumerate(qubit_end):
        str2 = ("I" * (number_of_edges - (value + 1)) + "Z" + "I" * value, -0.5)
        pauli_end_term.append(str2)
    h_fin = SparsePauliOp.from_list(pauli_end_term)

    print(f"\n Contrainte de fin = {h_fin}")
    return h_fin
