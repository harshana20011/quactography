import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import Estimator, Sampler


def get_exact_sol(hamiltonian):
    mat_hamiltonian = np.array(hamiltonian.to_matrix())
    eigenvalues, eigenvectors = np.linalg.eig(mat_hamiltonian)

    best_indices = np.where(eigenvalues == np.min(eigenvalues))
    # print(eigenvalues[int("0111", 2)])
    print("Eigenvalues : ", eigenvalues[best_indices])
    print("Eigenvectors : ", eigenvectors[best_indices])

    binary_paths = [bin(idx[0]).lstrip("-0b") for idx in best_indices]
    # print("Binary paths : ", binary_paths)

    # costs and paths to all best solutions
    return eigenvalues[best_indices], binary_paths


def check_hamiltonian_terms(hamiltonian_term, binary_paths_classical_read):
    estimator = Estimator(options={"shots": 1000000, "seed": 42})
    circuit = QuantumCircuit(len(binary_paths_classical_read[0]))
    for i in range(len(binary_paths_classical_read)):
        for j in range(len(binary_paths_classical_read[i])):
            if binary_paths_classical_read[i][j] == "1":
                circuit.x(j)

        print(
            circuit,
            "\n Cost for path (classical read -> left=q0)",
            binary_paths_classical_read[i],
            " : ",
            estimator.run(circuit, hamiltonian_term).result().values[0],
        )
