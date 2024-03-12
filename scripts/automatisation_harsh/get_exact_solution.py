import numpy as np


def get_exact_sol(hamiltonian):
    mat_hamiltonian = np.array(hamiltonian.to_matrix())
    eigenvalues, eigenvectors = np.linalg.eig(mat_hamiltonian)

    best_indices = np.where(eigenvalues == np.min(eigenvalues))
    print(eigenvalues[int("0111", 2)])
    print("Eigenvalues : ", eigenvalues[best_indices])
    print("Eigenvectors : ", eigenvectors[best_indices])

    binary_paths = [bin(idx[0]).lstrip("-0b") for idx in best_indices]
    print("Binary paths : ", binary_paths)
    # costs and paths to all best solutions
    return eigenvalues[best_indices], binary_paths
