from qiskit.primitives import Estimator, Sampler
from qiskit.circuit.library import QAOAAnsatz
from scipy.optimize import minimize
import numpy as np


# Fonction à faire en parallèle :
def _find_shortest_path_parallel(args):
    """Summary : Utilisation de multiprocessing pour trouver le chemin optimal en parallèle pour les différentes valeurs de alphas.

    Args:
        args (Sparse Pauli list): Différents termes dans l'hamiltonien.

    Returns:
        liste : résultats de l'optimisation, valeur minimale trouvée pour le coût, alpha et le chemin optimal.
    """
    hc1 = args[0]
    hdep1 = args[1]
    hfin1 = args[2]
    hint1 = args[3]
    alpha = args[4]

    # Fonction coût en représentation QUBO:
    h = -hc1 + alpha * ((hdep1**2) + (hfin1**2) + hint1)

    mat = h.to_matrix()
    u, s, v = np.linalg.svd(mat)

    print("u", u)
    print("s", s)
    print("v", v)

    # Create QAOA circuit.
    ansatz = QAOAAnsatz(h, reps=1)

    # print(ansatz.decompose(reps=1).draw())

    # Run on local estimator and sampler. Fix seeds for results reproducibility.
    estimator = Estimator(options={"shots": 1000000, "seed": 42})
    sampler = Sampler(options={"shots": 1000000, "seed": 42})

    # Cost function for the minimizer.
    # Returns the expectation value of circuit with Hamiltonian as an observable.
    def cost_func(params, estimator, ansatz, hamiltonian):
        cost = (
            estimator.run(ansatz, hamiltonian, parameter_values=params)
            .result()
            .values[0]
        )
        return cost

    # Generate starting point. Fixed to zeros for results reproducibility.
    # x0 = 2 * np.pi * np.random.rand(ansatz.num_parameters)
    x0 = np.zeros(ansatz.num_parameters)

    res = minimize(cost_func, x0, args=(estimator, ansatz, h), method="COBYLA")
    # print(res)

    min_cost = cost_func(res.x, estimator, ansatz, h)

    # print(f"Minimum cost: {min_cost}")

    # Get probability distribution associated with optimized parameters.
    circ = ansatz.copy()
    circ.measure_all()
    dist = sampler.run(circ, res.x).result().quasi_dists[0]

    # plot_distribution(dist.binary_probabilities(), figsize=(7, 5))

    # print(max(dist.binary_probabilities(), key=dist.binary_probabilities().get))  # type: ignore
    bin_str = list(map(int, max(dist.binary_probabilities(), key=dist.binary_probabilities().get)))  # type: ignore
    bin_str.reverse()
    bin_str = np.array(bin_str)

    # Concaténer chaque liste en une seule chaîne de caractères
    str_path = ["".join(map(str, bin_str))]  # type: ignore
    str_path = str_path[0]  # type: ignore

    # Save parameters alpha and min_cost with path in csv file:
    alpha_min_cost = [alpha, min_cost, str_path]

    # print(sorted(dist.binary_probabilities(), key=dist.bina
    # ry_probabilities().get))  # type: ignore
    print("Finished with alpha : ", alpha)

    return (res, min_cost, alpha_min_cost)
