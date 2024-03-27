from qiskit.primitives import Estimator, Sampler
from qiskit.circuit.library import QAOAAnsatz
from qiskit.visualization import plot_distribution
from qiskit.visualization import plot_circuit_layout
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np
import matplotlib

matplotlib.use("Agg")
from get_exact_solution import get_exact_sol
from get_exact_solution import check_hamiltonian_terms


# Function to find the shortest path in a graph using QAOA algorithm with parallel processing:
def _find_shortest_path_parallel(args):
    """Summary :  Usage of QAOA algorithm to find the shortest path in a graph.

    Args:
        args (Sparse Pauli list):  Hamiltonian in QUBO representation

    Returns:
        res (minimize):  Results of the minimization
        min_cost (float):  Minimum cost
        alpha_min_cost (list):  List of alpha, minimum cost and binary path
    """
    hc1 = args[0]
    hdep1 = args[1]
    hfin1 = args[2]
    hint1 = args[3]
    alpha = args[4]
    reps = args[5]
    min_weights = args[6]
    # Cost function for the minimizer:
    h = -hc1 + alpha * ((hdep1**2) + (hfin1**2) + hint1)

    # Eigendecomposition of the Hamiltonian matrix with optimal solution:
    _, path_hamiltonian = get_exact_sol(h)
    # Pad with zeros to the left to have the same length as the number of edges:
    for i in range(len(path_hamiltonian)):
        path_hamiltonian[i] = path_hamiltonian[i].zfill(len(hdep1) + 1)
    print("Path Hamiltonian (quantum reading -> right=q0) : ", path_hamiltonian)
    # Reverse the binary path to have the same orientation as the classical path:
    path_hamiltonian_classical_read = [path[::-1] for path in path_hamiltonian]
    # Create QAOA circuit.
    ansatz = QAOAAnsatz(h, reps, name="QAOA")

    # Plot the circuit layout:
    ansatz.decompose(reps=3).draw(output="mpl", style="iqp")
    plt.savefig("output/qaoa_circuit.png")

    # Check if the Hamiltonian terms are correct with custom circuit:
    check_hamiltonian_terms(
        hamiltonian_term=h, binary_paths_classical_read=path_hamiltonian_classical_read
    )

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
    # todo: check maxiter parameter to avoid maximum number of function evaluations exceeded (default = 1000)
    res = minimize(
        cost_func,
        x0,
        args=(estimator, ansatz, h),
        method="COBYLA",
        options={"maxiter": 5000, "disp": True},
        # tol=0.1 * min_weights,
        tol=1e-4,
    )

    # print(res)

    min_cost = cost_func(res.x, estimator, ansatz, h)
    # print(f"Minimum cost: {min_cost}")

    # Get probability distribution associated with optimized parameters.
    circ = ansatz.copy()
    circ.measure_all()
    dist = sampler.run(circ, res.x).result().quasi_dists[0]
    # Plot distribution of probabilities:
    plot_distribution(
        dist.binary_probabilities(),
        figsize=(10, 8),
        title="Distribution of probabilities",
        color="pink",
    )
    # Save plot:
    plt.savefig(f"output/distribution_alpha_{alpha:.2f}.png")

    # print(max(dist.binary_probabilities(), key=dist.binary_probabilities().get))  # type: ignore
    bin_str = list(map(int, max(dist.binary_probabilities(), key=dist.binary_probabilities().get)))  # type: ignore
    bin_str_reversed = bin_str[::-1]
    bin_str_reversed = np.array(bin_str_reversed)  # type: ignore

    # Check if optimal path in a subset of most probable paths:
    sorted_list_of_mostprobable_paths = sorted(dist.binary_probabilities(), key=dist.binary_probabilities().get, reverse=True)  # type: ignore

    # Dictionary keys and values where key = binary path, value = probability:
    # Find maximal probability in all values of the dictionary:
    max_probability = max(dist.binary_probabilities().values())
    selected_paths = []
    for path, probability in dist.binary_probabilities().items():

        probability = probability / max_probability
        dist.binary_probabilities()[path] = probability
        print(
            f"Path (quantum read -> right=q0): {path} with ratio proba/max_proba : {probability}"
        )

        percentage = 0.5
        # Select paths with probability higher than percentage of the maximal probability:
        if probability > percentage:
            selected_paths.append(path)

    print("_______________________________________________________________________\n")
    print(
        f"Selected paths among {percentage*100} % of solutions (right=q0): {selected_paths}"
    )

    print(
        f"Optimal path obtained by diagonal hamiltonian minimum costs (right=q0): {path_hamiltonian}"
    )

    match_found = False
    for i in selected_paths:
        if i in path_hamiltonian:
            match_found = True
            break
    if match_found:
        print(
            "The optimal solution is in the subset of solutions found by QAOA.\n_______________________________________________________________________"
        )
    else:
        print(
            "The solution is not in given subset of solutions found by QAOA.\n_______________________________________________________________________"
        )

    # Concatenate the binary path to a string:
    str_path_reversed = ["".join(map(str, bin_str_reversed))]  # type: ignore
    str_path_reversed = str_path_reversed[0]  # type: ignore

    # Save parameters alpha and min_cost with path in csv file:
    alpha_min_cost = [alpha, min_cost, str_path_reversed]

    # print(sorted(dist.binary_probabilities(), key=dist.bina
    # ry_probabilities().get))  # type: ignore
    print("Finished with alpha : ", alpha)

    return res, min_cost, alpha_min_cost, selected_paths
