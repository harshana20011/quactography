import numpy as np
import pandas as pd

# todo: rename to mat_adj.py
# todo: choice a: reads path load , choice b: generates random matrix, load and save to csv
# todo: graph.py: Class Graph: self.num_nodes, self.adjacency_matrix, self.edges, self.weights, self.starting_nodes, self.ending_nodes, self.q_indices, self.all_weights_sum, self.max_weight etc.


# Remove all-zero columns and rows from the adjacency matrix
def remove_zero_columns_rows(mat: np.ndarray):
    """Remove all-zero columns and rows from the adjacency matrix.

    Args:
        mat (np.ndarray): adjacency matrix

    Returns:
        mat (np.ndarray) : adjacency matrix with all-zero columns and rows removed
    """
    zero_cols = np.all(mat == 0, axis=0)  # type: ignore
    zero_rows = np.all(mat == 0, axis=1)
    non_zero_cols = np.where(~zero_cols)[0]
    non_zero_rows = np.where(~zero_rows)[0]
    return mat[np.ix_(non_zero_rows, non_zero_cols)]


# New method to generate random adjacency matrix:
def generate_random_adj_matrix_given_nodes_edges(
    num_nodes: int, num_edges: int, edges_matter: bool
):
    """Generate a random adjacency matrix given number of nodes and edges.

    Args:
        num_nodes (int): number of nodes desired in the graph
        num_edges (int): number of edges desired in the graph
        edges_matter (bool): if True, num_edges is the exact number of edges in the graph, if False, num_edges is the maximum number of edges in the graph

    Returns:
        mat_adj (np.ndarray) : adjacency matrix of the graph
        adj_matrix_from_csv (pd.DataFrame) : adjacency matrix of the graph in pandas DataFrame format
    """
    max_num_edges = ((num_nodes * num_nodes) - num_nodes) / 2
    print(f"max number of edges: {max_num_edges}")

    if edges_matter == True:
        num_nodes = int((1 + np.sqrt(1 + 8 * num_edges)) / 2)
    print(num_nodes)
    print(num_edges)

    if edges_matter == False:
        if num_edges > max_num_edges:
            num_edges = max_num_edges  # type: ignore

    print("num edges wanted", num_edges)
    mat = np.zeros((num_nodes, num_nodes), dtype=float)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i is not j:
                mat[i, j] = np.random.randint(1, 3 + 1)
                mat[j, i] = mat[i, j]

    num_edges_in_mat = 0
    mat_flatten = mat.flatten()
    for i in mat_flatten:
        if i != 0:
            num_edges_in_mat += 1
    num_edges_in_mat = num_edges_in_mat / 2
    num_edges_too_much = num_edges_in_mat - num_edges
    print("num edges to delete:", num_edges_too_much)

    print(mat_flatten)
    mat_triu = np.triu(mat)
    mat_triu_flatten = mat_triu.flatten()
    # print(num_edges_too_much)
    if num_edges_too_much > 0:
        while num_edges_too_much > 0:
            for pos, value in enumerate(mat_triu_flatten):
                if value != 0:
                    first_non_zero_pos_upper_mat = pos
                    break

            print(first_non_zero_pos_upper_mat)
            mat_flatten[first_non_zero_pos_upper_mat] = 0
            mat_triu_flatten[first_non_zero_pos_upper_mat] = 0
            # print(mat_flatten)
            mat = mat_flatten.reshape((num_nodes, num_nodes))

            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i is not j:
                        mat[j, i] = mat[i, j]
            print(num_edges_too_much)
            num_edges_too_much -= 1

    print(num_edges_too_much)
    print(mat_triu_flatten)

    # Remove all-zero columns and rows
    mat = remove_zero_columns_rows(mat)
    # Save adjacency matrix to csv file
    filename = r"scripts\automatisation_harsh\matrices\random_adjacency_matrix.csv"
    save_adjacency_matrix_to_csv(
        mat,
        filename,
    )
    # # Visualize the graph test:
    # graph_cross = rx.PyGraph()
    # nodes_list = graph_cross.add_nodes_from(range(len(mat)))

    # # Add edges
    # edges = []
    # for i in range(mat.shape[0]):
    #     for j in range(mat.shape[1]):
    #         if mat[i, j] != 0:
    #             edges.append((i, j, mat[i, j].round(3)))

    # graph_cross.add_edges_from(edges)

    # # Draw the graph
    # draw(graph_cross, with_labels=True, edge_labels=str)  # type: ignore
    # plt.show()
    return filename


def generate_random_adjacency_matrix(num_nodes, num_zeros_to_add, max_weight=0.5):
    """
    Generates a symetric adjacency matrix with random weights, and adds num_zeros_to_add zeros to the matrix to break the complete graph

    Args:
        num_zeros_to_add (int): Number of zeros to add to the adjacency matrix
        num_nodes (int): Number of nodes in the graph
        max_weight (float): Maximum weight for the edges

    Returns:
        adjacency_matrix (np.ndarray): Adjacency matrix of the graph with first row as number of nodes and matrix starts at second row
    """
    # Matrice vide de taille num_nodes x num_nodes:
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=float)

    # Remplir la matrice avec des poids aléatoires pour les arêtes
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            weight = np.random.uniform(
                0, max_weight
            )  # Poids aléatoire entre 0 et max_weight
            adjacency_matrix[i, j] = weight
            adjacency_matrix[j, i] = weight  # La matrice est symétrique
            # Mettez à zéro au moins la moitié des connexions supplémentaires

    for _ in range(num_zeros_to_add):
        i, j = np.random.randint(0, num_nodes, size=2)
        adjacency_matrix[i, j] = 0
        adjacency_matrix[j, i] = 0

    return adjacency_matrix


def generate_random_adjacency_matrix_from_zeros(num_nodes, probability):
    """Generates a random matrix from all zeros then adds edges based on probability.

    Args:
        num_nodes (int): number of nodes in the graph
        probability (float): probability of adding an edge between two nodes

    Returns:
         adjacency_matrix (np.ndarray): Adjacency matrix of the graph with first row as number of nodes and matrix starts at second row
    """
    # Initialize a null matrix with zeros :
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=float)

    # Fill the matrix with random edges based probability :
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and np.random.rand() < probability:
                # Set an edge between nodes i and j :
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1

    # Assign random weights to existing edges :
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adjacency_matrix[i, j] == 1:
                weight = np.random.uniform(0.01, 0.5)
                adjacency_matrix[i, j] = weight
                adjacency_matrix[j, i] = weight

    return adjacency_matrix


def save_adjacency_matrix_to_csv(adjacency_matrix, filename="adjacency_matrix.csv"):
    """
    Save adjacency matrix to a CSV file.

    Args:
        adjacency_matrix (np.ndarray): adjacency matrix to save
        filename (str): name of the file to save the adjacency matrix to
    """
    adj_matrix_csv = pd.DataFrame(adjacency_matrix)
    adj_matrix_csv.to_csv(filename, index=False)


def save_adjacency_matrix_to_npz(adjacency_matrix, filename="adjacency_matrix.npz"):
    """
    Save adjacency matrix to a npz file.

    Args:
        adjacency_matrix (np.ndarray): adjacency matrix to save
        filename (str): name of the file to save the adjacency matrix to
    """
    np.savez(filename, adjacency_matrix=adjacency_matrix)


# generate_random_adj_matrix_given_nodes_edges(4, 10, True)
