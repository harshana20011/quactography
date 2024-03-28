import numpy as np


def connexions_edges(mat_adj, num_nodes):
    """Determine different connexions between nodes and the number of edges in the graph

    Args:
        mat_adj (nparray): adjacency matrix
        num_nodes (int): number of nodes in the graph
    Returns:
        number_of_edges (int): number of edges in the graph
        weights (list): list of weights
        starting_nodes (list): list of nodes of starting_nodesure
        ending_nodes (list): list of ending_nodes nodes
        q_indices (list): list of indices corresponding to the edges same as the qubit indices
        all_weights_sum (float): sum of all weights in the graph
    """
    mat_triang_sup = np.triu(mat_adj)
    mat_triang_sup = np.array(mat_triang_sup)

    # # Determine different connexions between nodes:
    # all_possible_connexions = []

    # for start in range(num_nodes):
    #     start_node_adj_mat = mat_adj[start]
    #     possible_starts = []
    #     for node, value in enumerate(start_node_adj_mat):
    #         if value > 0:
    #             possible_starts.append(node)
    #     all_possible_connexions.extend([possible_starts])
    #     print(f"Connexions possibles depuis {start} au(x) noeud(s) : {possible_starts}")
    # print(f"Toutes connexions possibles (doublées) : {all_possible_connexions} \n")

    # Determine edges indices and weights:
    list_of_nodes_for_naming_edges = []
    ending_nodes = []
    number_of_edges = 0

    for start in range(num_nodes - 1):
        start_node_adj_mat = mat_triang_sup[start]
        end_edge = []
        for node, value in enumerate(start_node_adj_mat):
            if value > 0:
                end_edge.append(node)
                number_of_edges += 1
                ending_nodes.append(node)
        list_of_nodes_for_naming_edges.extend([end_edge])
        print(f"Edges from node {start} to node : {end_edge}")
    print(
        f"All possible connexions without doubles: {list_of_nodes_for_naming_edges}\n"
    )

    q_indices = []
    starting_nodes = []
    index = 0
    num_nodes_minus_1 = num_nodes - 1
    for i in range(num_nodes_minus_1):
        for _ in list_of_nodes_for_naming_edges[i]:
            starting_nodes.append(i)
            q_indices.append(index)
            index += 1

    print(f"Index :{q_indices}")
    print(f"Start :{starting_nodes}")
    print(f"End   :{ending_nodes}")

    weights = []
    for _ in range(number_of_edges):
        for _ in starting_nodes:
            for _ in ending_nodes:
                weight_qubit = mat_adj[starting_nodes, ending_nodes]
    weights.append(weight_qubit)
    # weights[0][0]
    all_weights_sum = sum(np.tril(mat_adj).flatten())
    max_weight = max(np.tril(mat_adj).flatten())
    return (
        number_of_edges,
        weights,
        starting_nodes,
        ending_nodes,
        q_indices,
        all_weights_sum,
        max_weight,
    )
