import numpy as np
import pandas as pd


def remove_orphan_nodes(graph, node_indices, keep_indices=None):
    out_graph = []
    out_it = []
    for it, graph_row in enumerate(graph):
        if np.count_nonzero(graph_row) > 0 or not _test_removable_indice(
            it, node_indices, keep_indices
        ):
            out_graph.append(graph_row)
            out_it.append(it)
    out_graph = np.take(np.asarray(out_graph), out_it, axis=1)
    out_indices = node_indices[np.asarray(out_it)]

    return out_graph, out_indices


def remove_intermediate_connections(graph, node_indices=None, keep_indices=None):
    skipped_at_least_one = True
    while skipped_at_least_one:
        skipped_at_least_one = False
        for it, graph_row in enumerate(graph):
            if np.count_nonzero(graph_row) == 2 and _test_removable_indice(
                it, node_indices, keep_indices
            ):
                indices = np.flatnonzero(graph_row)
                graph[indices[0], indices[1]] = np.sum(graph_row)
                graph[indices[1], indices[0]] = np.sum(graph_row)
                graph[it, :] = 0.0
                graph[:, it] = 0.0
                if indices[0] < it and indices[1] < it:
                    skipped_at_least_one = True
    return graph


def _test_removable_indice(it, node_indices, keep_indices):
    if keep_indices is None or node_indices is None:
        return True
    return not (keep_indices == node_indices[it]).any()


def choose_region_m_nodes(graph, m, node_indices, keep_indices):
    # Il faut mettre à jour les indices ...
    # La première colonne est inutile ...
    graph = graph[0:m]
    df = pd.DataFrame(graph)
    df.to_csv("mat_adj.csv")
    df = pd.read_csv("mat_adj.csv")
    nb_nodes = m

    list_node = []
    for i in range(nb_nodes):
        list_node.append(str(i))
    df = df[list_node]
    df = df.iloc[list_node]

    df.to_csv("mat_adj.csv", index=False, header=True)

    return graph


def choose_region_nodes(graph, start, end, node_indices, keep_indices):

    graph = graph[start : end + 1, start : end + 1]
    print(graph)
    df = pd.DataFrame(graph)
    df.to_csv("mat_adj_selection.csv", index=False, header=True)
    return graph
