import rustworkx as rx
import matplotlib.pyplot as plt
from rustworkx.visualization import mpl_draw as draw


def visualize_num_nodes(df, mat_adj):
    """Visualize the graph and determine the number of nodes in the graph.
    Args:
        df (csv):  csv file containing the adjacency matrix
        mat_adj (nparray):  numpy array containing the adjacency matrix
    return:
    num_nodes_cross_graph (int): number of nodes in the graph
    """
    graph_cross = rx.PyGraph()
    num_nodes_cross_graph = len(df)
    nodes_list = graph_cross.add_nodes_from((range(num_nodes_cross_graph)))
    # Add edges :
    edges = []
    for i in range(num_nodes_cross_graph):
        for j in range(num_nodes_cross_graph):
            if mat_adj[i, j] != 0:
                edges.append((i, j, mat_adj[i, j]))

    # print(edges)
    graph_cross.add_edges_from(edges)

    draw(graph_cross, with_labels=True, edge_labels=str)  # type: ignore
    # Save figure in output
    plt.savefig("output/graph_dep.png")
    plt.close()
    return num_nodes_cross_graph
