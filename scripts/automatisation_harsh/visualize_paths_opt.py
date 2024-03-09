import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import rustworkx as rx

from rustworkx.visualization import mpl_draw


# todo: change display of paths taken (not put to zero edge we don't take)
def visualize(
    depart,
    destination,
    mat_adj,
    bin_str,
    alpha,
    min_cost,
    noeud_de_depart,
    noeud_de_fin,
    reps,
    all_weights_sum,
):
    """_summary_ :Visualize the path taken in the graph and save the figure in the output folder

    Args:
        depart (list int): list of starting points
        destination (list int): list of destination points
        mat_adj (np array):  adjacency matrix
        list(map(int, bin_str)) (liste int): list of 0 and 1 representing the path taken
    """
    bin_str = list(map(int, bin_str))
    bin_str.reverse()

    # Create a graph
    G = nx.Graph()

    for i, value in enumerate(bin_str):
        if value == 1:
            G.add_edge(
                depart[i], destination[i], weight=mat_adj[depart[i], destination[i]]
            )
        else:
            # Add the edge with a weight of 0 if the edge is not taken
            G.add_edge(depart[i], destination[i], weight=0)

    # Draw the graph with different colors for the edges taken and not taken
    def mpl_draw(graph, with_labels=True, edge_labels=None):
        pos = nx.shell_layout(graph)
        nx.draw(
            graph,
            pos,
            with_labels=with_labels,
            node_size=200,
            node_color="skyblue",
            font_size=10,
            font_color="black",
            font_weight="bold",
            edge_color=[
                (1, 0.6, 0.6) if graph[edges[0]][edges[1]]["weight"] == 0 else "black"
                for edges in graph.edges(data=True)
            ],
        )
        if edge_labels:
            edge_labels = nx.get_edge_attributes(graph, edge_labels)
            nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    # Draw the graph
    mpl_draw(G, with_labels=True, edge_labels="weight")
    # plt.show()
    plt.axis("off")
    # plt.tight_layout()
    plt.legend(
        [
            f"alpha = {(alpha/all_weights_sum):.2f},\n Coût: {min_cost:.2f}\n Noeud de départ : {noeud_de_depart}, \n Noeud de fin : {noeud_de_fin},\n reps : {reps}"
        ],
        loc="upper right",
    )
    # plt.show()
    plt.savefig(f"output/chemin_opt_alpha_{alpha:.2f}.png")  # Save the figure
    plt.close()
