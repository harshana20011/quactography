import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import rustworkx as rx

from rustworkx.visualization import mpl_draw


def visualize(
    starting_nodes,
    ending_nodes,
    mat_adj,
    bin_str,
    alpha,
    min_cost,
    starting_node,
    ending_node,
    reps,
    all_weights_sum,
):
    """_summary_ :Visualize the path taken in the graph and save the figure in the output folder

    Args:
        starting_nodes (list int): list of starting points
        ending_nodes (list int): list of ending_nodes points
        mat_adj (np array):  adjacency matrix
        list(map(int, bin_str)) (liste int): list of 0 and 1 representing the path taken
    """
    bin_str = list(map(int, bin_str))

    # Create a graph where the edges taken are in green and the edges not taken are in black
    G = nx.Graph()
    edges_taken = []
    edges_not_taken = []

    for i, value in enumerate(bin_str):
        if value == 1:
            edge_taken = (
                starting_nodes[i],
                ending_nodes[i],
                {
                    "weight": mat_adj[starting_nodes[i], ending_nodes[i]],
                    "color": "green",
                },
            )
            edges_taken.append(edge_taken)
        elif value == 0:
            edge_not_taken = (
                starting_nodes[i],
                ending_nodes[i],
                {
                    "weight": mat_adj[starting_nodes[i], ending_nodes[i]],
                    "color": "black",
                },
            )
            edges_not_taken.append(edge_not_taken)
    G.add_edges_from(edges_taken)
    G.add_edges_from(edges_not_taken)

    # Draw the graph
    edge_label = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos=nx.planar_layout(G), edge_labels=edge_label)
    nx.draw_planar(
        G,
        with_labels=True,
        node_size=200,
        node_color="skyblue",
        font_size=10,
        edge_color=[G[u][v]["color"] for u, v in G.edges()],
        font_color="black",
        font_weight="bold",
    )

    # plt.show()
    plt.axis("off")
    # plt.tight_layout()
    plt.legend(
        [
            f"alpha = {(alpha/all_weights_sum):.2f},\n Cost: {min_cost:.2f}\n Starting node : {starting_node}, \n Ending node : {ending_node},\n reps : {reps}"
        ],
        loc="upper right",
    )
    # plt.show()
    plt.savefig(f"output/Opt_path_alpha_{alpha:.2f}.png")  # Save the figure
    plt.close()
