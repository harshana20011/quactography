import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import rustworkx as rx

from rustworkx.visualization import mpl_draw


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
    """_summary_ : Visualiser le graphe et le chemin optimal trouvé.

    Args:
        depart (list int): liste de points départs
        destination (list int): liste de points finaux
        mat_adj (np array):  matrice d'adjacence
        list(map(int, bin_str)) (liste int): Chaîne d'entiers binaires représentant le chemin
    """
    bin_str = list(map(int, bin_str))
    bin_str.reverse()

    # Créer un graphe dirigé
    G = nx.Graph()

    for i, value in enumerate(bin_str):
        if value == 1:
            G.add_edge(
                depart[i], destination[i], weight=mat_adj[depart[i], destination[i]]
            )
        else:
            # Ajouter une arête avec un poids 0 et une couleur rose
            G.add_edge(depart[i], destination[i], weight=0)

    # Fonction pour dessiner le graphe avec des couleurs personnalisées
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

    # Dessiner le graphe
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
