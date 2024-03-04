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
):
    """_summary_ : Visualiser le graphe et le chemin optimal trouvé.

    Args:
        depart (list int): liste de points départs
        destination (list int): liste de points finaux
        mat_adj (np array):  matrice d'adjacence
        list(map(int, bin_str)) (liste int): Chaîne d'entiers binaires représentant le chemin
    """

    G = nx.Graph()
    for i, _ in enumerate(depart):
        G.add_edge(
            depart[i],
            destination[i],
            weight=(mat_adj[depart[i], destination[i]]).round(2),
        )
    print(G)

    pos = nx.spring_layout(G, seed=0)  # Adjust the seed for reproducibility
    edge_labels = nx.get_edge_attributes(G, "weight")

    # fig, ax = plt.subplots()

    nx.draw_networkx_nodes(G, pos, node_size=400, node_color="#797EF6")
    nx.draw_networkx_edges(G, pos, width=2, edge_color="#797EF6")
    nx.draw_networkx_labels(G, pos, font_family="sans-serif", font_color="w")
    nx.draw_networkx_edge_labels(G, pos, edge_labels)

    plt.axis("off")
    # plt.tight_layout()
    # plt.show()

    plt.savefig("output/graphe_general.png")  # Save the figure
    plt.close()

    bin_str = list(map(int, bin_str))
    bin_str.reverse()
    pos = nx.spring_layout(G, seed=0)
    edge_labels = nx.get_edge_attributes(G, "weight")

    e_in = [(u, v) for i, (u, v, d) in enumerate(G.edges(data=True)) if bin_str[i]]
    e_out = [(u, v) for i, (u, v, d) in enumerate(G.edges(data=True)) if not bin_str[i]]

    print(e_in)

    color_map = np.array(["#D3D3D3"] * G.number_of_nodes())
    print(list(sum(e_in, ())))
    color_map[list(sum(e_in, ()))] = "#EE6B6E"

    nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=300)  # type: ignore
    nx.draw_networkx_edges(
        G, pos, edgelist=e_in, width=2, alpha=1, edge_color="#EE6B6E", style="dashed"
    )
    nx.draw_networkx_edges(G, pos, edgelist=e_out, width=2, edge_color="#D3D3D3")
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    nx.draw_networkx_labels(
        G, pos, font_size=10, font_family="sans-serif", font_color="w"
    )

    ax = plt.gca()

    plt.axis("off")
    # plt.tight_layout()
    plt.legend(
        [
            f"alpha = {alpha:.2f},\n Coût: {min_cost:.2f}\n Noeud de départ : {noeud_de_depart}, \n Noeud de fin : {noeud_de_fin},\n reps : {reps}"
        ],
        loc="upper right",
    )
    # plt.show()
    plt.savefig(f"output/chemin_opt_alpha_{alpha:.2f}.png")  # Save the figure
    plt.close()


def visualizerust(
    depart,
    destination,
    mat_adj,
    bin_str,
    alpha,
    min_cost,
    noeud_de_depart,
    noeud_de_fin,
    reps,
    num_nodes,
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
            f"alpha = {alpha:.2f},\n Coût: {min_cost:.2f}\n Noeud de départ : {noeud_de_depart}, \n Noeud de fin : {noeud_de_fin},\n reps : {reps}"
        ],
        loc="upper right",
    )
    # plt.show()
    plt.savefig(f"output/chemin_opt_alpha_{alpha:.2f}.png")  # Save the figure
    plt.close()
