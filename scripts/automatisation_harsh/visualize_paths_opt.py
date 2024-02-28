import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Code pour matrice d'adjacence déjà existante :
df = pd.read_csv(r"scripts\automatisation_harsh\matrices\mat_adj.csv")
mat_adj = np.array(df)

G = nx.Graph()

G.add_edge(0, 1, weight=int(mat_adj[0, 1]))
G.add_edge(0, 3, weight=int(mat_adj[0, 3]))
G.add_edge(0, 4, weight=int(mat_adj[0, 4]))
G.add_edge(1, 2, weight=int(mat_adj[1, 2]))
G.add_edge(1, 3, weight=int(mat_adj[1, 3]))
G.add_edge(1, 4, weight=int(mat_adj[1, 4]))
G.add_edge(2, 5, weight=int(mat_adj[2, 5]))
G.add_edge(2, 6, weight=int(mat_adj[2, 6]))
G.add_edge(3, 4, weight=int(mat_adj[3, 4]))
G.add_edge(4, 5, weight=int(mat_adj[4, 5]))
G.add_edge(5, 6, weight=int(mat_adj[5, 6]))

pos = nx.spring_layout(G)
edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_nodes(G, pos, node_size=400, node_color="#797EF6")
nx.draw_networkx_edges(G, pos, width=2, edge_color="#797EF6")
nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif", font_color="w")
nx.draw_networkx_edge_labels(G, pos, edge_labels)

ax = plt.gca()
ax.margins(0.08)
plt.axis("off")
plt.tight_layout()
plt.show()

# Display graph
bin_str = list(map(int, max(dist.binary_probabilities(), key=dist.binary_probabilities().get)))  # type: ignore
bin_str.reverse()
print(bin_str)

pos = nx.spring_layout(G, seed=7)
edge_labels = nx.get_edge_attributes(G, "weight")

e_in = [(u, v) for i, (u, v, d) in enumerate(G.edges(data=True)) if bin_str[i]]
e_out = [(u, v) for i, (u, v, d) in enumerate(G.edges(data=True)) if not bin_str[i]]

print(e_in)

color_map = np.array(["#D3D3D3"] * G.number_of_nodes())
print(list(sum(e_in, ())))
color_map[list(sum(e_in, ()))] = "#EE6B6E"

nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=400)  # type: ignore
nx.draw_networkx_edges(
    G, pos, edgelist=e_in, width=2, alpha=1, edge_color="#EE6B6E", style="dashed"
)
nx.draw_networkx_edges(G, pos, edgelist=e_out, width=2, edge_color="#D3D3D3")
nx.draw_networkx_edge_labels(G, pos, edge_labels)
nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif", font_color="w")

ax = plt.gca()
ax.margins(0.08)
plt.axis("off")
plt.tight_layout()
plt.show()
