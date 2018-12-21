import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# loading data
G_lesmis = nx.read_gml('lesmis.gml')  # Les Mis

# eigenvector centrality
Ceig_lesmis = nx.eigenvector_centrality(G_lesmis)  

# sorting nodes by eigenvector centrality
Ceig_lesmis_node = Ceig_lesmis.keys()
Ceig_lesmis_k = Ceig_lesmis.values()
sortedNodes_lesmis = sorted(zip(Ceig_lesmis_node, Ceig_lesmis_k), 
                            key=lambda x: x[1], reverse=True)
sCeig_lesmis_node, sCeig_lesmis_k = zip(*sortedNodes_lesmis)

# top nodes and their eigenvector centrality
print('Les Mis network -- Top degree centrality nodes')
print('Node           \tEigenvector centrality')
for iNode in range(5):
    print('%-14s\t' % str(sCeig_lesmis_node[iNode]), end='')
    print('%6.4f' % sCeig_lesmis_k[iNode])
print()

# drawing the graph --- Kamada-Kawai layout
plt.figure(figsize=[9,9])
pos = nx.kamada_kawai_layout(G_lesmis, weight=None) # positions for all nodes
nx.draw_networkx_nodes(G_lesmis, pos, 
                       cmap=plt.cm.coolwarm, node_color=list(Ceig_lesmis_k))
nx.draw_networkx_edges(G_lesmis, pos, edge_color='lightblue')
nx.draw_networkx_labels(G_lesmis, pos, font_size=10, font_color='White')
plt.axis('off')
plt.title('Les Mis network\nand eigenvector centrality')
vmin = sCeig_lesmis_k[-1]
vmax = sCeig_lesmis_k[0]
sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, 
                           norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []
cbar = plt.colorbar(sm, shrink=0.5)
cbar.ax.set_ylabel('Eigenvector centrality')
plt.show()
