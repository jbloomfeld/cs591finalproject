import numpy as np
import networkx as nx
import collections as c
import matplotlib.pyplot as plt


### Load Datasets

# Read the Les Miserables co-occurrence graph.
graphLM = nx.read_gml('lesmis.gml')
matrixLM = nx.to_scipy_sparse_matrix(graphLM)
# Layout a graph using the spring force algorithm.
nx.draw_spring(graphLM)
# Print the matrix.
plt.spy(matrixLM, precision=1e-3, marker='.', markersize=5)



### Fundamental Network Statistics

## Degree
degreeLM = matrixLM.sum(0)

# Plotting
# Degree Distribution
np.squeeze(np.asarray(degreeLM))
plt.hist(np.squeeze(np.asarray(degreeLM)))
plt.title("Degree Distribution")
plt.xlabel("Degree")
plt.ylabel("Frequency")

# Degree Rank
plt.loglog(sorted(np.squeeze(np.asarray(degreeLM)), reverse=True), 'b-', marker='o')
plt.title("Degree Rank")
plt.ylabel("Degree")
plt.xlabel("Rank")

plt.axes([0.45,0.45,0.45,0.45])
Gcc=max(nx.connected_component_subgraphs(graphLM), key=len)
pos=nx.spring_layout(Gcc)
plt.axis('off')
nx.draw_networkx_nodes(Gcc, pos, node_size=20)
nx.draw_networkx_edges(Gcc, pos, alpha=0.4)

# log Binning
# Based on: http://stackoverflow.com/questions/16489655/plotting-log-binned-network-degree-distributions
def drop_zeros(a_list):
    return [i for i in a_list if i>0]

degreeLM_dict = dict(c.Counter(np.squeeze(np.asarray(degreeLM))))
max_x = np.log10(max(degreeLM_dict.keys()))
max_y = np.log10(max(degreeLM_dict.values()))
max_base = max([max_x,max_y])

min_x = np.log10(min(drop_zeros(degreeLM_dict.keys())))
bins = np.logspace(min_x, max_base, num=10)

bin_means_y = (np.histogram(degreeLM_dict.keys(), bins, weights=degreeLM_dict.values())[0] / np.histogram(degreeLM_dict.keys(),bins)[0])
bin_means_x = (np.histogram(degreeLM_dict.keys(), bins, weights=degreeLM_dict.keys())[0] / np.histogram(degreeLM_dict.keys(),bins)[0])

plt.xscale('log')
plt.yscale('log')
plt.scatter(bin_means_x, bin_means_y, c='r', marker='s', s=50)
plt.xlim((0.75,70))
plt.ylim((.9,75))
plt.xlabel('Connections (normalized)')
plt.ylabel('Frequency')



# Cumulative Degree Distribution
# The cumulative degree distribution can simply be computed by:
# sorting the degrees of each vertex in descending order
# compute the corresponding ranks 1...n
# plot the rank divided by the number of vertices as a function or the degree

cumDegreeLM = np.array([np.sort(np.squeeze(np.asarray(degreeLM)))[::-1]])
cumDegreeLM = np.concatenate((cumDegreeLM, np.array([range(1, degreeLM.shape[1]+1)], dtype=np.float)), 0)
cumDegreeLM = np.concatenate((cumDegreeLM, np.array([cumDegreeLM[1]/degreeLM.shape[1]])), 0)

plt.loglog(cumDegreeLM[0,:], cumDegreeLM[2,:])
plt.title("Cumulative Degree Distribution")
plt.xlabel("Degree (k)")
plt.ylabel("$P(x \geq k)$")


## Get Minimum/Maximum Degrees
print([np.min(degreeLM), np.max(degreeLM)]) 

## Get Number of Edges
edgesLM = matrixLM.sum()/2
print(edgesLM)

## Get Mean Degree
cLM = 2 * edgesLM/matrixLM.shape[0]
print(cLM)

## Get Density
rhoLM = cLM/(matrixLM.shape[0]-1.0)
print(rhoLM)



### Network Centrality
graphLM = nx.read_gml('data/lesmis.gml')

## Eigenvalue spectrum
spectrum = np.sort(nx.laplacian_spectrum(graphLM))
plt.plot(spectrum)

## Degree Centrality
degreeCentrality = nx.degree_centrality(graphLM)
layout=nx.spring_layout(graphLM,k=.2,iterations=1000, scale=5)
values = [degreeCentrality.get(node)/max(degreeCentrality.values()) for node in graphLM.nodes()]
nx.draw(graphLM, pos=layout, cmap = plt.get_cmap('jet'), node_color=values, with_labels=False)
plt.savefig('data/lesMiserables-degree-centrality.svg')
plt.savefig('data/lesMiserables-degree-centrality.pdf')


## Closeness
closenessCentrality = nx.closeness_centrality(graphLM)
values = [closenessCentrality.get(node)/max(closenessCentrality.values()) for node in graphLM.nodes()]
nx.draw(graphLM, pos=layout, cmap = plt.get_cmap('jet'), node_color=values, with_labels=False)
plt.savefig('data/lesMiserables-closeness-centrality.svg')
plt.savefig('data/lesMiserables-closeness-centrality.pdf')


## Betweenness
betweennessCentrality = nx.betweenness_centrality(graphLM)
values = [betweennessCentrality.get(node)/max(betweennessCentrality.values()) for node in graphLM.nodes()]
nx.draw(graphLM, pos=layout, cmap = plt.get_cmap('jet'), node_color=values, with_labels=False)
plt.savefig('data/lesMiserables-betweenness-centrality.svg')
plt.savefig('data/lesMiserables-betweenness-centrality.pdf')


## Eigenvector
eigenCentrality = nx.eigenvector_centrality(graphLM)
values = [eigenCentrality.get(node)/max(eigenCentrality.values()) for node in graphLM.nodes()]
nx.draw(graphLM, pos=layout, cmap = plt.get_cmap('jet'), node_color=values, with_labels=False)
plt.savefig('data/lesMiserables-eigen-centrality.svg')
plt.savefig('data/lesMiserables-eigen-centrality.pdf')
