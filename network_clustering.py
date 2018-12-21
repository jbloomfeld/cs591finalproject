#!/usr/bin/env python
# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.cluster import KMeans

class graph_clustering():
    def __init__(self, K, n_clusters, lap_kind):
        """
        K : int
        n_clusters : int
        lap_kind : None or str
            Ex. "rw"
        """
        self.lap_kind = lap_kind
        self.K = K
        self.n_clusters = n_clusters

    def _calc_g_laplacian(self, X):
        D = np.diag(np.sum(X, axis=0))
        L = D - X
        self.L = L
        self.D = D
        return L

    def _calc_rw_g_laplacian(self, X):
        L = self._calc_g_laplacian(X)
        Lrw = np.dot(np.linalg.inv(self.D), L)
        self.L = Lrw
        return Lrw

    def _get_K_eigen_vec(self, Lam, V):
        sorted_index = np.argsort(Lam.real)
        Lam_K = Lam[sorted_index][0:self.K]
        V_K = V[sorted_index][0:self.K]

        self.Lam_K = Lam_K
        self.V_K = V_K

        return Lam_K, V_K

    def _Kmeans_V_K(self, V_K):
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(V_K.T.real)
        clusters = kmeans.labels_
        self.clusters = clusters
        return clusters

    def laplacian_clustering(self, X):
        """
        X : ndarray
            a matrix representation of undirected graph
        """
        if self.lap_kind is None:
            L = self._calc_g_laplacian(X)
        elif self.lap_kind == "rw":
            L = self._calc_rw_g_laplacian(X)
        else: 
            raise NotImplementedError

        Lam, V = np.linalg.eig(L)
        Lam_K, V_K = self._get_K_eigen_vec(Lam, V)
        clusters = self._Kmeans_V_K(V_K)
        return clusters

# for plot
def get_labels_dic(G):
    """
    G : networkx.classes.graph.Graph
    """
    labels_dic = { key:G.node[key]['label'] for key in G.node.keys() }
    return labels_dic

def get_labels_list(G):
    labels_list = np.array([ G.node[key]['label'] for key in G.node.keys() ])
    return labels_list

def split_labels(labels_list, clusters):
    """
    labels_list : list
        return value of get_labels_list function.
    clusters : ndarray
        return value of graph_clustering.laplacian_clustering
    """
    class_uniq = np.unique(clusters)
    node_index_split= []
    labels_split = []
    index = np.arange(clusters.shape[0])
    for class_elem in class_uniq:
        node_index_split.append(list(index[clusters == class_elem]))
        labels_split.append(list(labels_list[clusters == class_elem]))
    return node_index_split, labels_split

def plot_clusters(G, node_index_split):
    """
    G : networkx.classes.graph.Graph
    node_index_split : list (2-dim list)
        return value of split_labels function.
    """
    labels_dic = get_labels_dic(G)
    pos = nx.spring_layout(G)
    colors = [ cm.jet(x) for x in np.arange(0, 1, 1/len(node_index_split)) ]
    for i, nodes in enumerate(node_index_split):
        nx.draw_networkx(G, node_color=colors[i], labels=labels_dic, nodelist=nodes, pos=pos, font_color="r")
    plt.show()

data_path = "./data/lesmis/lesmis.gml"
G = nx.read_gml(data_path)
nx.is_directed(G)

# X, order = nx.attr_matrix(G)
# X = np.array(X)

X = np.array(nx.to_numpy_matrix(G))
# type(M) numpy.matrixlib.defmatrix.matrix

GClus = graph_clustering(K=15, n_clusters=5, lap_kind="rw")
clusters = GClus.laplacian_clustering(X)
labels_list = get_labels_list(G)
node_index_split, labels_split = split_labels(labels_list, clusters)
plot_clusters(G, node_index_split)

##### 
nx.draw(G)
plt.show()
plt.close()

labels_dic = get_labels_dic(G)
    nx.draw_networkx(G, node_color="b", labels=labels_dic)
    plt.show()


    D = np.diag(np.sum(X, axis=0))

    L = D - X

    Lam, V = np.linalg.eig(L)
    np.sum(np.abs(L - np.dot(np.dot(V, np.diag(Lam)), V.T)))
    L.size

    Lam.real
    Lam.imag




K = 15
Lam_K, V_K = get_K_eigen_vec(K, Lam, V)


kmeans.cluster_centers_

Lrw = np.dot(np.linalg.inv(D), L)

Lam_rw, V_rw = np.linalg.eig(Lrw)

K = 15
Lam_rw_K, V_rw_K = get_K_eigen_vec(K, Lam_rw, V_rw)
kmeans = KMeans(n_clusters=2, random_state=0).fit(V_rw_K.T.real)
kmeans.labels_

type(G.node)
labels = np.array([ G.node[key]['label'] for key in G.node.keys() ])
labels[kmeans.labels_.astype(bool)]

clusters = laplacian_clustering(X, n_clusters=5, lap_kind="rw")
class_list = np.unique(clusters)



node_index_list, labels_list = split_labels(clusters)

nodelist1 = list(np.arange(77)[kmeans.labels_.astype(bool)])
labels_inv = np.abs(kmeans.labels_ - 1).astype(bool)
nodelist2 = list(np.arange(77)[labels_inv])
#pos = nx.spring_layout(G)
#pos = nx.spectral_layout(G)
pos = nx.circular_layout(G)
nx.draw_networkx(G, node_color="b", labels=labels_dic, nodelist=nodelist1, pos=pos)
nx.draw_networkx(G, node_color="r", labels=labels_dic, nodelist=nodelist2, pos=pos)

plt.show()

    plt.close("all")