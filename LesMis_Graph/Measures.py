from pylab import show, hist, figure
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import networkx as nx
from collections import Counter
import matplotlib.pyplot as plt
import os


def most_important(G):
    ranking = nx.betweenness_centrality(G).items()
    r = [x[1] for x in ranking]
    m = sum(r) / len(r)  # mean centrality
    t = m * 3  # threshold, we keep only the nodes with 3 times the mean
    Gt = G.copy()
    for k, v in ranking:
        if v < t:
            Gt.remove_node(k)
    return Gt


def trim_nodes(G, d):
    """ returns a copy of G without
     the nodes with a degree less than d """
    Gt = G.copy()
    dn = nx.degree(Gt)
    for n in Gt.nodes():
        if dn[n] <= d:
            Gt.remove_node(n)
    return Gt


def metrics(G):
    # drawing the full network
    plot_graph(G)
    show()

    print(nx.info(G))

    clos = nx.closeness_centrality(G).values()
    bet = nx.betweenness_centrality(G, normalized=True).values()

    print("isolated nodes: ", nx.number_of_isolates(G))
    print("Density:", nx.density(G))
    print("Diameter:", nx.diameter(G))
    print("Connectivity:", nx.node_connectivity(G))
    print("Average short path", nx.average_shortest_path_length(G))
    print("Assortativity:",nx.degree_assortativity_coefficient(G))
    giant = sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)
    print("Size of biggest GCC (nodes): ", giant[0].order())
    print("Max betweeness", np.max(np.array(list(bet))))
    print("Max closeness", np.max(np.array(list(clos))))
    print("AVG betweeness", np.mean(np.array(list(bet))))
    print("AVG closeness", np.mean(np.array(list(clos))))
    PlotMostImp(G)


def plot_graph(G):
    pos = nx.spring_layout(G)
    # draw the nodes and the edges (all)
    nx.draw_networkx_nodes(G, pos, node_color='red', node_size=80)
    nx.draw_networkx_edges(G, pos, )
    if not os.path.exists("output"):  # create dir in same place of this file ipython, if there isn't yet
        os.makedirs("output")
    plt.savefig("output/" + "graph" + ".png", format="PNG")
    plt.show()


def distDegree(G):
    degree_dic = Counter(dict(G.degree()))

    degree_hist = pd.DataFrame({"degree": list(degree_dic.values()),
                                "Number of Nodes": list(degree_dic.keys())})
    plt.figure(figsize=(20, 10))
    sns.barplot(y='degree', x='Number of Nodes',
                data=degree_hist,
                color='darkblue')
    plt.xlabel('Node Degree', fontsize=30)
    plt.ylabel('Number of Nodes', fontsize=30)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.show()


def plot_degree_distribution(degrees):
    N = len(degrees)
    hist = [list(degrees.values()).count(x) / N for x in degrees.values()]
    fig = plt.figure(figsize=(17, 17))
    plt.title("Degree Distribution")
    plt.grid(True)
    sns.set_style("ticks")
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_xlabel('Degree')
    ax.set_ylabel('$p_k$')

    plt.scatter(degrees.values(), hist, color='orange')


def PlotBar(degrees, title):
    hist = [list(degrees.values()).count(x) for x in degrees.values()]
    fig = plt.figure(figsize=(12, 12))
    plt.title(title + "degree distribution")
    plt.grid(True)
    sns.set_style("ticks")
    plt.bar(degrees.values(), hist, color='red')
    if not os.path.exists("output"):  # create dir in same place of this file ipython, if there isn't yet
        os.makedirs("output")
    plt.savefig("output/" + "DegreeDistribution" + ".png", format="PNG")
    plt.show()


def PlotMostImp(G):
    Gt = most_important(G)
    # create the layout
    pos = nx.spring_layout(G)
    # draw the nodes and the edges (all)
    nx.draw_networkx_nodes(G, pos, node_color='r', alpha=0.2, node_size=8)
    nx.draw_networkx_edges(G, pos, alpha=0.1)
    # draw the most important nodes with a different style
    nx.draw_networkx_nodes(Gt, pos, node_color='b', alpha=0.4, node_size=254)
    # also the labels this time
    nx.draw_networkx_labels(Gt, pos, font_size=12, font_color='b')
    if not os.path.exists("output"):  # create dir in same place of this file ipython, if there isn't yet
        os.makedirs("output")
    plt.savefig("output/" + "MostImportBet" + ".png", format="PNG")
    plt.show()
