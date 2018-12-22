import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import operator
import random
import matplotlib.ticker as mtick
import os


def NetworkAttacks(G):
    G_simple = nx.Graph(G)
    G_simple2 = nx.Graph(G)
    G_simple3 = nx.Graph(G)
    G_simple4 = nx.Graph(G)

    between = nx.betweenness_centrality(G_simple)

    sortedClust_X = sorted(nx.clustering(G).items(), key=operator.itemgetter(1), reverse=True)
    sorted_x = sorted(between.items(), key=operator.itemgetter(1), reverse=True)
    sortedDeg_x = sorted(G.degree(), key=operator.itemgetter(1), reverse=True)
    n_nodes = len(G_simple.nodes)
    rand_x = list(range(0, n_nodes))

    random.shuffle(rand_x)
    between_giant = [77]
    between_rand = [77]
    between_Clu = [77]
    between_Deg = [77]

    for x in range(76):
        removeBet = sorted_x[x]
        removeRand = sorted_x[rand_x[x]]

        removeDeg = sortedDeg_x[x]
        removeClu = sortedClust_X[x]

        G_simple.remove_nodes_from(removeBet)
        G_simple3.remove_nodes_from(removeDeg)

        G_simple2.remove_nodes_from(removeRand)
        G_simple4.remove_nodes_from(removeClu)

        giant = len(max(nx.connected_component_subgraphs(G_simple), key=len))
        giant2 = len(max(nx.connected_component_subgraphs(G_simple2), key=len))
        giant3 = len(max(nx.connected_component_subgraphs(G_simple3), key=len))
        giant4 = len(max(nx.connected_component_subgraphs(G_simple4), key=len))

        between_giant.append(giant)
        between_rand.append(giant2)

        between_Deg.append(giant3)
        between_Clu.append(giant4)

    perc = np.linspace(0, 100, len(between_giant))
    fig = plt.figure(1, (12, 8))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(np.linspace(0, 100, len(between_giant)), between_giant)
    ax.plot(np.linspace(0, 100, len(between_rand)), between_rand)
    ax.plot(np.linspace(0, 100, len(between_Deg)), between_Deg)
    ax.plot(np.linspace(0, 100, len(between_Clu)), between_Clu)

    fmt = '%.0f%%'  # Format you want the ticks, e.g. '40%'
    xticks = mtick.FormatStrFormatter(fmt)
    ax.xaxis.set_major_formatter(xticks)
    ax.set_xlabel('Fraction of Nodes Removed')
    ax.set_ylabel('Giant Component Size')
    ax.legend(['betweenness', 'random', 'HighDegree', 'highest clustering coefficient '])
    if not os.path.exists("output"):  # create dir in same place of this file ipython, if there isn't yet
        os.makedirs("output")
    plt.savefig("output/" + "Attack" + ".png", format="PNG")
    plt.show()
