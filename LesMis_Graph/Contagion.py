import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def FinalContagion(G, pa, pb):
    pa = pa
    pb = pb

    pos = nx.spring_layout(G)
    final_infect = contagion(G, pa, pb, pos, 0.1)
    print(final_infect)


def direct_neighbor(G, node, flag):  # flag=1 out edges, 0 in edges
    # returns neighbors
    return ([neigh[flag] for neigh in (G.in_edges(node) if flag == 0 else G.out_edges(node))]
            if G.is_directed() else G.neighbors(node))


# pick p% of the nodes at random and set their status to B, others go to A
def initialize_random_status(G, p):
    N_sel = int(len(G) * p)
    sel_nodes = np.random.permutation(G.node)[:N_sel]  # infected nodes
    for nd in G.nodes:
        if nd in sel_nodes:
            G.node[nd]['status'] = 'B'
        else:
            G.node[nd]['status'] = 'A'
    return list(sel_nodes)


def headToHead(G, node, pa, pb):
    neig_nodes = direct_neighbor(G, node, 0)
    payoff = 0

    for nd in neig_nodes:  # check all neightbor, if the payoff > 0, CHANCE!!!
        # 1 vs 1
        if G.node[nd]['status'] == 'A':
            payoff -= pa  # does want to change
        else:
            payoff += pb  # wants to change
    if payoff > 0:  # CONTAGIATE :)))))))
        G.node[node]['status'] = 'B'


def plot_contagion(G, pos, i): # Plot epidemy
    st_dic = nx.get_node_attributes(G, 'status')
    blue_nodes = []
    red_nodes = []

    for x in st_dic:
        if st_dic[x] == 'B':
            blue_nodes.append(x)

        else:
            red_nodes.append(x)

    plt.figure(1, (12, 8))
    nx.draw_networkx_nodes(G, pos, nodelist=red_nodes, node_color='r', alpha=0.6)
    nx.draw_networkx_nodes(G, pos, nodelist=blue_nodes, node_color='b', alpha=0.8)

    nx.draw_networkx_edges(G, pos, color='green', width=0.3, alpha=0.3)
    nx.draw_networkx_labels(G, pos)
    plt.axis('off')

    if not os.path.exists("output"):  # create dir in same place of this file ipython, if there isn't yet
        os.makedirs("output")
    plt.savefig("output/" + "contagion" + str(i) + ".png", format="PNG")
    plt.show(G)
    return len(blue_nodes) / float(len(G))


def contagion(G, pa, pb, pos, perc_inf):  # MAIN FUNC
    new_blues_list = initialize_random_status(G, perc_inf)
    old_blues_list = list()

    i = 1

    final_infct = plot_contagion(G, pos, 0)  # i=0
    stall = False

    # until there's in stall situation
    while not stall:
        # update all nodes

        for nd in new_blues_list:
            for nd_neigh in direct_neighbor(G, nd, 1):  # only blues neighbors are affected
                headToHead(G, nd_neigh, pa, pb)

        old_blues_list += new_blues_list  # i don't want to visit these infected again

        # takes only infected not already visited
        new_blues_list = [key for key, status in nx.get_node_attributes(G, 'status').items() if status == 'B' and \
                          key not in old_blues_list]

        # check stall
        if not new_blues_list:
            stall = True
        else:
            # plot
            final_infct = plot_contagion(G, pos, i)
            i += 1
    return final_infct
