import matplotlib.pyplot as plt
import networkx as nx
filename = 'lesmis.gml'
G_lesmis=nx.read_gml(filename)
#G_ka=nx.karate_club_graph()
G_lesmis.node

print(nx.info(G_lesmis))

#Create network layout for visualizations
spring_pos = nx.spring_layout(G_lesmis)

plt.axis("off")
nx.draw_networkx(G_lesmis, pos = spring_pos, with_labels = False, node_size = 15)



############### Community detection ###########
import community as com
parts = com.best_partition(G_lesmis)
values = [parts.get(node) for node in G_lesmis.nodes()]



plt.axis("off")
nx.draw_networkx(G_lesmis, pos = spring_pos, cmap = plt.get_cmap("jet"), node_color = values, font_size=20,node_size = 80, with_labels = False)


## Calculate the modularity ##
com.modularity(parts, G_lesmis)

## induced graph : each community is represented as one node ##
help(com)
G_induced=com.induced_graph(parts, G_lesmis)
plt.axis("off")
nx.draw_networkx(G_induced, cmap = plt.get_cmap("jet"),  font_size=20,node_size = 80, with_labels = False)





