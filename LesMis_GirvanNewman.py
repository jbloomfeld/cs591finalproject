
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import networkx as nx

filename = './data/lesmis.gml'
G_lesmis = nx.read_gml(filename)

print(nx.info(G_lesmis))


# In[2]:


from networkx.algorithms import community
com = community.girvan_newman(G_lesmis)
top_level_communities = next(com)
next_level_communities = next(com)
sorted(map(sorted, next_level_communities))

