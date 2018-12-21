
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import networkx as nx

filename = './data/lesmis.gml'
G_lesmis = nx.read_gml(filename)

print(nx.info(G_lesmis))


# In[2]:


# maximal cliques with Bron–Kerbosch algorithm 
from networkx.algorithms import clique

# Returns the number of maximal cliques in the graph.
print(clique.graph_number_of_cliques(G_lesmis))

#Returns the number of maximal cliques for each node.
print(clique.number_of_cliques(G_lesmis))

# Returns all maximal cliques in the graph.
cliques = list(nx.find_cliques(G_lesmis))
sorted(cliques)


# In[3]:


# Fixed-sized cliques
from networkx.algorithms.community import k_clique_communities
c3 = list(k_clique_communities(G_lesmis, 3))
sorted(c3[0])


# In[4]:


sorted(c3[1])


# In[5]:


sorted(c3[2])


# In[6]:


sorted(c3[3])


# In[7]:


c4 = list(k_clique_communities(G_lesmis, 4))
sorted(c4[0])


# In[8]:


sorted(c4[1])


# In[9]:


sorted(c4[2])


# In[10]:


sorted(c4[3])

