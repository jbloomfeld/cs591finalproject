
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import networkx as nx

filename = './data/lesmis.gml'
G_lesmis = nx.read_gml(filename)

print(nx.info(G_lesmis))


# In[2]:


from networkx.algorithms.community import greedy_modularity_communities
'''
 Find communities in graph using Clauset-Newman-Moore greedy modularity maximization. 
 This method currently supports the Graph class without considering edge weights.

 Greedy modularity maximization begins with each node in its own community and joins the pair of communities 
 that most increases modularity until no such pair exists.
'''
c = list(greedy_modularity_communities(G_lesmis))
sorted(c[0])


# In[3]:


sorted(c[1])


# In[4]:


sorted(c[2])


# In[5]:


sorted(c[3])


# In[6]:


sorted(c[4])


# In[18]:


print(len(c[0]),len(c[1]),len(c[2]),len(c[3]),len(c[4]))

