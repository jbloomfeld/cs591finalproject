import networkx as nx
import Measures
import Robustness
import Contagion


def main():
    """
    # read the graph (gml format)
    G = nx.read_gml('lesmiserables.gml')
    # Labo1
    Measures.metrics(G)
    Measures.PlotBar(dict(G.degree()),"Les Miserables ")
    # labo2
    Robustness.NetworkAttacks(G)
    # labo3
    Contagion.FinalContagion(G, 2, 7)
    """

    barabasi_graph = nx.barabasi_albert_graph(77, 5, seed=None)
    Measures.PlotBar(dict(barabasi_graph.degree()), "Barab ")
    Measures.metrics(barabasi_graph)
    Robustness.NetworkAttacks(barabasi_graph)
    Contagion.FinalContagion(barabasi_graph,2,6)


if __name__ == "__main__":
    main()
