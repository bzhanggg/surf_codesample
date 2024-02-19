import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

class DiTree:
    _seed = 42
    _rng = np.random.default_rng(_seed)
    G = nx.DiGraph()
    W = nx.to_numpy_array(G)

    def __init__(self):
        return

    def __random_root_tree(self, v, addl_e):
        G = nx.DiGraph()
        # root node
        G.add_node(0)
        for i in range(1, v):
            parent_node = self._rng.choice(list(G.nodes))
            G.add_node(i)
            G.add_edge(parent_node, i)
        i = 0
        while i < addl_e:
            v_a, v_b = tuple(self._rng.choice(v, size=2, replace=False))
            if not G.has_edge(v_a, v_b):
                G.add_edge(v_a, v_b)
                i += 1
        return G

    def __random_sink_tree(self, v, addl_e):
        G = nx.DiGraph()
        # sink node
        G.add_node(0)
        for i in range(1, v):
            child_node = self._rng.choice(list(G.nodes))
            G.add_node(i)
            G.add_edge(i, child_node)
        i = 0
        while i < addl_e:
            v_a, v_b = tuple(self._rng.choice(v, size=2, replace=False))
            if not G.has_edge(v_a, v_b):
                G.add_edge(v_a, v_b)
                i += 1
        return G
    
    def __assign_weights(self, G: nx.DiGraph, is_row_stoch: bool):
        A = nx.to_numpy_array(self.G)
        W = np.zeros((A.shape))
        if is_row_stoch:
            for i in G.nodes:
                r_i = self._rng.integers(low=1, high=5)
                in_neighbors = list(G.predecessors(i))
                for j in in_neighbors:
                    W[i][j] = 1 / (len(in_neighbors) + r_i)
                W[i,i] = r_i / (len(in_neighbors) + r_i)
        else:
            for i in G.nodes:
                c_i = self._rng.integers(low=1, high=5)
                out_neighbors = list(G.successors(i))
                for j in out_neighbors:
                    W[j][i] = 1 / (len(out_neighbors) + c_i)
                W[i,i] = c_i / (len(out_neighbors) + c_i)
        return W

    def sparse_tree_digraph(self, v, additional_e, is_row_stoch=True, draw=False):
        self.G = self.__random_root_tree(v, additional_e) if is_row_stoch else self.__random_sink_tree(v, additional_e)
        self.W = self.__assign_weights(self.G, is_row_stoch)
        return self.G, self.W

    def get_stoch_matrix(self):
        return self.W

def is_row_stochastic(A: np.ndarray):
    _, n = A.shape
    print(f"Sum of rows: {A.sum(1)}")
    return np.equal(np.round(A.sum(1)), np.ones(n)).all() # must round to handle floating point error

def is_col_stochastic(A: np.ndarray):
    m, _ = A.shape
    print(f"Sum of columns: {A.sum(0)}")
    return np.equal(np.round(A.sum(0)), np.ones(m)).all()

def draw_dir_graph(G: nx.DiGraph):
    pos = nx.spring_layout(G)
    nx.draw(G, pos=pos)
    plt.show()