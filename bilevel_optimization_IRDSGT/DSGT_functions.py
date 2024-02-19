import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from numpy import linalg as LA
from math import log, exp, ceil, sqrt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

"""shuffle and split data to m local training sets and a test set"""
def divide_train_test_among_agents(train_set: pd.DataFrame, num_agents: int) -> list:
    rng = np.random.default_rng(42)
    N, _ = train_set.shape
    split_idx = sorted(rng.choice(N, size = num_agents-1, replace = False))
    return np.split(train_set, split_idx)

def split_train_test(merged_data: pd.DataFrame, train_set_size: int, test_set_size: int, n_agents: int):
    train_df, test_df = train_test_split(merged_data, train_size=train_set_size, test_size=test_set_size)
    shuffled = train_df.sample(frac=1)
    train_sets = divide_train_test_among_agents(shuffled, n_agents)
    for i in range(n_agents):
        train_sets[i].reset_index(drop=True, inplace=True) # reindex each agent set to start from 0
    return train_df, train_sets, test_df

# Laplacian Heuristics to build doubly stochastic matrix, see [[https://web.stanford.edu/~boyd/papers/pdf/fastavg.pdf]]
def is_nonnegative(A: np.ndarray):
    return np.min(A) >= 0

"""
Returns a tuple (a, b, c) of Booleans, where a is True iff A is square, b is True iff A is nonnegative,
and c is True iff AA
"""
def is_doubly_stochastic(A: np.ndarray):
    x,y,z = True, True, True
    m, n = A.shape
    if (m != n):
        x = False
    if not is_nonnegative(A):
        y = False
    z = np.equal(A.sum(0), np.ones(m)).all() and np.equal(A.sum(1), np.ones(n)).all()
    return x,y,z

def constant_edge_weights(A: np.ndarray, **kwargs):
    m, n = A.shape
    if (m != n):
        print("error: matrix provided is not a valid graph")
        return
    L = kwargs.get('L', A @ A.T)
    eigenvalues = sorted(np.linalg.eigvals(L))
    eig1, eig2 = eigenvalues[0], eigenvalues[-2]
    alpha = 2 / (eig1 + eig2)

    W = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            if i == j:
                d_i = L[i,j]
                W[i,j] = 1 - (d_i * alpha)
            elif A[i,j] != 0:
                W[i,j] = alpha
    return W

def max_degree_weights(A: np.ndarray, **kwargs):
    m, n = A.shape
    if (m != n):
        print("error: matrix provided is not a valid graph")
        return
    L = kwargs.get('L', A @ A.T)
    d_max = np.max(L)
    a_md = 1 / d_max
    W = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            if i == j:
                d_i = L[i,j]
                W[i,j] = 1 - (d_i * a_md)
            elif A[i,j] != 0:
                W[i,j] = a_md
    return W

def local_degree_weights(A: np.ndarray, **kwargs):
    m, n = A.shape
    if (m != n):
        print("error: matrix provided is not a valid graph")
        return
    L = kwargs.get('L', A @ A.T)
    W = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            if A[i,j] != 0:
                d_i, d_j = L[i,i], L[j,j]
                W[i,j] = 1 / max(d_i, d_j)
    return W

def erdos_renyi_graph(v, e, draw=False):
    # use seed for reproducibility
    seed = 0
    G = nx.gnm_random_graph(v, e, seed)
    while not nx.is_connected(G):
        seed += 1
        G = nx.gnm_random_graph(v, e, seed)

    pos = nx.spring_layout(G, seed=seed)  # Seed for reproducible layout
    if draw:
        nx.draw(G, pos=pos)
        plt.show()
    A = nx.to_numpy_array(G)
    n, n = A.shape
    for i in range(n):
        for j in range(n):
            if i > j and A[i,j] == 1:
                A[i, j] = -A[i,j]
    W = max_degree_weights(A)
    return G, W

def complete_graph(v, draw=False):
    if v == 1:
        return nx.complete_graph(1), np.asarray([[1]])
    # use seed for reproducibility
    seed = 42
    G = nx.complete_graph(v)
    pos = nx.spring_layout(G, seed=seed)
    if draw:
        nx.draw(G, pos=pos)
        plt.show()
    A = nx.to_numpy_array(G)
    n, n = A.shape
    for i in range(n):
        for j in range(n):
            A[i,j] = -A[i,j]
    W = max_degree_weights(A)
    return G, W

def sparse_tree_graph(v, additional_e, draw=False):
    seed = 42
    rng = np.random.default_rng(seed)
    G = nx.random_tree(v, seed=seed)
    i = 0
    while i < additional_e:
        v_a, v_b = tuple(rng.integers(v, size=2))
        if not G.has_edge(v_a, v_b):
            G.add_edge(v_a, v_b)
            i += 1
    pos = nx.spring_layout(G, seed=seed)
    if draw:
        nx.draw(G, pos=pos)
        plt.show()
    A = nx.to_numpy_array(G)
    n, n = A.shape
    for i in range(n):
        for j in range(n):
            A[i,j] = -A[i,j]
    W = max_degree_weights(A)
    return G, W

def ring_graph(v, draw=False):
    seed = 42
    G = nx.cycle_graph(v)
    pos = nx.spring_layout(G, seed=seed)
    if draw:
        nx.draw(G, pos=pos)
        plt.show()
    A = nx.to_numpy_array(G)
    n,n = A.shape
    for i in range(n):
        for j in range(n):
            A[i,j] = -A[i,j]
    W = max_degree_weights(A)
    return G, W

def draw_graph(G):
    seed = 42
    pos = nx.spring_layout(G, seed=seed)
    nx.draw(G, pos = pos)
    plt.show()

# UPPER LEVEL FUNCTION
def F_global(x: np.ndarray, mu_param, n_agents: int):
    return sum(f_local(x, mu_param) for _ in range(n_agents)) / n_agents

def f_local(x: np.ndarray, mu_param):
    return 0.5 * float(LA.norm(x, ord=2))**2

def grad_f(x_vec: np.ndarray, mu_param):
    return x_vec

def grad_WF(X_matrix: np.ndarray, mu_param):
    m, n = X_matrix.shape
    output = np.zeros((m,n))
    for i in range(m):
        x_vec = X_matrix[i, :].reshape((n,1))
        grad_vec = grad_f(x_vec, mu_param)
        output[i,:] = grad_vec.reshape((1,n))
    return output/m

# LOWER LEVEL FUNCTION
def G_global(x: np.ndarray, train_sets: list[np.ndarray], n_agents: int):
    return sum(g_local(x, train_sets[i]) for i in range(n_agents)) / n_agents

def g_local(x: np.ndarray, train_set):
    train_set = train_set.to_numpy()
    data, labels = train_set[:, :-1], train_set[:, -1:]     # data is an array size N*784, labels is vector size N*1
    n_labels, n_attributes = data.shape
    x.reshape((n_attributes,1)) # x is a vector size 784*1

    # calculate logit
    ux = data @ x # this should now be N*1
    vux = -np.multiply(labels, ux)
    obj_val = 0
    obj_val = sum([vux[i,0] if vux[i,0] > 709 else log(1 + exp(vux[i, 0])) for i in range(n_labels)]) / n_labels

    output = obj_val
    return output

def sgd_g(x_vec: np.ndarray, train_set, batch_size: int, batch_idx_row: np.ndarray):
    train_set = train_set.to_numpy()
    data, labels = train_set[:, :-1], train_set[:, -1:]
    n_labels, n_attributes = data.shape
    output = np.zeros((n_attributes, 1))
    for r in batch_idx_row:
        i = np.ndarray.item(r) if type(r) == np.ndarray else r
        output += (-labels[i,0]/(1+exp(min(709,labels[i,0]*np.dot(data[i,:],x_vec))))*data[[i],:]).reshape((n_attributes, 1))
    return output/batch_size

def sgd_WG(X_matrix: np.ndarray, train_sets: list[np.ndarray], batch_size: int, batch_idxs: list):
    m, n = X_matrix.shape
    output = np.zeros((m,n))
    for i in range(m):
        x_vec = X_matrix[i,:].reshape((n,1))
        grad_vec = sgd_g(x_vec, train_sets[i], batch_size, batch_idxs[i])
        output[i,:] = grad_vec.reshape((1,n))
    return output/m

"""
F is the upper level function
G is the lower level function
regularizer: L2 norm
"""
def seq_DSGT(train_sets: list[np.ndarray],
    X_0: np.ndarray,
    W: np.ndarray,
    step_size_coefficient: float,
    mu_param: float,
    batch_size: int,
    max_T: int,
    eta_param: int,
    K_param: int):

    # update rule for stepsize (eta is constant for inner DSGT)
    stepsize_rule = lambda k: step_size_coefficient/(k+10)

    m,n = X_0.shape
    const_eta = 1
    X_eta = np.zeros((m,n))

    # set up lists of f, h values
    X_ave = X_0[0] if m == 1 else X_0.mean(0)
    g_vals = [G_global(X_ave, train_sets, m)]
    f_vals = [F_global(X_ave, mu_param, m)]
    consensus_err_vals = [LA.norm(X_0 - np.dot(np.ones((m,1)), X_ave.reshape(1,n)))]

    batch_idxs = [np.random.randint((len(train_sets[i])), size=(batch_size)) for i in range(m)]
    grad_g_mat_next = sgd_WG(X_0, train_sets, batch_size, batch_idxs)
    grad_f_mat_next = grad_WF(X_0, mu_param)

    # Y_0 is initialized to the gradient of the base function
    Y_next = grad_g_mat_next + const_eta * grad_f_mat_next

    epochs = [0]
    iter_ct = 0
    for t in range(max_T):
        const_eta, K = eta_param**-t, K_param**t
        X_next = X_0 if t == 0 else X_eta
        for k in range(K):
            stepsize = stepsize_rule(k)*np.ones((1,m))

            X_now = X_next
            Y_now = Y_next

            grad_f_mat_now = grad_g_mat_next
            grad_h_mat_now = grad_f_mat_next

            X_next = np.dot(W, (X_now - np.dot(stepsize, Y_now)))
            batch_idxs = [np.random.randint((len(train_sets[i])), size=(batch_size)) for i in range(m)]
            grad_g_mat_next = sgd_WG(X_next, train_sets, batch_size, batch_idxs)
            grad_f_mat_next = grad_WF(X_next, mu_param)
            # update Y tracker
            Y_next = np.dot(W, Y_now) + grad_g_mat_next - grad_f_mat_now + const_eta*(grad_f_mat_next - grad_h_mat_now)

            iter_ct += 1
        X_eta = X_next
        X_ave = X_next[0] if m == 1 else X_next.mean(0)

        g_val = G_global(X_ave, train_sets, m)
        print(f"iteration: {iter_ct}, g_val = {g_val}")
        g_vals.append(g_val)
        f_vals.append(F_global(X_ave, mu_param, m))
        consensus = LA.norm(X_next - np.dot(np.ones((m,1)), X_ave.reshape(1,n)))
        consensus_err_vals.append(consensus)
        epochs.append(iter_ct)

    print(consensus_err_vals)
    g_vals, f_vals, consensus_err_vals = np.asarray(g_vals), np.asarray(f_vals), np.asarray(consensus_err_vals)
    return X_eta, g_vals, f_vals, consensus_err_vals, epochs

def iter_DSGT(train_sets: list[np.ndarray],
    epochs: list[int],
    X_0: np.ndarray,
    W: np.ndarray,
    step_size_coefficient: float,
    eta_coefficient: float,
    mu_param: float,
    batch_size: int):

    # update rules for stepsize and eta:
    stepsize_rule = lambda k: step_size_coefficient/pow(k+10, 0.5)
    eta_rule = lambda k: eta_coefficient/pow(k+10, 0.25)
    
    m,n = X_0.shape
    g_vals = []
    f_vals = []
    consensus_err_vals = []
    X_next = X_0
    batch_idxs = [np.random.randint((len(train_sets[i])), size=(batch_size)) for i in range(m)]

    eta_k_next = eta_coefficient
    grad_g_mat_next = sgd_WG(X_next, train_sets, batch_size, batch_idxs)
    grad_f_mat_next = grad_WF(X_next, mu_param)

    # Y_0 is initialized to the gradient of the base function
    Y_next = grad_g_mat_next + eta_k_next * grad_f_mat_next
    max_iter = epochs[-1]
    for k in range(max_iter + 1):
        # store preious values:
        eta_k_now = eta_k_next
        X_now = X_next
        Y_now = Y_next

        # store previous gradients:
        grad_f_mat_now = grad_g_mat_next
        grad_h_mat_now = grad_f_mat_next

        # update eta and stepsize:
        stepsize = stepsize_rule(k)*np.ones((1,m))
        eta_k_next = eta_rule(k)        # don't have to do k+1 anymore?
        # update X:
        X_next = np.dot(W, (X_now - np.dot(stepsize, Y_now)))
        # update gradients:
        batch_idxs = [np.random.randint((len(train_sets[i])), size=(batch_size)) for i in range(m)]
        grad_g_mat_next = sgd_WG(X_next, train_sets, batch_size, batch_idxs)
        grad_f_mat_next = grad_WF(X_next, mu_param)
        # update Y tracker:
        Y_next = np.dot(W, Y_now) + grad_g_mat_next - grad_f_mat_now + eta_k_next * grad_f_mat_next - eta_k_now * grad_h_mat_now

        if k in epochs:
            x_ave = X_now[0] if m == 1 else X_now.mean(0)
            g_val = G_global(x_ave, train_sets, m)
            g_vals.append(g_val)
            f_vals.append(F_global(x_ave, mu_param, m))
            consensus_err_vals.append(LA.norm(X_now - np.dot(np.ones((m,1)), x_ave.reshape(1,n))))
            print(f"iteration: {k}, g_val = {g_val}")

    g_vals, f_vals, consensus_err_vals = np.asarray(g_vals), np.asarray(f_vals), np.asarray(consensus_err_vals)
    return X_next, g_vals, f_vals, consensus_err_vals