from __future__ import annotations
from collections.abc import Sequence

import copy
import warnings

import networkx as nx
import numpy as np
import pandapower as pp

warnings.filterwarnings('ignore', category=FutureWarning)


def create_RX_from_net(net: pp.pandapowerNet, noise: float = 0,
                       modify: str | None = None, seed: int | None = 123,
                       check_pd: bool = True
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Creates R,X matrices from a pandapowerNet.

    Args
    - net: pandapowerNet with (n+1) buses including substation
    - noise: float, optional add uniform noise to impedances, values in [0,1]
    - modify: str, how to modify network, one of [None, 'perm', 'linear', 'rand']
    - seed: int, for generating the uniform noise
        seed must be provided if (noise > 0) or (modify is not None)
    - check_pd: bool, whether to assert that returned R,X are PD

    Returns: tuple (X, R)
    - X: np.array, shape [n, n], positive definite and entry-wise positive
    - R: np.array, shape [n, n], positive definite and entry-wise positive
    """
    assert 0 <= noise <= 1, 'noise must be a float in [0,1]'

    # read in r and x matrices from data
    n = len(net.bus) - 1  # number of buses, excluding substation
    r = np.ones((n+1, n+1)) * np.inf
    x = np.ones((n+1, n+1)) * np.inf

    r_ohm_per_km = net.line['r_ohm_per_km'].values
    x_ohm_per_km = net.line['x_ohm_per_km'].values

    if seed is not None:
        rng = np.random.default_rng(seed)

    if noise > 0:
        # Do NOT update r/x_ohm_per_km in-place. We do not want to change
        # the underlying net object.
        noise_limit = r_ohm_per_km * noise
        r_ohm_per_km = r_ohm_per_km + rng.uniform(-noise_limit, noise_limit)

        noise_limit = x_ohm_per_km * noise
        x_ohm_per_km = x_ohm_per_km + rng.uniform(-noise_limit, noise_limit)

    if modify in ('perm', None):  # permute the line numbers
        net = copy.deepcopy(net)  # don't modify original net
        if modify == 'perm':
            order = np.zeros(n+1, dtype=int)
            order[1:] = rng.permutation(np.arange(1, n+1))
            net.line['from_bus'] = net.line['from_bus'].map(order.__getitem__)
            net.line['to_bus'] = net.line['to_bus'].map(order.__getitem__)

        r[net.line['from_bus'], net.line['to_bus']] = r_ohm_per_km
        r[net.line['to_bus'], net.line['from_bus']] = r_ohm_per_km
        x[net.line['from_bus'], net.line['to_bus']] = x_ohm_per_km
        x[net.line['to_bus'], net.line['from_bus']] = x_ohm_per_km
        G = pp.topology.create_nxgraph(net)

    elif modify in ('linear', 'rand'):
        if modify == 'linear':  # random undirected linear tree
            # substation (node 0) is not necessarily at one end of the path,
            # could be in the middle
            path = rng.permutation(n+1)
            G = nx.path_graph(path)
        else:
            G = nx.random_tree(n+1)  # uniformly random undirected tree

        r_sample = rng.choice(r_ohm_per_km, size=len(G.edges), replace=True)
        x_sample = rng.choice(x_ohm_per_km, size=len(G.edges), replace=True)
        for i, (e0, e1) in enumerate(G.edges):
            r[e0, e1] = r_sample[i]
            x[e0, e1] = x_sample[i]
            r[e1, e0] = r[e0, e1]
            x[e1, e0] = x[e0, e1]
    else:
        raise ValueError(f'Unexpected value for `modify`: {modify}')

    R, X = create_RX_from_rx(r, x, G, check_pd)
    return R, X

def create_RX_from_rx(r: np.ndarray, x: np.ndarray, G: nx.Graph,
                      check_pd: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Creates R,X matrices from line impedance matrices r and x.

    Args
    - r: np.array, shape [n+1, n+1], symmetric and entry-wise positive
    - x: np.array, shape [n+1, n+1], symmetric and entry-wise positive
    - G: nx.Graph, undirected graph, nodes are numbered {0, ..., n}
    - check_pd: bool, whether to assert that returned R,X are PD

    Returns: tuple (X, R)
    - X: np.array, shape [n, n], positive definite and entry-wise positive
    - R: np.array, shape [n, n], positive definite and entry-wise positive
    """
    n = r.shape[0] - 1

    R = np.zeros((n+1, n+1), dtype=float)
    X = np.zeros((n+1, n+1), dtype=float)

    # P_i
    paths = nx.shortest_path(G, source=0)  # node i => path from node 0 to i
    for i in range(1, n+1):
        for j in range(i, n+1):
            intersect = get_intersecting_path(paths[i], paths[j])
            R[i, j] = sum(r[e] for e in intersect)
            X[i, j] = sum(x[e] for e in intersect)
            R[j, i] = R[i, j]
            X[j, i] = X[i, j]

    R = 2 * R[1:, 1:]
    X = 2 * X[1:, 1:]

    assert np.all(R != np.inf)
    assert np.all(X != np.inf)

    if check_pd:
        assert is_pos_def(R)
        assert is_pos_def(X)
    return R, X

def get_intersecting_path(path1, path2):
    """Gets the intersection between two paths. Assumes that the paths only
    intersect in the beginning.

    Args
    - path1: list of int
    - path2: list of int

    Returns: list of tuple, edges in the intersecting path
    """
    ret = []
    for k in range(1, min(len(path1), len(path2))):
        u = path1[k]
        v = path2[k]
        if u == v:
            edge = (path1[k-1], u)
            ret.append(edge)
        else:
            break
    return ret

def is_pos_def(A: np.ndarray) -> bool:
    """Checks whether a matrix is positive definite.

    Args
    - A: np.array, matrix

    Returns: bool, true iff A>0
    """
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False