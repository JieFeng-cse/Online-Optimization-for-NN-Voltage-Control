import random
from numpy import linalg as LA
import numpy as np
from collections import defaultdict, deque

################################################################################################################
########################################## For Full Observation ################################################

def compute_level_sets(X_p, probing_nodes, tolerance = 3e-3):
    level_sets = {}
    for node_idx in range(len(probing_nodes)):
        x_m = X_p[:,node_idx]
        # Add a zero entry at the top of the vector xm
        x_m = np.insert(x_m,0,0.0)
        # Sort the entries of xm

        sorted_indices = np.argsort(x_m)
        sorted_values = x_m[sorted_indices]
        # print(probing_nodes[node_idx], sorted_values, sorted_indices)
        # assert False

        # Group the entries of x_m to find the level sets of node m
        # we assume a small tolerance to group close values together
        level_set = []
        current_level = [sorted_indices[0]]
        for i in range(1,len(sorted_values)):
            if np.abs(sorted_values[i] - sorted_values[i-1]) <= tolerance:
                current_level.append(sorted_indices[i])
            else:
                level_set.append(set(current_level))
                current_level = [sorted_indices[i]]
        
        # Append the last level
        level_set.append(set(current_level))

        # Assign the level set for this probing node
        level_sets[probing_nodes[node_idx]] = level_set
    return level_sets

def root_and_branch(probing_nodes, k, last_node, level_sets, line_resistances, R, original_probing_nodes, failed=False):
    """
    Stage s3: Recursively recover the topology based on probing nodes, level sets, and ancestors.
    
    Parameters:
    probing_nodes (list): List of probing nodes at the current level.
    k (int): The current depth in the recursion.
    level_sets (dict): Dictionary containing the level sets for each probing node.
    line_resistances (dict): Dictionary to store line resistances.
    R (numpy.ndarray): The R matrix being constructed.

    Returns:
    None: The function modifies the R matrix in place.
    """
    if not probing_nodes:
        return

    # Step 1: Identify the common k-depth ancestor by intersecting level sets
    for node in probing_nodes:
        if k>len(level_sets[node])-1:
            failed=True
            # print('failed')
            return 
    common_ancestor = list(set.intersection(*(set(level_sets[node][k]) for node in probing_nodes)))
    if len(common_ancestor)==0:
        failed=True
        # print('failed')
        return 
    else:
        common_ancestor = common_ancestor[0]

    # Step 2: Calculate the resistance between the common ancestor and its parent (using Eq (7))
    
        # Use the first node in the probing_nodes to compute the resistance
    
    if k>0:
        node = probing_nodes[0] # randomly pick one node
        node_id = original_probing_nodes.index(node)
        if last_node-1<0:
            resistance = R[common_ancestor-1, node_id]
        else:
            resistance = R[common_ancestor-1, node_id] - R[last_node-1, node_id]
        line_resistances[(last_node, common_ancestor)] = resistance

    # First, exclude the common ancestor {n} from further partitioning
    probing_nodes = [node for node in probing_nodes if node != common_ancestor]
    # print(probing_nodes)

    # Step 3: Partition the probing nodes based on their (k+1)-depth ancestor, following Proposition 2
    partitions = {}
    for node in probing_nodes:
        # Group nodes that share the same (k)-depth level set
        key = tuple(level_sets[node][k])
        if key not in partitions:
            partitions[key] = []
        partitions[key].append(node)

    # Recursively call root_and_branch on the partitions
    for partition_key, partition_nodes in partitions.items():
        root_and_branch(partition_nodes, k + 1, common_ancestor, level_sets, line_resistances, R, original_probing_nodes)


def compute_full_R(R, all_nodes, line_resistances, root=0):
    """
    Recursively compute the full R matrix (stage s3).

    Parameters
    ----------
    R : (N, N) numpy.ndarray
        Matrix to be filled in-place.
    all_nodes : list[int]
        List of *non-root* bus IDs.  Usually [1, 2, …, N].
    line_resistances : dict[(int, int) -> float]
        (Undirected) edge → resistance.  Either orientation is allowed.
    root : int, optional
        ID of the reference bus.  Default is 0.

    Returns
    -------
    numpy.ndarray
        The completed R matrix (same object as the input `R`).
    """
    # ------------------------------------------------------------------
    # 1) Build an undirected adjacency list
    adj = defaultdict(list)
    buses = set()
    for a, b in line_resistances:
        adj[a].append(b)
        adj[b].append(a)
        buses.update([a, b])

    # ------------------------------------------------------------------
    # 2) Breadth-first search from the root → one parent per node
    parent = {root: None}
    q = deque([root])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in parent:
                parent[v] = u
                q.append(v)

    # ------------------------------------------------------------------
    # 3) Pre-compute the ancestor set of every bus
    max_bus = max(buses)
    ancestor_sets = [set() for _ in range(max_bus + 1)]  # index by bus ID
    for bus in range(max_bus + 1):
        cur = bus
        while cur is not None:
            ancestor_sets[bus].add(cur)
            cur = parent.get(cur)

    # Optional: show what we found (skipping the root)
    # print([ancestor_sets[n] for n in all_nodes])

    # Helper to fetch the resistance of an undirected edge
    def r(c, d):
        return line_resistances.get((c, d),0.0)

    # ------------------------------------------------------------------
    # 4) Fill the R matrix using Eq. (3)
    for m in all_nodes:
        for n in all_nodes:
            common = ancestor_sets[m].intersection(ancestor_sets[n]) 
            R[m - 1, n - 1] = sum(r(c, d) for c in common for d in common)

    return R

def topology_recovery(X_p, probing_nodes,linear_system=False, N=12):
    R = np.zeros((N, N))
    all_nodes = np.arange(N)+1
    if linear_system:
        level_sets = compute_level_sets(X_p, probing_nodes, tolerance=1e-5)
    else:
        level_sets = compute_level_sets(X_p, probing_nodes)
    # print(level_sets)

    # Dictionary to store line resistances
    line_resistances = {}

    # Start the recursive recovery
    # probing_nodes = list(level_sets.keys())
    failed = False
    root_and_branch(probing_nodes, 0, 0, level_sets, line_resistances, X_p, probing_nodes.copy(), failed)
    if not failed:
        # Fill in the full R matrix using line resistances
        R = compute_full_R(R, all_nodes, line_resistances)

        return R, line_resistances
    else:
        return R, line_resistances


def build_sensitivity(probing_nodes, dis_list, v_list, cond_bound=1450,
                      ridge_init=1e-8, ridge_max=1e6, ridge_growth=10.0,
                      return_k=False):
    dis_matrix = np.concatenate(dis_list, axis=1)   # D : (m x T)
    v_matrix   = np.concatenate(v_list,  axis=1)    # V : (n x T)
    find = False
    X_P  = None
    k_used = 0.0

    m = len(probing_nodes)
    G = dis_matrix @ dis_matrix.T                   # G = D D^T

    # decide if we need ridge
    rank_ok = (np.linalg.matrix_rank(dis_matrix) >= m)
    use_ridge = True
    if rank_ok:
        try:
            use_ridge = (np.linalg.cond(G) >= cond_bound)
        except np.linalg.LinAlgError:
            use_ridge = True

    # build stabilized system (drop cond-bound gating once ridged)
    if use_ridge:
        s = np.linalg.svd(G, compute_uv=False)
        smax = s[0] if s.size else 0.0
        smin = s[-1] if s.size else 0.0
        # simple, monotone k search to separate tiny singular values
        k = ridge_init
        while (smin + k) <= 0 and k <= ridge_max:
            k *= ridge_growth
        Gk = G + k * np.eye(G.shape[0], dtype=G.dtype)
        k_used = k
    else:
        Gk = G

    try:
        if np.linalg.cond(Gk) < 1e4:
            d_inv_dis = dis_matrix.T @ np.linalg.inv(Gk)
            X_P = v_matrix @ d_inv_dis
            find = True
    except np.linalg.LinAlgError:
        pass

    if return_k:
        return X_P, find, k_used
    return X_P, find
    
def rls_update(u,v, X_rls, P_rls, delta = 1e2, lam = 0.995):
    # for simplicity, n parallel scalar-output RLS  share the same covariance.
    u = u.reshape(-1,1)
    v = v.reshape(-1,1)
    
    k = (P_rls@u)/ (lam + u.T@P_rls@u)
    e = v - X_rls @ u
    X_rls += e @ k.T
    
    P_rls = (P_rls - k @ u.T @ P_rls) / lam
    return X_rls, P_rls

def rls_update_fullP(u, v, X, P, lam=0.995):
    """
    Full vectorized RLS (big P in R^{(n*m) x (n*m)}).
    Model: v_t ≈ X u_t, where X ∈ R^{n x m}, u ∈ R^{m}, v ∈ R^{n}.

    Args:
        u:  (m,) or (m,1) control vector at time t
        v:  (n,) or (n,1) measured voltage diff
        X:  (n,m) current estimate of sensitivity matrix
        P:  (n*m, n*m) covariance matrix (full, not shared)
        lam: forgetting factor (0,1]

    Returns:
        X_new: (n,m)
        P_new: (n*m, n*m)
    """
    # Shapes
    u = np.asarray(u).reshape(-1, 1)    # (m,1)
    v = np.asarray(v).reshape(-1, 1)    # (n,1)
    n, m = X.shape
    nm = n * m
    assert P.shape == (nm, nm), "P must be (n*m, n*m)"

    # --- Residual r = v - X u  (n x 1)
    r = v - X @ u  # (n,1)

    # --- Build block views of P: P_{ij} ∈ R^{m x m}
    # P is arranged with row-blocks of size m and column-blocks of size m
    # Block (i,j) spans rows [i*m:(i+1)*m], cols [j*m:(j+1)*m]
    # We'll compute:
    #   S = lam*I_n + Φ P Φ^T, where S_ij = lam*δ_ij + u^T P_{ij} u
    #   B = P Φ^T ∈ R^{(nm) x n}, column j is concat_i [ P_{ij} u ] (i=0..n-1)

    # --- Compute S (n x n)
    S = lam * np.eye(n)
    for i in range(n):
        i_slice = slice(i*m, (i+1)*m)
        for j in range(n):
            j_slice = slice(j*m, (j+1)*m)
            P_ij = P[i_slice, j_slice]         # (m,m)
            S[i, j] += float(u.T @ P_ij @ u)   # scalar

    # --- Compute B = P Φ^T (nm x n)
    B = np.zeros((nm, n))
    for j in range(n):
        j_slice = slice(j*m, (j+1)*m)
        # Column j: concat over i of (P_{ij} @ u)
        col_blocks = []
        for i in range(n):
            i_slice = slice(i*m, (i+1)*m)
            P_ij = P[i_slice, j_slice]     # (m,m)
            col_blocks.append(P_ij @ u)    # (m,1)
        B[:, j:j+1] = np.vstack(col_blocks)  # (nm,1)

    # --- Gain K = B S^{-1}  (nm x n)
    # Solve rather than invert for stability
    # S is n x n; we solve S^T X^T = B^T  => X = B @ S^{-T}
    # but more directly: K = B @ np.linalg.inv(S)
    K = B @ np.linalg.inv(S)  # (nm, n)

    # --- Update omega (vec(X)): ω_new = ω + K r
    omega = X.reshape(nm, 1)            # (nm,1)
    omega_new = omega + K @ r           # (nm,1)
    X_new = omega_new.reshape(n, m)     # (n,m)

    # --- Update P: P_new = lam^{-1} (P - K Φ P)
    # Compute Φ P ∈ R^{n x nm}; row i is concat_j [ u^T P_{ij} ]
    PhiP = np.zeros((n, nm))
    for i in range(n):
        i_row = []
        for j in range(n):
            j_slice = slice(j*m, (j+1)*m)
            i_slice = slice(i*m, (i+1)*m)
            P_ij = P[i_slice, j_slice]          # (m,m)
            i_row.append((u.T @ P_ij).reshape(1, m))  # (1,m)
        PhiP[i, :] = np.hstack(i_row)  # (1, nm)

    P_new = (P - K @ PhiP @ P) / lam
    # Optional: symmetrize to tame numeric drift
    P_new = 0.5 * (P_new + P_new.T)

    return X_new, P_new
    
################################################################################################################
########################################## For Partial Observation ################################################


def compute_level_sets_paritial_o(X_p, probing_nodes, tolerance = 1e-6):
    level_sets = {}
    for node_idx in range(len(probing_nodes)):
        x_m = X_p[:,node_idx]
        # add a zero entry at the top of the vector xm
        x_m = np.insert(x_m,0,0.0)
        #sort the entries of xm

        sorted_indices = np.argsort(x_m)
        sorted_values = x_m[sorted_indices]

        #group the entries of x_m to find the level sets of node m
        #we assume a small tolerance to group close values together
        tolerance = tolerance #1e-3 for nonlinear, 1e-6 for linear
        level_set = []
        current_level = [sorted_indices[0]]
        for i in range(1,len(sorted_values)):
            if np.abs(sorted_values[i] - sorted_values[i-1]) <= tolerance:
                current_level.append(probing_nodes[sorted_indices[i]-1])
            else:
                level_set.append(set(current_level))
                current_level = [probing_nodes[sorted_indices[i]-1]]
        
        # append the last level
        level_set.append(set(current_level))

        # Assign the level set for this probing node
        level_sets[probing_nodes[node_idx]] = level_set
    return level_sets

def root_and_branch_partial(probing_nodes, k, last_node, level_sets, line_resistances, R, original_probing_nodes, internal_identifiable_nodes):
    """
    Stage s3: Recursively recover the topology based on probing nodes, level sets, and ancestors.
    
    Parameters:
    probing_nodes (list): List of probing nodes at the current level.
    k (int): The current depth in the recursion.
    level_sets (dict): Dictionary containing the level sets for each probing node.
    line_resistances (dict): Dictionary to store line resistances.
    R (numpy.ndarray): The R matrix being constructed.

    Returns:
    None: The function modifies the R matrix in place.
    """
    if not probing_nodes:
        return

    # Step 1: Identify the common k-depth ancestor by intersecting level sets
    candidate_parent = None
    for node in probing_nodes:
        if level_sets[node][k] == set(probing_nodes):
            candidate_parent = node
            break
    if not candidate_parent:
        candidate_parent = internal_identifiable_nodes[0]
        internal_identifiable_nodes.pop(0)


    # Step 2: Calculate the resistance between the common ancestor and its parent (using Eq (7))
    
        # Use the first node in the probing_nodes to compute the resistance
    
    if k>1:
        for node in probing_nodes:
            node_s1 = next(iter(level_sets[node][k-1]))
            node_s2 = next(iter(level_sets[node][k]))
            node_id_m = original_probing_nodes.index(node)
            node_id_s1 = original_probing_nodes.index(node_s1)
            node_id_s2 = original_probing_nodes.index(node_s2)
            resistance = R[node_id_s2,node_id_m]-R[node_id_s1,node_id_m]
            break
        line_resistances[(last_node, candidate_parent)] = resistance

    # First, exclude the common ancestor {n} from further partitioning
    probing_nodes = [node for node in probing_nodes if node != candidate_parent]
    # print(probing_nodes)

    # Step 3: Partition the probing nodes based on their (k+1)-depth ancestor, following Proposition 2
    partitions = {}
    for node in probing_nodes:
        # Group nodes that share the same (k)-depth level set
        key = tuple(level_sets[node][k])
        if key not in partitions:
            partitions[key] = []
        partitions[key].append(node)

    # Recursively call root_and_branch on the partitions
    for partition_key, partition_nodes in partitions.items():
        root_and_branch_partial(partition_nodes, k + 1, candidate_parent, level_sets, line_resistances, R, original_probing_nodes, internal_identifiable_nodes)

def sub_to_root_resistance(probing_nodes, ancestor_sets, R_PP, line_resistances, delta=0.0001):
    ite_num = 0
    tmp_x = 0
    line_resistances_new = line_resistances.copy()
    #set delta to 0.001 for nonlinear dynamics
    for id_1, node in enumerate(probing_nodes):
        for id_2, node_2 in enumerate(probing_nodes):
            if len(ancestor_sets[node].intersection(ancestor_sets[node_2])) == 1:
                if np.abs(tmp_x - R_PP[id_1, id_2]) < delta and ite_num > 0:
                    line_resistances_new[(0, 1)] = R_PP[id_1, id_2]
                    return line_resistances_new
                ite_num += 1
                tmp_x = R_PP[id_1, id_2]
    return None  # If no match is found

def compute_partial_R(R, all_revealed_nodes, line_resistances, ancestor_sets):
    """
    Stage s3: Recursively compute the full R matrix using the probing nodes, level sets, and line resistances.
    
    Parameters:
    R (numpy.ndarray): The full R matrix being constructed (N x N).
    probing_nodes (list): List of probing node indices.
    level_sets (dict): The computed level sets from Stage s2.
    line_resistances (dict): Dictionary containing the resistance values for each node pair (computed from Step 2).
    
    Returns:
    numpy.ndarray: The full R matrix.
    """
    # Iterate over each node and fill in the R matrix using Eq (3)



    for m in all_revealed_nodes: # except the root
        for n in all_revealed_nodes:
            # Find common ancestors by comparing level sets across depths
            common_ancestors = []
            
            ancestors_m = ancestor_sets[m]
            ancestors_n = ancestor_sets[n]

            common_ancestors = ancestors_m.intersection(ancestors_n)

            ind_m = all_revealed_nodes.index(m)
            ind_n = all_revealed_nodes.index(n)
            R[ind_m, ind_n] = sum(line_resistances.get((c, d), 0) for c in common_ancestors for d in common_ancestors)
    
    return R

from collections import defaultdict
def find_ancestor(probing_nodes, line_resistances):
    tree = defaultdict(list)
    for line_key, line_res in line_resistances.items():
        parent, child = line_key
        tree[parent].append(child)
    # print(tree)
    
    ancestor_sets = {}
    def dfs(node, ancestors):
        current_ancestors = ancestors.union({node})
        ancestor_sets[node] = current_ancestors

        for child in tree[node]:
            dfs(child, current_ancestors)
    
    child = set(line_key[1] for line_key, line_res in line_resistances.items())
    roots = set(probing_nodes)-child
    # print('child', child)
    for root in roots:
        dfs(root,set())
    return ancestor_sets