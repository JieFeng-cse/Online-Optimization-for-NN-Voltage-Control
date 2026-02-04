import itertools
import numpy as np
from collections import deque
from topo_estimate import compute_full_R
import gurobipy as gp
from gurobipy import GRB

# =============================================================================
# OVERVIEW (Topology change detection + sparse line-change estimation)
#
# This file implements a real-time pipeline to detect distribution-network
# topology changes (line add/remove) from streaming measurements, then recover
# the updated topology.
#
# Big picture:
#   1) Monitor stage: watch a scalar error metric over time and detect a “jump”.
#      - If the jump is small/temporary -> treat as load/power change.
#      - If the jump persists -> treat as a topology change event.
#
#   2) Row activity stage: from the newest residual matrix R(t), detect which
#      buses/rows are “active” (abnormal) and keep a short sliding window.
#      - A row is “stable active” if it stays active for most of the window.
#      - These stable rows define an “involved bus set”.
#
#   3) Candidate edge construction: build all candidate line pairs among the
#      involved buses (complete graph on involved set). Each candidate line
#      corresponds to one column of an incidence-like matrix E.
#
#   4) Sparse regression to pick changed lines:
#      Goal: find a sparse vector gamma over candidate lines such that the
#      predicted residual R_hat matches the observed residual R.
#
#      Two main solvers appear:
#        - QP + L1 penalty (LASSO-style): quadratic data-fit + λ||gamma||_1.
#          Implemented via _solve_l1_qp_gurobi() and the reweighted wrapper
#          solve_gamma_l1_qp_reweighted().
#        - LAD-LASSO (robust): L1 residual loss + λ||gamma||_1 as a single LP,
#          implemented in solve_lad_lasso().
#
#      Practical enhancements:
#        - λ auto-scan / model selection based on error tolerance and sparsity.
#        - “stable support” voting across nearby λ values for robustness.
#        - Topology feasibility filters (physics constraints):
#            * sign constraints (+ add / − remove) consistent with previous topo
#            * net add/remove counts match
#            * resulting graph must remain a tree (_is_tree)
#        - Debias refit on chosen support:
#            * solves a small weighted least-squares on the selected support
#            * uses Huber IRLS for robustness
#            * adds “endogeneity-aware” per-column weights (downweight columns
#              where regressor aligns with error)
#
# Outputs:
#   - gamma: sparse magnitudes for changed candidate lines (sign indicates add/remove)
#   - Delta_Xinv (or X update proxy): reconstructed matrix from gamma and E
#   - R_hat: reconstructed residual
#   - X_new + updated line dictionary when a consistent topology change is found
#
# Key classes/functions:
#   - topology_detection_lasso:
#       * monitor(err): robust jump detector
#       * update(R, v): row activity tracking with sliding window
#       * est_line_change(ids): build candidates + quick rank check (SVD)
#       * recover_full_topo(...): runs sparse estimation + physics checks + update
#   - _solve_l1_qp_gurobi(): QP with L1 penalty (LASSO)
#   - solve_lad_lasso(): LAD loss + L1 penalty (LP)
#   - debias(): robust refit on selected support (Huber IRLS + endog weights)
# =============================================================================

def edge_vector(n, a, b):
    """Return incidence vector e_a - e_b  (a,b are node indices 0…n-1)."""
    u = np.zeros(n)
    u[a] = 1.0
    u[b] = -1.0
    return u

def estimate_gamma_iv_reconstruct(
    E_org, V_obs, R_hist_T, invX,
    ridge_A=1e-9, ridge_iv=1e-8,      # softer shrink
    center=True,
    min_instr=1e-8,                   # stronger instrument filter
    keep_frac=0.8,                    # keep top 80% strongest columns
    min_corr=0.2,                     # screen poor alignment per-row
    normalize_by_W=True,
    gamma_prior=None, prior_lambda=0.0,
    sign_bounds=None, bounds=None     # e.g., ('nonneg'), or (gmin,gmax)
):
    A = E_org.T @ E_org
    Y = E_org.T @ R_hist_T
    Z = E_org.T @ V_obs

    # Reconstruct V_pred = V_obs + solve(invX, R)
    V_pred = V_obs + np.linalg.solve(invX, R_hist_T)
    W = E_org.T @ V_pred

    # S = (E^T E)^{-1} E^T R
    S = np.linalg.solve(A + ridge_A*np.eye(A.shape[0]), Y)

    if center:
        Z = Z - Z.mean(axis=1, keepdims=True)
        W = W - W.mean(axis=1, keepdims=True)
        S = S - S.mean(axis=1, keepdims=True)

    # ---- Global column screening by instrument strength ----
    strength = np.linalg.norm(W, axis=0)
    keep_k = max(1, int(round(keep_frac * W.shape[1])))
    idx = np.argpartition(strength, -keep_k)[-keep_k:]
    idx = idx[strength[idx] >= min_instr]
    if idx.size == 0:  # fall back if too strict
        idx = np.argpartition(strength, -keep_k)[-keep_k:]

    Z, W, S = Z[:, idx], W[:, idx], S[:, idx]

    # ---- Per-column normalization ----
    if normalize_by_W:
        scale = np.linalg.norm(W, axis=0) + 1e-12
        Z, W, S = Z/scale, W/scale, S/scale

    k = Z.shape[0]
    gamma = np.zeros(k)
    for i in range(k):
        zi, wi, si = Z[i, :].copy(), W[i, :].copy(), S[i, :].copy()

        # Row-wise alignment screen
        if zi.size >= 3 and min_corr is not None:
            zc, wc = zi - zi.mean(), wi - wi.mean()
            zstd = np.linalg.norm(zc); wstd = np.linalg.norm(wc)
            corr = 0.0 if zstd == 0 or wstd == 0 else (zc @ wc) / (zstd * wstd)
            if corr < min_corr:
                good = (zi * wi) > 0
                if good.any():
                    zi, wi, si = zi[good], wi[good], si[good]

        if zi.size == 0:
            gamma[i] = 0.0
            continue

        mu = 0.0 if gamma_prior is None else float(gamma_prior[i])
        num = float(si @ wi) + prior_lambda * mu
        den = float(zi @ wi) + prior_lambda + ridge_iv

        gi = 0.0 if abs(den) < 1e-12 else num / den

        # Apply bounds
        if sign_bounds == 'nonneg': gi = max(gi, 0.0)
        if sign_bounds == 'nonpos': gi = min(gi, 0.0)
        if bounds is not None:
            gmin, gmax = bounds
            gi = min(max(gi, gmin), gmax)

        gamma[i] = gi

    Delta_Xinv = E_org @ np.diag(gamma) @ E_org.T
    R_hat = Delta_Xinv @ V_obs
    return gamma, Delta_Xinv, R_hat

def _is_tree(n_nodes: int, edges) -> bool:

    # A tree must have exactly n_nodes‑1 edges
    if len(edges) != n_nodes - 1:
        return False

    # Normalise to 0‑based indexing for the DSU algorithm
    min_node = min(min(u, v) for u, v in edges)
    if min_node == 1:  # convert 1‑based → 0‑based
        edges_norm = {(u - 1, v - 1) for u, v in edges}
    else:              # assume already 0‑based
        edges_norm = edges

    parent = list(range(n_nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    # Union‑Find to detect cycles
    for u, v in edges_norm:
        ru, rv = find(u), find(v)
        if ru == rv:            # cycle detected
            return False
        parent[rv] = ru         # union

    # Check connectivity: all nodes share the same root
    root = find(0)
    return all(find(i) == root for i in range(1, n_nodes))




def _solve_l1_qp_gurobi(
    E_sub, V, R, lam, verbose=False,
    weights=None,
    nonneg=False,
    ridge_scale=1e-10,
    signs=None,
):  
    """
    Solve an L1-regularized quadratic program using Gurobi.

    This function solves a LASSO-style problem with a quadratic data-fit term
    and an L1 penalty to promote sparsity:

        min_gamma  0.5 * gamma^T H gamma - b^T gamma + lam * ||gamma||_1

    Internally, the problem is rescaled for numerical stability and rewritten
    in standard QP form with auxiliary variables to represent the L1 norm.

    Parameters
    ----------
    E_sub : (n, k0) ndarray
        Submatrix of edge/incidence features. Each column corresponds to a candidate component.
    V : (n, p) ndarray
        Voltage (or state) measurements.
    R : (n, p) ndarray
        Residual or response matrix.
    lam : float
        L1 regularization strength (controls sparsity).
    verbose : bool, optional
        If True, enable Gurobi solver output.
    weights : (k0,) ndarray or None, optional
        Optional per-coordinate weights for the L1 penalty.
    nonneg : bool, optional
        If True, enforce gamma >= 0.
    ridge_scale : float, optional
        Small ridge added to the quadratic term for numerical stability.
    signs : (k0,) ndarray or None, optional
        Sign constraints on gamma:
            +1 enforces gamma_i >= 0,
            -1 enforces gamma_i <= 0,
            0 means unconstrained.

    Returns
    -------
    gammaS_val : (k0,) ndarray
        Estimated sparse coefficient vector.
    Delta_Xinv_sub : (n, n) ndarray
        Reconstructed matrix from the sparse solution.
    R_hat : (n, p) ndarray
        Reconstructed response using the estimated coefficients.
    lam_used : float
        Regularization value actually used by the solver.
    """


    n, k0 = E_sub.shape
    _, p  = V.shape

    # ---------- Precompute ----------
    Zs = E_sub.T @ V           # (k0, p)
    Ys = E_sub.T @ R           # (k0, p)
    GE = E_sub.T @ E_sub       # (k0, k0)
    GZ = Zs @ Zs.T             # (k0, k0)
    H  = GE * GZ               # Hadamard
    H  = 0.5 * (H + H.T)       # symmetrize
    b  = (Zs * Ys).sum(axis=1) # (k0,)

    # ---------- Standardize: beta = D * gamma ----------
    alpha = np.linalg.norm(E_sub, axis=0) * np.linalg.norm(Zs, axis=1)
    alpha[alpha == 0.0] = 1.0
    Dinv = 1.0 / alpha
    H = (Dinv[:, None] * H) * Dinv[None, :]
    b = b * Dinv

    # ---------- Make H safely PSD (NEW) ----------
    # ridge
    trH = float(np.trace(H))
    if trH > 0.0 and k0 > 0:
        H[np.diag_indices(k0)] += ridge_scale * trH / max(1, k0)
    # eigen-clip (small k0): guarantees PSD numerically
    if k0 <= 2000:  # cheap enough
        w, Q = np.linalg.eigh(H)
        floor = max(1e-14, 1e-12 * (w.max() if np.isfinite(w.max()) else 1.0))
        w = np.maximum(w, floor)
        H = (Q * w) @ Q.T       # Q @ diag(w) @ Q.T

    # Lambda scale
    lam_used = lam if lam is not None else (0.1 * (float(np.max(np.abs(b))) if k0 > 0 else 0.0))

    # Weights
    if weights is None:
        wgt = np.ones(k0, dtype=float)
    else:
        wgt = np.asarray(weights, dtype=float)
        if wgt.shape != (k0,):
            raise ValueError("weights must have shape (k0,)")

    # ---------- Build & solve QP ----------
    def solve_once(H_mat, b_vec, method=2, extra_ridge=0.0):
        m = gp.Model("sparse_gamma_reduced")
        if not verbose:
            m.Params.OutputFlag = 0
        m.Params.Method         = method   # 2=barrier, 1=dual simplex
        m.Params.Crossover = 0
        m.Params.NumericFocus = 3
        m.Params.Presolve = 2
        m.Params.ScaleFlag = 2
        m.Params.BarHomogeneous = 1
        m.Params.OptimalityTol = 1e-9
        m.Params.FeasibilityTol = 1e-9
        m.Params.BarConvTol = 1e-12

        # bounds on beta (beta has same sign as gamma since alpha>0)
        lb = np.full(k0, -gp.GRB.INFINITY, dtype=float)
        ub = np.full(k0,  gp.GRB.INFINITY, dtype=float)

        if nonneg:
            # nonneg on gamma means gamma >= 0  ==> beta >= 0 / Dinv = 0
            lb[:] = np.maximum(lb, 0.0)

        if signs is not None:
            s = np.asarray(signs, dtype=int)
            if s.shape != (k0,):
                raise ValueError("signs must have shape (k0,), values in {-1,0,+1}")
            # gamma_i >= 0  ==> beta_i >= 0
            lb[s == +1] = np.maximum(lb[s == +1], 0.0)
            # gamma_i <= 0  ==> beta_i <= 0
            ub[s == -1] = np.minimum(ub[s == -1], 0.0)

        beta = m.addMVar(k0, lb=lb, ub=ub, name="beta")
        t    = m.addMVar(k0, lb=0.0, name="t")
        bvars = beta.tolist()

        # |beta| <= t
        m.addConstr(beta - t <= 0, name="abs_pos")
        m.addConstr(-beta - t <= 0, name="abs_neg")

        # Quadratic: 0.5 * beta^T (H + extra_ridge*I) beta
        quad = gp.QuadExpr()
        if extra_ridge > 0.0:
            H_mat = H_mat.copy()
            H_mat[np.diag_indices(k0)] += extra_ridge

        rows, cols = np.triu_indices(k0)
        for i, j in zip(rows.tolist(), cols.tolist()):
            hij = float(H_mat[i, j])
            if hij != 0.0:
                quad += (0.5 * hij if i == j else hij) * bvars[i] * bvars[j]

        # Linear: - b^T beta + lam * sum w_i t_i
        lin = gp.LinExpr()
        for i, bi in enumerate(b_vec.tolist()):
            if bi != 0.0:
                lin.addTerms(-float(bi), bvars[i])
        lin += lam_used * gp.quicksum(float(wgt[i]) * t[i] for i in range(k0))

        m.setObjective(quad + lin, GRB.MINIMIZE)
        m.optimize()
        return m, beta, t

    # First attempt: barrier with PSD-clipped H
    m, beta_var, t_var = solve_once(H, b, method=2, extra_ridge=0.0)

    # Accept OPTIMAL or (SUBOPTIMAL/NUMERICAL) with incumbent (NEW)
    acceptable_statuses = {GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.NUMERIC}
    if m.Status not in acceptable_statuses or m.SolCount == 0:
        # Retry once: add stronger ridge and switch to dual simplex
        strong_ridge = max(1e-8, ridge_scale * 1e3)
        m, beta_var, t_var = solve_once(H, b, method=1, extra_ridge=strong_ridge)
        if (m.Status not in acceptable_statuses) or (m.SolCount == 0):
            raise RuntimeError(f"Gurobi status {m.Status}: no usable solution (SolCount={m.SolCount}).")

    # beta_val   = beta_var.X.copy()
    # gammaS_val = beta_val * Dinv  # map back

    beta_val = beta_var.X.copy()
    beta_val[signs == +1] = np.maximum(beta_val[signs == +1], 0.0)
    beta_val[signs == -1] = np.minimum(beta_val[signs == -1], 0.0)
    gammaS_val = beta_val * Dinv

    # Outputs
    if np.count_nonzero(gammaS_val) == 0:
        Delta_Xinv_sub = np.zeros((n, n), dtype=E_sub.dtype)
        R_hat = np.zeros_like(V)
    else:
        Delta_Xinv_sub = (E_sub * gammaS_val[None, :]) @ E_sub.T
        R_hat = E_sub @ (gammaS_val[:, None] * Zs)
    return gammaS_val, Delta_Xinv_sub, R_hat, lam_used


def solve_gamma_l1_qp_reweighted(
    E_sub, V, R,
    lam=None,
    verbose=False,
    nonneg=False,
    reweight_iters=2,      # 0 = plain L1; >=1 does adaptive L1 (often increases sparsity)
    reweight_eps=1e-3,     # stability term for weights = 1/(|gamma|+eps)
    ridge_scale=1e-10,
    signs = None,
):
    """
    Convenience wrapper that:
      1) runs (adaptive) re-weighted L1 for 'reweight_iters' outer loops;
      2) optional hard-threshold to 'target_k' largest |gamma|;
      3) debias refit (lam=0) on the selected support.

    Returns:
      gamma_S, Delta_Xinv_sub, R_hat
    """
    k0 = E_sub.shape[1]
    w = np.ones(k0, dtype=float)
    last_gamma = np.zeros(k0, dtype=float)
    lam_used_final = None

    for it in range(max(1, reweight_iters)):
        gammaS, _, _, lam_used = _solve_l1_qp_gurobi(
            E_sub, V, R, lam=lam,
            verbose=verbose, weights=w, nonneg=nonneg, ridge_scale=ridge_scale, signs=signs,
        )
        lam_used_final = lam_used
        last_gamma = gammaS.copy()
        # adaptive weights for next round
        w = 1.0 / (np.abs(gammaS) + reweight_eps)

    gammaS = last_gamma

    # Final products
    if np.count_nonzero(gammaS) == 0:
        Delta_Xinv_sub = np.zeros((E_sub.shape[0], E_sub.shape[0]), dtype=E_sub.dtype)
        R_hat = np.zeros_like(V)
    else:
        Zs = E_sub.T @ V
        Delta_Xinv_sub = (E_sub * gammaS[None, :]) @ E_sub.T
        R_hat = E_sub @ (gammaS[:, None] * Zs)

    return gammaS, Delta_Xinv_sub, R_hat


def estimate_gamma_l1_screened_auto(
    E, V, R,
    lam=None,                  # None/'auto' -> build data-driven path
    err_tol=1.5e-2,            # used if auto_err_tol=False
    lam_up_multipliers=(1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 50.0),
    reweight_iters=2,
    reweight_eps=1e-3,
    nonneg=False,
    verbose=False,
    signs=None,
    topo_dict=None,
    changed_lines=None,
    eps_gamma=0.5,
    # path knobs
    k_target=2,
    auto_err_tol=True,
    lam_span=30.0,
    n_points=7,            # ↓ 9 -> 7 (fewer QPs, still robust)
    # SIMPLE knobs for pair selection
    max_pairs=None,        # None => keep adding improving pairs; or int cap
    topL_each=6,           # try at most L strongest + and L strongest −
    min_rel_improve=1e-4,  # fractional error drop to accept a pair
    pair_budget=6,         # NEW: test at most K candidate (+,-) pairs per round
):
    """
    Returns: gamma (k,), Delta_Xinv (n,n), R_hat (n,p)
    """
    import numpy as np

    def rel_err(Rhat):
        num = np.linalg.norm(R - Rhat, ord='fro')
        den = max(1e-18, np.linalg.norm(R, ord='fro'))
        return num / den

    def _build_data_path(E, V, R, lam_span, n_points):
        Z = E.T @ V
        Y = E.T @ R
        b = (Z * Y).sum(axis=1)         # (k,)
        lam_max = float(np.linalg.norm(b, ord=np.inf))
        if not np.isfinite(lam_max) or lam_max <= 0:
            lam_max = 1.0
        lam_min = lam_max / max(1.0, float(lam_span))
        t = np.linspace(0.0, 1.0, int(n_points))
        return lam_max * (lam_min / lam_max) ** t

    # ---- λ candidates ----
    if lam is None or lam == 'auto':
        lam_path = _build_data_path(E, V, R, lam_span=float(lam_span), n_points=int(n_points))
    else:
        base = float(lam)
        lam_path = np.array([base * float(m) for m in lam_up_multipliers], dtype=float)

    # ---- evaluate λ path (store all to reuse later) ----
    cand = []  # (nnz, err, lam_try, gS, Dsub, Rhat)
    for lam_try in lam_path:
        gS, Dsub, Rhat = solve_gamma_l1_qp_reweighted(
            E, V, R,
            lam=lam_try,
            verbose=verbose,
            nonneg=nonneg,
            reweight_iters=reweight_iters,
            reweight_eps=reweight_eps,
            ridge_scale=1e-9,
            signs=signs,
        )
        nnz = int(np.count_nonzero(gS))
        err = rel_err(Rhat)
        cand.append((nnz, err, lam_try, gS, Dsub, Rhat))
        if verbose:
            print(f"[λ-scan] λ={lam_try:.3e} nnz={nnz:2d} rel_err={err:.4e}")

    # ---- auto err_tol from path near k_target (if enabled) ----
    if auto_err_tol:
        band = [c[1] for c in cand if abs(c[0] - k_target) <= 1]
        if len(band) == 0:
            errs = np.array([c[1] for c in cand])
            err_tol_eff = float(np.percentile(errs, 30))
        else:
            err_tol_eff = min(band) * 1.05
    else:
        err_tol_eff = err_tol

    # ---- pick best model given err_tol_eff and sparsity target ----
    feasible = [c for c in cand if c[1] <= err_tol_eff]
    if len(feasible) > 0:
        feasible.sort(key=lambda x: (abs(x[0] - k_target), x[1], -x[2]))  # nnz closeness, error, prefer larger λ
        best = feasible[0]
    else:
        def score(c):
            nnz, err, lam_try = c[0], c[1], c[2]
            over = max(0, nnz - k_target)
            return (err, over)
        cand.sort(key=score)
        best = cand[0]

    # Best solution and its index in the path
    nnz, err, lam_best, gS_best, _, Rhat_best = best
    j_best = int(np.argmin([abs(c[2] - lam_best) for c in cand]))

    # ======================================================================
    # Candidate pool = prev/best/next stability (reuse path only)
    # ======================================================================
    idxs = [j for j in [j_best - 1, j_best, j_best + 1] if 0 <= j < len(cand)]
    G = np.vstack([cand[j][3] for j in idxs])               # (m<=3, k)
    k = E.shape[1]

    # simple vote threshold (median-based)
    g_abs_best = np.abs(gS_best)
    nz = g_abs_best[g_abs_best > 0]
    med = float(np.median(nz)) if nz.size else 0.0
    eps_vote = max(float(eps_gamma), 0.15 * med)            # lighter than q75, faster

    votes = (np.abs(G) > eps_vote).sum(axis=0)
    stable_idx = np.where(votes >= max(2, len(idxs)//2 + 1))[0]
    if stable_idx.size == 0:
        stable_idx = np.where(g_abs_best > eps_vote)[0]     # fallback: support at λ*

    # topo-feasible majority sign per index
    def _maj_sign(i):
        s = np.sign(G[:, i]).sum()
        if s == 0: s = np.sign(gS_best[i])
        return 1 if s > 0 else (-1 if s < 0 else 0)

    if topo_dict is not None and changed_lines is not None and stable_idx.size > 0:
        keep = []
        for i in stable_idx:
            sgn = _maj_sign(i)
            u0, v0 = changed_lines[i]
            if sgn < 0 and (u0 + 1, v0 + 1) in topo_dict:     # feasible removal
                keep.append(i)
            if sgn > 0 and (u0 + 1, v0 + 1) not in topo_dict: # feasible addition
                keep.append(i)
        stable_idx = np.array(keep, dtype=int) if len(keep) else np.array([], dtype=int)

    # rank by hybrid score: avg|γ| (stability) + residual correlation (cheap KKT)
    if stable_idx.size > 0:
        avg_abs = np.mean(np.abs(G[:, stable_idx]), axis=0)
        sgns = np.array([_maj_sign(i) for i in stable_idx])

        # residual at the best λ model (fast to compute)
        M = R - Rhat_best
        Z = E.T @ V
        Ei_norm = np.linalg.norm(E, axis=0) + 1e-12
        Zi_norm = np.linalg.norm(Z, axis=1) + 1e-12
        t = (E.T @ M)                           # (k, p)
        t = np.einsum('kp,kp->k', t, Z)         # row-wise dot
        grad = -t                                # KKT-ish
        kkt = np.abs(grad[stable_idx]) / (Ei_norm[stable_idx] * Zi_norm[stable_idx] + 1e-12)

        # normalize each score to [0,1] and blend (α=0.6 favours stability)
        def _norm(x):
            x = np.asarray(x, float)
            lo, hi = x.min(), x.max()
            return (x - lo) / (hi - lo + 1e-12)
        s_stab = _norm(avg_abs)
        s_kkt  = _norm(kkt)
        s = 0.6 * s_stab + 0.4 * s_kkt

        pos_pool = stable_idx[sgns > 0]
        neg_pool = stable_idx[sgns < 0]
        if pos_pool.size == 0 or neg_pool.size == 0:
            pos_pool = np.array([], dtype=int)
            neg_pool = np.array([], dtype=int)
        else:
            # sort within each sign by the blended score
            s_pos = s[sgns > 0]
            s_neg = s[sgns < 0]
            pos_pool = pos_pool[np.argsort(-s_pos)]
            neg_pool = neg_pool[np.argsort(-s_neg)]
    else:
        pos_pool = np.array([], dtype=int)
        neg_pool = np.array([], dtype=int)

    # ======================================================================
    # Greedy pair selection with a tiny pair budget (fast; debias on tiny sets)
    # ======================================================================
    def _debias(supp):
        return debias(np.array(supp, dtype=int), E, V, R, signs)

    # early exit if the best path model already “good enough”
    err_best = rel_err(Rhat_best)
    if err_best <= err_tol_eff * 1.02:
        # stick to λ* support and debias (keeps behaviour consistent)
        supp0 = np.where(np.abs(gS_best) > eps_vote)[0]
        # topo-feasible filter on supp0 (same logic)
        if topo_dict is not None and changed_lines is not None and supp0.size > 0:
            mask = np.ones(len(supp0), dtype=bool)
            for idx, i in enumerate(supp0):
                u0, v0 = changed_lines[i]
                if gS_best[i] < 0 and (u0 + 1, v0 + 1) not in topo_dict: mask[idx] = False
                if gS_best[i] > 0 and (u0 + 1, v0 + 1)     in topo_dict: mask[idx] = False
            supp0 = supp0[mask]
        return _debias(supp0)

    supp = np.array([], dtype=int)
    gamma0 = np.zeros(k, dtype=E.dtype)
    Delta0 = np.zeros((E.shape[0], E.shape[0]), dtype=E.dtype)
    Rhat0  = np.zeros_like(R)
    err0   = rel_err(Rhat0)

    Lp = int(min(topL_each, pos_pool.size))
    Ln = int(min(topL_each, neg_pool.size))
    pos_try = pos_pool[:Lp]
    neg_try = neg_pool[:Ln]

    # build a small candidate list of pairs by interleaving (limits O(L²))
    def interleave(a, b):
        out = []
        for i in range(max(len(a), len(b))):
            if i < len(a): out.append(('p', a[i]))
            if i < len(b): out.append(('n', b[i]))
        return out

    pos_list = [i for t,i in interleave(pos_try, [])]
    neg_list = [i for t,i in interleave(neg_try, [])]

    improved = True
    while improved and len(pos_list) > 0 and len(neg_list) > 0:
        improved = False
        best = None
        best_err = err0

        # examine only the first `pair_budget` interleaved pairs
        tested = 0
        for ip in pos_list:
            for ineg in neg_list:
                trial = np.unique(np.concatenate([supp, [ip, ineg]])).astype(int)
                g_t, D_t, Rhat_t = _debias(trial)
                err_t = rel_err(Rhat_t)
                tested += 1
                if err_t < best_err * (1.0 - float(min_rel_improve)):
                    best_err = err_t
                    best = (trial, g_t, D_t, Rhat_t)
                if tested >= int(pair_budget):
                    break
            if tested >= int(pair_budget):
                break

        if best is not None:
            supp, gamma0, Delta0, Rhat0 = best
            err0 = best_err
            improved = True
            if (max_pairs is not None) and (supp.size // 2 >= int(max_pairs)):
                break
            # shrink lists to avoid retesting same pair many times
            pos_list = [i for i in pos_list if i not in supp]
            neg_list = [i for i in neg_list if i not in supp]
        # else: no improving pair ⇒ stop

    # final debias (covers the case where nothing improved)
    gamma, Delta, Rhat = _debias(supp)
    return gamma, Delta, Rhat


def debias( supp, E, V, R, signs,
            huber_delta=0.001,          # Huber transition (in ||column residual||_2 units)
            irls_max_iter=3,            # few IRLS steps are enough
            irls_eps=1e-12,             # numerical floor to avoid /0
            debias_ridge=1e-10,
    ):
    """
    Robust debias on chosen support with endogeneity-aware column weighting.
    Inputs/outputs unchanged.

    Minimal modification:
      - Precompute per-column 'endogeneity' weights based on the cosine between
        Z_s[:,j] = (E_s^T V)[:,j] and Y_s[:,j] = (E_s^T R)[:,j].
      - Multiply these into the existing Huber IRLS weights to downweight
        columns where regressors align with error (likely endogeneity).
    """
    n, k = E.shape

    if supp.size == 0:
        gamma = np.zeros(k, dtype=E.dtype)
        Delta = np.zeros((n, n), dtype=E.dtype)
        Rhat = np.zeros_like(V)
        return gamma, Delta, Rhat

    E_s = E[:, supp]                 # n x s
    Z_s = E_s.T @ V                  # s x p
    Y_s = E_s.T @ R                  # s x p
    GE_s = E_s.T @ E_s               # s x s  (geometry in E-space)

    # ---------- NEW: Endogeneity-aware fixed column weights ----------
    # Measure cosine similarity per column between Z_s[:,j] and Y_s[:,j].
    # Large |cos| => likely endogeneity (regressor aligned with error) => downweight.
    # o_j = 1 / (1 + alpha * cos^2), alpha >= 0
    alpha_endog = 10.0
    z_norm = np.linalg.norm(Z_s, axis=0)          # length p
    y_norm = np.linalg.norm(Y_s, axis=0)          # length p
    denom = (z_norm * y_norm) + 1e-18
    cos_zy = np.sum(Z_s * Y_s, axis=0) / denom    # safe cosine per column
    o_weight = 1.0 / (1.0 + alpha_endog * (cos_zy ** 2))  # shape (p,)

    # helper to build H and b with column weights w (shape p,)
    def build_H_b(w):
        # weighted Z_s Z_s^T: Z * diag(w) * Z^T
        ZwZt = (Z_s * w[None, :]) @ Z_s.T
        Hs = 0.5 * (GE_s * ZwZt + (GE_s * ZwZt).T)  # symmetrize
        bs = ((Z_s * w[None, :]) * Y_s).sum(axis=1)
        return Hs, bs

    # initial (unweighted) LS step to seed IRLS
    Hs, bs = build_H_b(np.ones(Z_s.shape[1], dtype=Z_s.dtype))
    Hs_r = Hs + debias_ridge * np.eye(Hs.shape[0])
    try:
        gam = np.linalg.solve(Hs_r, bs)
    except np.linalg.LinAlgError:
        gam = np.linalg.lstsq(Hs_r, bs, rcond=1e-12)[0]

    # optional sign / nonneg enforcement after each update
    def project(g):
        if signs is not None:
            s = signs[supp]
            g = np.sign(s) * np.maximum(0.0, np.sign(s) * g)
        return g

    gam = project(gam)

    for _ in range(int(irls_max_iter)):
        # compute column residual norms via Rhat
        Rhat_cols = E_s @ (gam[:, None] * Z_s)      # n x p
        residuals = R - Rhat_cols                   # n x p
        e = np.linalg.norm(residuals, axis=0)       # length p

        # Huber weights per column (robust to outliers)
        w_huber = np.where(e <= huber_delta, 1.0, huber_delta / (e + irls_eps)).astype(Z_s.dtype)

        # ---------- NEW: combine with endogeneity weights ----------
        w_total = (w_huber * o_weight).astype(Z_s.dtype)

        # rebuild weighted normal equations and solve
        Hs_w, bs_w = build_H_b(w_total)
        Hs_wr = Hs_w + debias_ridge * np.eye(Hs_w.shape[0])
        try:
            gam_new = np.linalg.solve(Hs_wr, bs_w)
        except np.linalg.LinAlgError:
            gam_new = np.linalg.lstsq(Hs_wr, bs_w, rcond=1e-12)[0]

        gam_new = project(gam_new)

        # small damping to avoid zig-zag on tough cases
        gam = 0.7 * gam + 0.3 * gam_new

    # map back to full gamma
    gamma = np.zeros(k, dtype=E.dtype)
    gamma[supp] = gam

    # final Delta, Rhat with robust-debiased gam (use true Z_s here, not weighted)
    Delta = (E_s * gam[None, :]) @ E_s.T
    Rhat  = E_s @ (gam[:, None] * Z_s)
    return gamma, Delta, Rhat

def solve_lad_lasso(
    E, V, R, signs,
    lam,
    nonneg=False,
    verbose=False
):
    """
    Solves the sparse gamma problem using a robust LAD (L1-norm) loss function
    and an L1-norm penalty on gamma (Lasso), formulated as a single LP.
    """
    n, k = E.shape
    _, p = V.shape

    m = gp.Model("lad_lasso")
    if not verbose:
        m.Params.OutputFlag = 0

    # 1. Define gamma variables
    lb = -gp.GRB.INFINITY if not nonneg else 0.0
    gamma = m.addMVar(k, lb=lb, name="gamma")
    if signs is not None:
        for i in range(k):
            if signs[i] == 1:
                gamma[i].lb = 0.0
            elif signs[i] == -1:
                gamma[i].ub = 0.0
    
    # 2. Define the L1 penalty on gamma (|gamma|_1)
    t_gamma = m.addMVar(k, lb=0.0, name="t_gamma")
    m.addConstr(gamma <= t_gamma)
    m.addConstr(-gamma <= t_gamma)
    l1_penalty_gamma = t_gamma.sum()

    # 3. Define the L1 loss on the residuals using a direct matrix expression
    # --- THIS IS THE CORRECTED PART ---
    # Create the R_hat variable matrix *correctly* by adding it to the model.
    R_hat_expr = m.addMVar((n, p), name="R_hat", lb=-gp.GRB.INFINITY)
    # ---
    
    # Pre-calculate the tensor M where R_hat = sum(gamma_i * M_i)
    M = np.zeros((k, n, p))
    for i in range(k):
        E_i = E[:, i:i+1]
        M[i, :, :] = E_i @ (E_i.T @ V)
    
    # Define R_hat using a constraint
    m.addConstr(R_hat_expr == gp.quicksum(gamma[i] * M[i, :, :] for i in range(k)))
    
    residual_expr = R - R_hat_expr

    T_res = m.addMVar(R.shape, lb=0.0, name="T_res")
    m.addConstr(residual_expr <= T_res)
    m.addConstr(-residual_expr <= T_res)
    l1_loss_residual = T_res.sum()

    # 4. Set the final objective
    m.setObjective(l1_loss_residual + lam * l1_penalty_gamma, GRB.MINIMIZE)
    
    # 5. Optimize
    m.optimize()
    
    if m.Status != GRB.OPTIMAL:
        print(f"Warning: Gurobi finished with status {m.Status}. Solution may not be optimal.")
        if m.SolCount == 0:
            return np.zeros(k), np.zeros((n,n)), np.zeros_like(R)

    gamma_sol = gamma.X
    
    # Calculate final R_hat for returning
    Zs = E.T @ V
    R_hat_sol = E @ (gamma_sol[:, None] * Zs)
    Dsub_sol = (E * gamma_sol[None, :]) @ E.T
    
    return gamma_sol, Dsub_sol, R_hat_sol

def L1_sub(
    E, V, R,
    lam=1e-2,
    err_tol=1.5e-2,
    lam_up_multipliers=(1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0),
    nonneg=False,
    verbose=False,
    signs=None,
    topo_dict=None,
    changed_lines=None,
    eps_gamma=0.5,
):
    def rel_err(Rhat):
        num = np.linalg.norm(R - Rhat, ord='fro')
        den = max(1e-18, np.linalg.norm(R, ord='fro'))
        return num / den

    cand = []
    base = lam if lam is not None else 1.0
    for mult in lam_up_multipliers:
        lam_try = base * float(mult)
        
        # --- The ONLY CHANGE is this single, simple call ---
        gS, Dsub, Rhat = solve_lad_lasso(
            E, V, R, signs,
            lam=lam_try,
            nonneg=nonneg,
            verbose=verbose,
        )
        # ---

        nnz = int(np.count_nonzero(np.abs(gS) > 1e-6))
        err = rel_err(Rhat)
        cand.append((nnz, err, lam_try, gS, Dsub, Rhat))
        if verbose:
            print(f"[auto] λ={lam_try:.3e}  nnz={nnz:3d}  rel_err={err:.4e}")

    # The rest of your function (model selection and debiasing) stays the same...
    feasible = [c for c in cand if c[1] <= err_tol]
    if len(feasible) > 0:
        feasible.sort(key=lambda x: (x[0], x[1]))
        _, _, _, gS, _, _ = feasible[0]
    else:
        cand.sort(key=lambda x: (x[1], x[0]))
        _, _, _, gS, _, _ = cand[0]

    supp = np.where(np.abs(gS) > eps_gamma)[0]
    return supp


class topology_detection_lasso:
    """
    Detect line-status changes in real time and recover the updated topology.

    Adaptive bits:
      - Jump detection in monitor(): robust z-score (median/MAD), freezes baselines while pending.
      - Row activity in update(): slow, clipped EWMA baseline per row + delta gate + top-K safeguard.
      - min_hits: auto≈0.6*window if not provided.

    NOTE: mask_gamma is now strictly hand-designed: keep |gamma| > mask_gamma.
    """

    def __init__(self, n_rows, lines,
                 window=5, min_hits=None,
                 eps=None,                 # if None, adaptive; else used as floor
                 detect_window=5,
                 jump_ratio=None,          # if None, use robust z-score (≈3.5σ) instead
                 tol=None,                 # if None, adaptive from dispersion
                 mask_gamma=None,
                 k_level=1.0,
                 active_th=0.01):         
        self.n_rows   = int(n_rows)
        self.window   = int(window)
        self.min_hits = int(min_hits) if min_hits is not None else None
        self.eps_user = eps
        self.detect_window = int(detect_window)
        self.jump_ratio_user = jump_ratio
        self.tol_user = tol
        self.mask_gamma_user = mask_gamma  # <-- used directly below
        self.active_th = float(active_th)

        # history buffers
        self.hist       = np.zeros((self.window, self.n_rows), dtype=bool)
        self.v_obs_list = np.zeros((self.window, self.n_rows))
        self.R_hist     = np.zeros((self.window, self.n_rows))
        self.absR_hist  = np.zeros((self.window, self.n_rows))
        self.counts     = np.zeros(self.n_rows, dtype=int)
        self.ptr        = 0
        self.hist_count = 0

        # for adaptive per-row gating (EWMA baseline + spread)
        self.alpha_mu   = 0.10     # slow baseline
        self.alpha_sig  = 0.20     # spread
        self.growth_cap = 0.15     # <= +15%/step baseline growth
        self.k_level    = k_level     # level gate strength (~3σ)
        self.K_min      = max(2*2 + 2, 6)  # ensure enough rows even if quiet; n_lines defaults to 2 below
        self.mu         = np.zeros(self.n_rows)
        self.sig        = np.full(self.n_rows, 1e-4)
        self.prev_rowscore = np.zeros(self.n_rows)

        # robust jump detector storage
        self.hist_detect = deque(maxlen=max(self.detect_window, 7))
        self.pending = None   # (t_jump, err_at_jump, median_before)
        self._z_thresh = 3.5 if self.jump_ratio_user is None else None
        self.t = -1

        # topo bookkeeping
        self.previous_lines = lines.copy()
        self.n_lines = 2
        self.X_est = np.zeros((self.n_rows, self.n_rows))
        self.previous_topo_available = False
        self.topo_change_detected = False
        self.current_topo_available = False

        # reduced-space helpers
        self.T_reduce = None
        self.T_recover = None
        self.mask = np.zeros(self.n_rows)

    # ------------------- utilities -------------------

    @staticmethod
    def _median_mad(x, axis=None, eps=1e-18):
        """
        Robust median + Gaussian-consistent MAD.
        Safe for axis=None (uses keepdims for broadcasting).
        Returns (median_value, madn_value) with axis reduced.
        """
        x = np.asarray(x)
        med_keep = np.median(x, axis=axis, keepdims=True)
        mad = np.median(np.abs(x - med_keep), axis=axis)
        med_val = np.median(x, axis=axis)
        madn = 1.4826 * mad + eps
        return med_val, madn

    def _auto_min_hits(self):
        # ~60% of window; clipped to [2, window]
        return int(np.clip(np.round(0.9 * self.window), 2, self.window))

    def _auto_err_gates(self, err):
        """
        Decide whether to arm a jump at this err, and return a dynamic tol.
        - If jump_ratio_user is given: emulate your original ratio test.
        - Else: robust z-score with median/MAD over hist_detect.
        """
        if len(self.hist_detect) < self.hist_detect.maxlen:
            # warm-up: don't arm; use fallback tol
            tol_dyn = float(self.tol_user) if self.tol_user is not None else 1.0
            return False, None, tol_dyn

        arr = np.asarray(self.hist_detect, dtype=float)
        med, madn = self._median_mad(arr)
        madn = float(max(madn, 1e-12))

        if self.jump_ratio_user is not None:
            arm = err > float(self.jump_ratio_user) * med
        else:
            z = (err - med) / madn
            arm = z > self._z_thresh

        if self.tol_user is not None:
            tol_dyn = float(self.tol_user)
        else:
            # scale tol by dispersion; keep within [0.2, 10]
            tol_dyn = float(np.clip(3.0 * (madn / max(1e-10, med)), 0.2, 10.0))
        return arm, None, tol_dyn

    def _auto_eps_per_row(self, row_score):
        """
        Adaptive per-row level threshold from slow EWMA baseline (frozen while pending).
        Returns vector thr_j = max(floor, mu_j + k_level * sig_j).
        """
        floor = 1e-5 if (self.eps_user is None) else float(self.eps_user)

        # only adapt when there's no undecided jump
        if self.pending is None:
            mu_prop = (1.0 - self.alpha_mu) * self.mu + self.alpha_mu * row_score
            # cap how fast baseline can rise (don't swallow spikes)
            self.mu = np.minimum(mu_prop, (1.0 + self.growth_cap) * np.maximum(self.mu, 1e-12))

            dev = np.abs(row_score - self.mu)
            self.sig = (1.0 - self.alpha_sig) * self.sig + self.alpha_sig * dev
            self.sig = np.maximum(self.sig, 1e-6)

        thr = np.maximum(floor, self.mu + self.k_level * self.sig)
        return thr

    # ------------------- public API -------------------

    def update_X_est(self, X_est):
        self.X_est = X_est
        self.previous_topo_available = True
        self.topo_change_detected = False
        self.current_topo_available = True

    def reset_hist(self):
        self.hist       = np.zeros((self.window, self.n_rows), dtype=bool)
        self.v_obs_list = np.zeros((self.window, self.n_rows))
        self.R_hist     = np.zeros((self.window, self.n_rows))
        self.absR_hist  = np.zeros((self.window, self.n_rows))
        self.counts     = np.zeros(self.n_rows, dtype=int)
        self.ptr        = 0
        self.hist_count = 0

        # reset adaptive baselines
        self.mu[:] = 0.0
        self.sig[:] = 1e-4
        self.prev_rowscore[:] = 0.0

        # jump detector state
        self.hist_detect.clear()
        self.pending = None
        self.t = -1

        # self.current_topo_available = False

    def monitor(self, err):
        """
        Streaming jump detector with robust stats.
        Returns:
            None, or (t_jump, tag) with tag in {"wait","ignore, numerical error",
                                               "real power change","topo change"}.
        """
        self.t += 1
        event = None

        # resolve a pending jump after a short delay (≥2 steps)
        if self.pending is not None:
            t_jump, err_jump, prev_med = self.pending
            if self.t - t_jump < 2:
                return (t_jump, "wait")

            _, _, tol_dyn = self._auto_err_gates(err)
            delta = err - prev_med

            if err < 1e-5:
                event = (t_jump, "ignore, numerical error")
            else:
                if np.abs(delta) <= tol_dyn * max(prev_med, 1e-10) or err < 5e-4:
                    event = (t_jump, "real power change")
                else:
                    event = (t_jump, "topo change")
            self.pending = None

        # arm a new pending jump using robust gating
        if len(self.hist_detect) == self.hist_detect.maxlen:
            prev_med = float(np.median(self.hist_detect))
            arm, _, _ = self._auto_err_gates(err)
            if arm:
                self.pending = (self.t, err, prev_med)

        self.hist_detect.append(err)
        return event
    
    def update(self, R, v):
        """
        Push the newest R (shape [n_rows, *]).  
        Returns:
        changed_now : rows that look changed *right now* (bool mask)
        stable_ids  : rows judged *persistently* changed (np.int64 list)
        """
        # --- scale-free row-energy test (robust normalization) ---
        # use L1 row energy to stay close to your original sum(|R|) logic
        row_energy = np.abs(R).sum(axis=1)                          # shape [n_rows]
        scale = max(1e-4, np.mean(row_energy))                # robust per-snapshot scale
        score = row_energy / scale                                  # normalized, scale-free
        active = score > self.active_th       # bool mask length n_rows
        # print(active, score)

        # roll the ring buffer
        old = self.hist[self.ptr]          # row that is falling out
        self.counts -= old                 # remove its contribution
        self.hist[self.ptr] = active       # write new snapshot
        self.counts += active              # add contribution

        # keep storing the *original* R and v (no normalization here)
        self.R_hist[self.ptr] = R.squeeze()
        self.v_obs_list[self.ptr] = v.squeeze()

        self.ptr = (self.ptr + 1) % self.window

        stable_ids = np.where(self.counts >= 0.8*self.window)[0]
        self.hist_count += 1

        return active, stable_ids

    def est_line_change(self, ids):
        """
        Build candidates and judge whether the subspace indicates ≥2 dof.
        Returns:
            line_change_flag, detect_failed, changed_lines(None), cand_edges(list)
        """
        assert len(ids) > self.n_lines, "not enough involved buses for line change detection"

        rows_drop = np.setdiff1d(np.arange(self.n_rows), ids, assume_unique=True)
        R_hist_use = self.R_hist.copy()
        R_hist_use[:, rows_drop] = 0.0

        cand_edges = list(itertools.combinations(ids, 2))

        T_reduce, T_recover = self.build_transfer_matrix(ids)
        self.R_reduce = T_reduce @ (R_hist_use.T)

        U_svd, s, Vt = np.linalg.svd(self.R_reduce, full_matrices=False)
        n_potential = int(np.sum(s > 1e-5))
        if n_potential < 2 or s[1] / max(s[0], 1e-12) < 1e-5:
            return False, False, None, cand_edges
        else:
            if len(cand_edges) < self.n_lines:
                return False, False, None, cand_edges
            return True, False, None, cand_edges

    def build_transfer_matrix(self, stable_ids):
        n_involved = len(stable_ids)
        self.T_reduce = np.zeros((n_involved, self.n_rows))
        self.mask = np.zeros(self.n_rows)
        self.T_recover = np.zeros((self.n_rows, n_involved))
        for i, id in enumerate(stable_ids):
            self.T_reduce[i, id] = 1
            self.T_recover[id, i] = 1
            self.mask[id] = 1
        return self.T_reduce, self.T_recover
    
    def recover_full_topo(self, changed_lines, solution, detect_failed=False):
        """
        Return (X_new, line_status_dict, line_succ).
        """
        changed_lines = [e for e in changed_lines if e != (0,1)]
        # convenience helpers
        def _physics_ok(indices, lines, gam):
            if len(indices) < 2:
                return False
            pos = sum(gam[t] > 0 for t in indices)
            neg = sum(gam[t] < 0 for t in indices)
            if pos != neg:
                return False
            # consistency against previous topology
            for t in indices:
                i, j = lines[t]
                if gam[t] < 0 and (i + 1, j + 1) not in self.previous_lines:
                    return False
                if gam[t] > 0 and (i + 1, j + 1) in self.previous_lines:
                    return False
            # tentative edge set after changes must be a tree
            edge_set = set(self.previous_lines.keys())
            for t in indices:
                i, j = lines[t]
                e = (i + 1, j + 1)
                if gam[t] < 0:
                    edge_set.discard(e)
                else:
                    edge_set.add(e)
            return _is_tree(self.n_rows + 1, edge_set)

        X_new = None
        n = self.n_rows

        # --- Design (masked rows) ---
        E_org    = np.column_stack([edge_vector(n, *e) for e in changed_lines])  # (n×k)
        R_hist_T = np.multiply(self.R_hist.T, self.mask[:, np.newaxis])          # (n×T)
        V_obs    = np.multiply(self.v_obs_list.T, self.mask[:, np.newaxis])      # (n×T)

        # signs: -1 removal if line exists; +1 addition otherwise
        signs_global = np.asarray(
            [-1 if (e[0] + 1, e[1] + 1) in self.previous_lines else 1 for e in changed_lines],
            dtype=int
        )

        # --- First regression ---
        gamma, Delta_Xinv, R_hat = estimate_gamma_l1_screened_auto(
            E_org, V_obs, R_hist_T,
            lam=None,
            err_tol=1.0e-2,
            reweight_iters=0,
            nonneg=False,
            verbose=False,
            signs=signs_global,
            topo_dict=self.previous_lines,
            changed_lines=changed_lines,
            eps_gamma=1.0
        )
        err_prefit = float(np.linalg.norm(self.R_hist.T - R_hat, ord='fro'))

        # Quick success check (loose mask)
        mask_keep_tmp     = np.abs(gamma) > 1e-4
        changed_lines_tmp = [changed_lines[t] for t in np.where(mask_keep_tmp)[0]]
        line_succ = 1 if solution is not None else 0
        if line_succ:
            for line in solution:
                if line not in changed_lines_tmp:
                    line_succ = 0
                    break

        # --- Hard threshold (prefit) ---
        thr  = 15.0 if (self.mask_gamma_user is None) else float(self.mask_gamma_user)
        keep = np.where(np.abs(gamma) > thr)[0]
        pre_lines  = [changed_lines[i] for i in keep]
        pre_gamma  = gamma[keep]

        # --- Physics check on prefit ---
        ok_prefit = _physics_ok(list(range(len(pre_lines))), pre_lines, pre_gamma)

        # If prefit passes physics, accept without debiasing
        if ok_prefit and len(pre_lines) > 0:
            final_lines  = pre_lines
            final_gamma  = pre_gamma
            final_err    = err_prefit
        else:
            # --- Debias only if physics failed (or empty) ---
            if keep.size == 0:
                line_status_dict = {'line': [], 'err': err_prefit, 'fault': True}
                self.topo_change_detected = False
                return None, line_status_dict, line_succ

            gamma_deb, Delta_deb, Rhat_deb = debias(keep, E_org, V_obs, R_hist_T, signs_global)
            deb_gamma = gamma_deb[keep]  # note: debias returns full-size; select by 'keep'
            deb_err   = float(np.linalg.norm(self.R_hist.T - Rhat_deb, ord='fro'))

            # Re-apply hard threshold after debias
            valid = np.where(np.abs(deb_gamma) > thr)[0]
            final_lines = [pre_lines[i] for i in valid]
            final_gamma = deb_gamma[valid]

            ok_deb = _physics_ok(list(range(len(final_lines))), final_lines, final_gamma)
            final_err = deb_err

            if (not ok_deb) or len(final_lines) == 0:
                line_status_dict = {
                    'line':   final_lines,
                    'err':    final_err,
                    'fault':  True,
                }
                self.topo_change_detected = False
                return None, line_status_dict, line_succ

        # --- If we reach here, we have a valid set (prefit or debiased) ---
        line_status_dict = {
            'line':   final_lines,
            'err':    final_err,
            'fault':  False,
        }

        # --- Apply updates and rebuild X_new ---
        if not line_status_dict['fault']:
            for idx, (i, j) in enumerate(final_lines):
                e = (i + 1, j + 1)
                if final_gamma[idx] < 0:            # removal
                    self.previous_lines.pop(e, None)
                else:                               # addition
                    self.previous_lines[e] = 1.0 / final_gamma[idx]

            all_nodes = np.arange(n) + 1
            X_new = np.zeros((n, n))
            X_new = compute_full_R(X_new, all_nodes, self.previous_lines)

        self.topo_change_detected = False
        return X_new, line_status_dict, line_succ
    