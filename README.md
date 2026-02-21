# Efficient Policy Adaptation for Voltage Control Under Unknown Topology Changes

Official code base for **Efficient Policy Adaptation for Voltage Control Under Unknown Topology Changes** (accepted at **PSCC 2026**).

This repository provides:
- Single‑phase 13‑bus and 56‑bus voltage control environments.
- Online adaptation with topology‑change detection and recovery.
- Utilities for rollouts, evaluation, and plotting.

---

## Requirements

This project is pure Python. Typical dependencies:
- Python 3.8+
- numpy, scipy
- torch
- pandapower
- numba
- gym
- networkx
- matplotlib
- tqdm

If you plan to use the LAD/LASSO solvers in topology detection, ensure `gurobipy` is available.
- gurobipy (requires a Gurobi license)
---

## Quick Start

### 1) Create environment and install dependencies (conda)
```bash
conda create -n topo-adapt python=3.8 -y
conda activate topo-adapt
conda install -y numpy scipy matplotlib tqdm networkx
pip install torch pandapower gym numba
```

Optional (requires a Gurobi license):
```bash
pip install gurobipy
```

### Pandapower API Compatibility (Important)

The environment simulation code in this repository currently uses an older `pandapower` API and intentionally keeps:

```python
pp_net = pp.converter.from_mpc(pp_model_pth, casename_mpc_file='case_mpc')
```

If you use a newer `pandapower` release, use:

```python
from pandapower.converter.matpower import from_mpc
pp_net = from_mpc(pp_model_pth, casename_mpc_file='case_mpc')
```

Also install `numba`, since newer `pandapower` setups commonly rely on it for performance and compatibility.

For this repo, do not change the code path if your local build is already configured for the older `pandapower` interface.

### 2) Run a real‑world load demo (notebook)
Open:
```
Real_world_load_with_topo_change.ipynb
```

### 3) Train baseline policies
```bash
python train_base_policy.py --help
python train_base_policy_56bus.py --help
```

### 4) Evaluate online performance
See `utils/run_rollouts.py` for evaluation utilities such as:
- `test_online_performance_NN_w_topo_change_lasso`

---

## Project Structure

```
varying_topo_online_opt/
  env/                # 13‑bus and 56‑bus environments, data models (.mat)
  topo_est/           # Topology change detection and topology recovery
  online_opt/         # Online policy adaptation (gradient updates)
  models/             # Policy network definitions
  utils/              # Rollout, plotting, and network utilities
  checkpoints/        # Pretrained models
  results/            # Saved trajectories/results
  figures/            # Plots output
  train_base_policy.py
  train_base_policy_56bus.py
  Real_world_load_with_topo_change.ipynb
```

---

## Notes

- Data files for the environments live in `env/models/`.
- Many scripts write outputs to `results/` and `figures/`.
- Random seeds are set in demos; you can adjust them for reproducibility.

---

## Limitations

- The topology change detector can **falsely trigger** change events under noisy or abrupt load variations.
- The LASSO‑based regression may require **additional denoising or smoothing** on real‑world signals to be reliable.

---

## Citation

If this repository is helpful in your research, please cite:

Paper: https://arxiv.org/abs/2602.10355

```bibtex
@article{feng2026efficient,
  title={Efficient Policy Adaptation for Voltage Control Under Unknown Topology Changes},
  author={Feng, Jie and Shi, Yuanyuan and Deka, Deepjyoti},
  journal={arXiv preprint arXiv:2602.10355},
  year={2026}
}
```
