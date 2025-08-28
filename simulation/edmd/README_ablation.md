# EDMD Trajectory‑Count Ablation Study (Quadruped Koopman MPC)

This README explains how the **trajectory‑count ablation** works for the Koopman EDMD models we train from the quadruped datasets, and how to reproduce the results from the notebook cells we prepared.

---

## What this ablation measures

We evaluate how model quality changes as we vary the **number of training trajectories** (episodes) used to fit the Koopman model.  
We train models with:
```
N_traj ∈ {10, 100, 500, 1000}
```
drawn from **experiments 2–11** (each has ~100 trajectories), and we evaluate on **experiment1** (held‑out test set).

For each saved model, evaluation samples **10 random test trajectories** and, within each, a **contiguous H‑step window** (default `H=20`) to compute prediction error. We then report:
- **Per‑state RMSE** (mean ± std over the 10 windows), and
- **Overall RMSE** (mean ± std, where overall = RMS of per‑state RMSE).

We also report the **eigen‑spectrum** of the lifted matrix \(A\) (with a unit‑circle plot) and the **controllability rank** of \((A,B)\) in the lifted space.

---

## Data assumptions

- HDF5 layout contains a `/recordings` group with arrays like `base_pos`, `base_ori_euler_xyz`, `base_lin_vel`, `base_ang_vel` of shape `(episodes, timesteps, dims)`.
- A custom loader is used:
  - `H5Reader(file_path, lazy=True)`  
  - `QuadrupedEDMDDataset(dataset=..., downsample=1, pos_zero_start=True, normalize=None).build()`
- The loader provides **flattened transitions**:
  - `ds.X0, ds.X1 ∈ ℝ^{N×nx}` and `ds.U0 ∈ ℝ^{N×nu}` aligned for one‑step learning.
- **Angles are SI radians.** We wrap Euler residuals (indices 3..5) with \((\alpha+\pi)\bmod 2\pi - \pi\) when computing RMSEs.
- Verified **logger step**: `dt = 0.002 s (500 Hz)`.

We treat **trajectory = episode**. Transition windows for an episode of length `T` are length `T‑1` when flattened.

---

## EDMD model (with inputs)

We learn a **discrete‑time, lifted linear model** with inputs at the logging rate (`dt=0.002 s`):
\[\n\Phi(x_{k+1}) \approx A\,\Phi(x_k) + B\,u_k,\n\]
where \(\Phi\) is a **monomial dictionary** with **identity‑first** ordering:
- bias \(1\) (optional, default **included**),
- identity \(x\),
- element‑wise monomials \(x^2, x^3, \dots, x^{p_{\max}}\) (no cross terms).

**Dictionary (identity‑first):**
```python
Phi = observables(x, p_max=P_MAX, include_constant=True)
```

**EDMD (covariance form with ridge):**
```python
# Φ(Y) ≈ A Φ(X) + B U  using empirical covariances G1, G2
PhiX = vstack([lift_fn(xi) for xi in X])    # each lift returns 1-D φ
PhiY = vstack([lift_fn(yi) for yi in Y])
Phihat = hstack([PhiX, U])
G1 = (PhiY.T @ Phihat) / M
G2 = (Phihat.T @ Phihat) / M + λ I
K  = G1 @ solve(G2, I)
A  = K[:, :N]
B  = K[:, N:]
```

**Important:** Our EDMD fitting calls `lift_fn` **per‑sample** and `vstack`s the results. Therefore, for fitting and for `eval_nstep_window`, we use a lift that returns **1‑D** vectors:
```python
# 1-D φ for per-sample lift
lift_fn_train = lambda x: observables(np.atleast_2d(x), p_max=P_MAX, include_constant=True).ravel()
lift_fn_eval  = lift_fn_train  # eval_nstep_window expects 1-D φ too
```

For **manual rollouts** (if used elsewhere), we keep the original batch lift that returns **2‑D** for `(1, nx)` inputs.

---

## Two-phase workflow

We split the notebook into two phases controlled by a single `MODE` switch:

```
MODE ∈ {'build', 'eval', 'both'}
```

### Build phase (`MODE='build'` or `'both'`)

1. **Training pool**: build a list of episode transition slices from **experiments 2–11**.  
2. For each `N_traj`:
   - Randomly sample `N_traj` episodes (seeded RNG) and concatenate their transitions.
   - Fit \((A,B)\) with the **covariance EDMD** and `lift_fn_train`.
   - Compute:
     - Eigenvalues of \(A\), spectral radius \(ρ(A)=\max|\lambda_i|\).
     - Controllability matrix \( \mathcal{C}=[B,\,AB,\,\dots,\,A^{n-1}B] \) in lifted space and its **rank**.
   - **Save**: `koopman_N{N_traj}_seed{SEED}.npz` with keys `A`, `B`, and `meta` (JSON string), including `N_traj`, `seed`, `p_max`, `l2_reg`, `nx`, `nu`, `nphi`, `dt`, `labels`, `rhoA`, `ctrb_rank`, and the indices of the selected trajectories.
   - Plot eigenvalues on the **unit circle** grid and print a metrics table (max|eig| and controllability rank).

**Reproducibility (`seed`)**: controls which training trajectories are sampled. The EDMD solve is deterministic once data is fixed.

### Eval phase (`MODE='eval'` or `'both'`)

For each saved model:
1. Sample **10 random test trajectories** from experiment1 and, within each, a random start such that a window of length `H` **fits** inside the episode.
2. Call **`eval_nstep_window`** with:
   - the model \((A,B)\),
   - `lift_fn_eval` (1‑D φ),
   - the test dataset `ds_test` and a trivial `{\"val_mask\": np.ones(...)}` split,
   - the absolute `start_index` and `H`,
   - `degrees=False` (angles are radians).
3. Compute **RMSE per state** and **overall RMSE** for each window; then report **mean ± std** across the 10 windows:
   - A compact **overall table** (mean/std, along with `ρ(A)`, controllability rank, and `nphi` from metadata),
   - A **per‑state table** (mean/std across the 10 windows).

**Reproducibility (`seed`)**: controls which test trajectories and window starts get chosen. You can use separate RNGs for build vs eval if desired.

---

## Configuration knobs (top of the notebook)

- `MODE`: `'build'`, `'eval'`, or `'both'`
- `TRAJ_COUNTS`: e.g. `[10, 100, 500, 1000]`
- `P_MAX`: monomial degree for the dictionary
- `L2_REG`: ridge strength in EDMD
- `H`: window length for evaluation
- `SEED`: RNG seed for both build and eval (or split into `rng_build`/`rng_eval`)
- `DT`: log step (here `0.002 s`)
- `MODEL_DIR`: where `.npz` models are saved/loaded

---

## Outputs you should see

- **Build phase**
  - A printed line per model, e.g.  
    `"[saved] koopman_N10_seed42.npz | ρ(A)=1.0016 | rank(C)=61/61"`
  - A **table** for all built models listing: `N_traj, nphi, nu, max|eig(A)| (ρ), min|eig(A)|, ctrb_rank, ctrb_full?`.
  - A **grid of eigenvalue plots** (one subplot per model) with the unit circle.

- **Eval phase**
  - For each model, a small table of the **10 random windows** actually used (`test_traj_id`, `start_offset`, `H_eff`, `overall_rmse`).
  - An **Overall RMSE** table (mean ± std).
  - A **Per‑state RMSE** table (mean ± std).

---

## Angle wrapping & labels

- We treat Euler angles as **radians**; residuals are wrapped before RMSE:
  ```python
  err[:, 3:6] = (err[:, 3:6] + np.pi) % (2*np.pi) - np.pi
  ```
- State labels default to:  
  `["x","y","z","roll","pitch","yaw","vx","vy","vz","wx","wy","wz"]` for `nx=12`, otherwise `["x0", ...]`.  
  If the dataset provides `ds.labels` of the right length, those are used.

---

## Notes & tips

- **Trajectory‑aware evaluation**: By choosing an **episode** first and then the start offset inside that episode, we guarantee each `H`‑step window stays **within a single trajectory** (no cross‑episode leaks).
- **eval_nstep_window expectations**: It multiplies as `z ← A @ z + B @ u`. Ensure `lift_fn_eval(x)` returns **1‑D φ** (we use `.ravel()` in the wrapper).
- **Multi‑rate MPC**: If your controller runs at 100 Hz while the model is learned at 500 Hz, aggregate \((A,B)\) to 10 ms via
  \[ A_{10\text{ms}} = A^5, \quad B_{10\text{ms}} = \sum_{i=0}^{4} A^i B \]
  (or simulate five sub‑steps per control sample under ZOH). This is **orthogonal** to the ablation but commonly needed downstream.

- **Troubleshooting**
  - Shape errors like `size 1 is different from Nφ`: your lift likely returns `(1,Nφ)` where a **1‑D** `(Nφ,)` is expected. Use the provided `lift_fn_train/lift_fn_eval` wrappers.
  - Missing inputs: the loader computes a correct `U0`. Do **not** read `/recordings/action` directly (some files store empty placeholders).
  - Very short episodes: if an episode has `T<2`, it yields zero transitions and is ignored.

---

## File formats

Each saved model is an `.npz` with:
- `A` — Koopman lift matrix (Nφ×Nφ),
- `B` — input matrix (Nφ×nu),
- `meta` — JSON string with:  
  `{"N_traj","seed","p_max","l2_reg","nx","nu","nphi","dt","labels","rhoA","ctrb_rank","traj_indices","timestamp"}`.

You can load it with:
```python
d = np.load("koopman_N100_seed42.npz", allow_pickle=True)
A, B = d["A"], d["B"]
meta = json.loads(str(d["meta"].tolist()))
```

---

## Reproducibility

Given the same data, config, and `SEED`, you will get **the same training subset**, saved model filename, and **the same evaluation windows** (if you reuse `SEED`). To separate concerns, consider:
```python
rng_build = np.random.default_rng(SEED)
rng_eval  = np.random.default_rng(SEED + 1)
```
and use them in the respective cells.

---

Happy ablation! If you want the notebook to also export CSVs of the summary tables or add additional dictionaries (e.g., cross‑terms, trigs for yaw), it’s straightforward to extend from here.
