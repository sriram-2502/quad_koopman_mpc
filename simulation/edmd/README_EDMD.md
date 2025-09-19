# EDMD Runner — Geometric Koopman with Inputs

This README documents the EDMD runner that learns a lifted linear model with inputs for SRB/centroidal data and validates it via open-loop rollout. All math is written with `$...$` (inline) or `$$...$$` (display). Each function includes a short pseudocode block.

---

## 1) Problem Setup (What we learn)

We learn Koopman matrices $(A,B)$ such that
$$
\Phi(x_{k+1}) \approx A\,\Phi(x_k) + B\,u_k,
$$
where
- $x_k \in \mathbb{R}^{n_x}$ is the physical state (SRB: $n_x=12$),
- $u_k \in \mathbb{R}^{n_u}$ is the input,
- $\Phi:\mathbb{R}^{n_x}\!\to\!\mathbb{R}^{N_\phi}$ is a **geometric** lifting.

**Row-vector convention** in code: $\ \phi_{k+1} = \phi_k A^\top + u_k B^\top.$

---

## 2) Geometric Lifting $\Phi$ and Decoding

Given $x=[p\in\mathbb{R}^3,\ \text{Euler}_{xyz}\in\mathbb{R}^3,\ v\in\mathbb{R}^3,\ \omega\in\mathbb{R}^3]$:
$$
\Phi(x)=\big[p,\ v,\ \mathrm{vec}(R),\ \mathrm{vec}(\hat\omega),\ \{\mathrm{vec}(R\hat\omega^p)\}_{p=1}^{p_{\max}}\big].
$$
- $R\in SO(3)$ from Euler angles (or directly from logs),
- $\hat\omega$ is the skew matrix of $\omega$,
- Higher-order terms $R\hat\omega^p$ enrich rotational-rate coupling.

**Decode** back to $x$ by slicing $\Phi$: $(p,v)$ directly, $R$ reshaped from 9-vector $\to$ Euler, and $\omega=\mathrm{vee}(\hat\omega)$.

---

## 3) EDMD with Inputs (Covariance / Normal Equations)

Given aligned transitions $(X_0,X_1,U_0)$:
1. Lift: $\Phi_X=\Phi(X_0)$, $\Phi_Y=\Phi(X_1)$.
2. Regressor: $\widehat\Phi=[\Phi_X\ \ U_0]$.
3. Empirical covariances (scale $1/M$): 
$$
G_1=\tfrac{1}{M}\Phi_Y^\top\widehat\Phi,\qquad
G_2=\tfrac{1}{M}\widehat\Phi^\top\widehat\Phi.
$$
4. Ridge: $G_2 \leftarrow G_2 + \lambda I$.
5. Solve: $K=G_1 G_2^{-1}$, then $A=K[:,:N_\phi],\ B=K[:,N_\phi:]$.

**Rollout** (open-loop): with $\phi_0=\Phi(x_0)$,
$$
\phi_{k+1}=\phi_k A^\top + u_k B^\top,\qquad x_{k+1}=\text{decode}(\phi_{k+1}).
$$

**Errors** (with angle wrapping on Euler): per-state RMSE and overall RMSE
$$
\mathrm{RMSE}_i=\sqrt{\tfrac{1}{H}\sum_k e_{k,i}^2},\quad
\mathrm{Overall}=\sqrt{\tfrac{1}{n_x}\sum_i \mathrm{RMSE}_i^2}.
$$

---

## 4) Function-by-Function Guide (with pseudocode)

> Symbols: `→` returns; `[ ]` indexing; `||` concat; `@` matrix-multiply

### `_is_numeric(arr)`
Return True if `arr.dtype` is numeric.

**Pseudocode**
```bash
function _is_numeric(arr):
    return dtype(arr) is numeric
```

---

### `_wrap_to_pi(a)`
Wrap angles to $(-\pi,\pi]$ elementwise: $(a+\pi) \bmod 2\pi - \pi$.

**Pseudocode**
```bash
function _wrap_to_pi(a):
    return (a + π) % (2π) - π
```

---

### `_as_str_list(x)`
Normalize possible bytes/arrays/strings to a flat `List[str]`.

**Pseudocode**
```bash
function _as_str_list(x):
    if x is None: return []
    if x is list/tuple/ndarray:
        out = []
        for v in x:
            out.append(bytes_to_str(v) if is_bytes(v) else str(v))
        return out
    if is_bytes(x): return [bytes_to_str(x)]
    return [str(x)]
```

---

### `lift_1d(x, p_max)`
Lift a **single** sample $x$ to a 1D $\phi=\Phi(x)$. Uses your `geom_observables`.

**Pseudocode**
```bash
function lift_1d(x, p_max):
    X = to_2d(x)         # shape (1,nx)
    Φ = geom_observables(X, p_max=p_max)  # (1,Nφ)
    return ravel(Φ)      # (Nφ,)
```

---

### `lift_row(X, p_max)`
Lift a **batch** $(N,n_x)$ to $\Phi(X)\in\mathbb{R}^{N\times N_\phi}$.

**Pseudocode**
```bash
function lift_row(X, p_max):
    X = asarray(X, float)
    return geom_observables(X, p_max=p_max)   # (N, Nφ)
```

---

### `_vee_from_skew(omega_hat)`
Inverse of skew (vee): $\mathrm{vee}(\hat\omega)=\omega$.

**Pseudocode**
```bash
function _vee_from_skew(Ωhat):  # (...,3,3)
    wx = Ωhat[..., 2, 1]
    wy = Ωhat[..., 0, 2]
    wz = Ωhat[..., 1, 0]
    return stack([wx, wy, wz], axis=-1)
```

---

### `decode_state_from_geom_phi(Phi, p_max, nx=12)`
Decode $x=[p,eul_{xyz},v,\omega]$ from geometric $\Phi$.

**Pseudocode**
```bash
function decode_state_from_geom_phi(Phi, p_max, nx=12):
    Φ = as2d(Phi)                  # (N, Nφ)
    i = 0
    p    = Φ[:, i:i+3];  i += 3
    v    = Φ[:, i:i+3];  i += 3
    Rvec = Φ[:, i:i+9];  i += 9
    ohv  = Φ[:, i:i+9];  i += 9
    Rmat     = reshape(Rvec, (N,3,3))
    eul_xyz  = euler_xyz_from_R(Rmat)     # scipy Rotation
    ωhat     = reshape(ohv, (N,3,3))
    ω        = vee(ωhat)
    X = zeros((N,nx))
    X[:,0:3]  = p
    X[:,3:6]  = eul_xyz
    X[:,6:9]  = v
    X[:,9:12] = ω
    return X
```

---

### `edmd_with_inputs(X, Y, U, lift_fn, l2_reg=1e-6)`
Compute $(A,B)$ by normal equations with ridge.

**Math**
$$
K=G_1(G_2+\lambda I)^{-1},\quad
A=K[:,0{:}N_\phi],\ B=K[:,N_\phi{:}].
$$

**Pseudocode**
```bash
function edmd_with_inputs(X, Y, U, lift_fn, l2_reg):
    ΦX = vstack(lift_fn(x) for x in X)    # (M,Nφ)
    ΦY = vstack(lift_fn(y) for y in Y)    # (M,Nφ)
    Φhat = hstack([ΦX, U])                # (M,Nφ+nu)
    M = rows(ΦX); s = 1/max(1,M)
    G1 = s * (ΦYᵀ @ Φhat)
    G2 = s * (Φhatᵀ @ Φhat)
    if l2_reg>0: G2 += l2_reg * I
    K = G1 @ solve(G2, I)                 # (Nφ, Nφ+nu)
    A = K[:, :Nφ]
    B = K[:, Nφ:]
    return A, B
```

---

### `train_val_split(X0, X1, U0, val_frac=0.2)`
Contiguous split preserving time ordering.

**Pseudocode**
```bash
function train_val_split(X0,X1,U0,val_frac):
    N = len(X0); Nval = max(1, round(val_frac*N)); Ntr = max(1, N-Nval)
    return {
      "train": {"X0":X0[:Ntr], "X1":X1[:Ntr], "U0":U0[:Ntr]},
      "val":   {"X0":X0[Ntr:], "X1":X1[Ntr:], "U0":U0[Ntr:]}
    }
```

---

### `rollout_open_loop_geom(A,B,p_max,x0,U_seq)`
Forecast in $\Phi$-space, decode each step.

**Pseudocode**
```bash
function rollout_open_loop_geom(A,B,p_max,x0,U_seq):
    H = len(U_seq)
    φ = lift_row(x0[None,:], p_max)    # (1,Nφ)
    Xpred = zeros((H,12))
    for k in 0..H-1:
        u = U_seq[k][None,:]
        φ = (φ @ Aᵀ) + (u @ Bᵀ if B.size>0 else 0)
        Xpred[k] = decode_state_from_geom_phi(φ, p_max, nx=12)[0]
    return Xpred
```

---

### HDF5 utilities

#### `_iter_numeric_datasets(h5)`
Iterate all numeric datasets: returns `[(path, array), ...]`.

**Pseudocode**
```bash
function _iter_numeric_datasets(h5):
    ds = []
    for each object (name,obj) in h5:
        if obj is Dataset:
            arr = np.array(obj)
            if is_numeric(arr): ds.append(("/"+name, arr))
    return ds
```

#### `_pick_dataset_by_name(datasets, candidates)`
Heuristics by name tokens and shape.

**Pseudocode**
```bash
function _pick_dataset_by_name(datasets, candidates):
    best = None; best_score = -1
    for (path, arr) in datasets:
        s = lower(path); score = 0
        for tok in candidates:
            if token_appears_like_segment(s, tok): score += 1
        if arr.ndim==2 and arr.shape[1]>=2: score += 1
        if score>best_score: best=(path,arr); best_score=score
    return best
```

#### `_labels_from_h5(h5)`
Try standard label nodes.

**Pseudocode**
```bash
function _labels_from_h5(h5):
    for k in ["/labels","/state_labels","/dataset/labels","/data/labels"]:
        if k in h5: return as_str_list(np.array(h5[k]).tolist())
    return None
```

#### `_build_X0X1U0_from_raw(h5)`
Auto-detect `X` and `U`, align to transitions.

**Pseudocode**
```bash
function _build_X0X1U0_from_raw(h5):
    datasets = _iter_numeric_datasets(h5)
    state = _pick_dataset_by_name(datasets, STATE_KEYS)
    input = _pick_dataset_by_name(datasets, INPUT_KEYS)
    if state is None: error("no state series")
    X = as2d(state.array)
    U = as2d(input.array) if input else None
    Tx = len(X); assert Tx>=2
    if U is None:
        X0=X[:-1]; X1=X[1:]; U0=zeros((Tx-1,0))
    else:
        Tu=len(U)
        if Tu==Tx:      X0=X[:-1]; X1=X[1:]; U0=U[:-1]
        elif Tu==Tx-1:  X0=X[:-1]; X1=X[1:]; U0=U
        else:           N=min(Tx-1,Tu); X0=X[:N]; X1=X[1:N+1]; U0=U[:N]
    labels = _labels_from_h5(h5) or default_labels(X.shape[1])
    return X0,X1,U0,labels
```

#### `load_X0_X1_U0_from_h5(h5_path)`
Prefer explicit `/X0,/X1,/U0`; else fallback to auto.

**Pseudocode**
```bash
function load_X0_X1_U0_from_h5(path):
    with h5py.File(path,"r") as h5:
        X0 = first_found(h5, ["/X0","/data/X0","/dataset/X0"])
        X1 = first_found(h5, ["/X1","/data/X1","/dataset/X1"])
        U0 = first_found(h5, ["/U0","/data/U0","/dataset/U0"])
        if X0 and X1:
            X0=as2d(X0); X1=as2d(X1); U0=as2d(U0) if U0 else zeros((len(X0),0))
            labels = _labels_from_h5(h5) or default_labels(X0.shape[1])
            return X0,X1,U0,labels
        else:
            return _build_X0X1U0_from_raw(h5)
```

---

### `run_edmd_and_save(h5_path, out_group="/eval", p_max=5, l2_reg=1e-6, val_frac=0.2, H=20, seed=42)`
Full pipeline: load → split → fit $(A,B)$ → rollout → metrics → save.

**Pseudocode**
```bash
function run_edmd_and_save(h5_path, out_group, p_max, l2_reg, val_frac, H, seed):
    X0,X1,U0,labels = load_X0_X1_U0_from_h5(h5_path)
    splits = train_val_split(X0,X1,U0,val_frac)
    tr, va = splits["train"], splits["val"]
    A,B = edmd_with_inputs(tr.X0, tr.X1, tr.U0, lift_fn=(x→lift_1d(x,p_max)), l2_reg=l2_reg)
    H_eff = min(H, len(va.X0)); x0 = va.X0[0]
    U_seq = va.U0[:H_eff]; X_true = va.X1[:H_eff]
    X_pred = rollout_open_loop_geom(A,B,p_max,x0,U_seq)
    err = X_pred - X_true
    if nx==12: err[:,3:6] = wrap_to_pi(err[:,3:6])
    rmse_state = sqrt(mean(err**2, axis=0))
    overall = sqrt(mean(rmse_state**2))
    save_to_h5(h5_path, out_group, X_true, X_pred, labels, A, B, meta)
    return {...}
```

---

## 5) Recommended Knobs

- `$p_{\max}\in[3,6]$`: richer coupling vs. conditioning.
- `$\lambda$` (ridge): $10^{-6}\!-\!10^{-4}$ typical.
- `H`: 20–100 for validation forecast.
- Always wrap Euler residuals; ensure radians.

---

## 6) Minimal Usage

```bash
from pathlib import Path
from edmd_runner import run_edmd_and_save

res = run_edmd_and_save(
    h5_path=Path("logs/go1_flat_01.h5"),
    out_group="/eval_geom_p5",
    p_max=5, l2_reg=1e-5, val_frac=0.2, H=50
)
print("Overall RMSE:", res["overall_rmse"])
A, B = res["A"], res["B"]
```

---

## 7) Notes for Koopman MPC

- Use cost **in base states** (no cost lift). Track $(p,\text{Euler},v,\omega)$ from decoded predictions.
- Maintain row-vector propagation convention to match training.
- If decoded $R$ drifts, re-orthonormalize by SVD $R\leftarrow UV^\top$ before Euler conversion.
