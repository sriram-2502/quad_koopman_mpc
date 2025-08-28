from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Iterable
import numpy as np
import h5py
import json
import time
import re
from scipy.spatial.transform import Rotation as R  # geometric decode
from basis import observables as geom_observables

# =========================================================
# 0) Small helpers
# =========================================================
def _is_numeric(arr: np.ndarray) -> bool:
    return np.issubdtype(arr.dtype, np.number)

def _wrap_to_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2*np.pi) - np.pi

def _as_str_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple, np.ndarray)):
        out = []
        for v in x:
            if isinstance(v, (bytes, bytearray)):
                out.append(v.decode("utf-8"))
            else:
                out.append(str(v))
        return out
    if isinstance(x, (bytes, bytearray)):
        return [x.decode("utf-8")]
    return [str(x)]

# =========================================================
# 1) Geometric basis wrappers + decoder (for your Φ)
# =========================================================
def lift_1d(x: np.ndarray, p_max: int) -> np.ndarray:
    """Per-sample lift → returns 1-D φ (for EDMD vstack & eval_nstep_window)."""
    return geom_observables(np.atleast_2d(np.asarray(x, float)), p_max=p_max).ravel()

def lift_row(X: np.ndarray, p_max: int) -> np.ndarray:
    """Batch/row lift → returns 2-D Φ (for rollout), accepts (1,nx) or (N,nx)."""
    return geom_observables(np.asarray(X, float), p_max=p_max)

def _vee_from_skew(omega_hat: np.ndarray) -> np.ndarray:
    """
    Inverse of skew (vee operator). For
      ω̂ = [[0, -wz,  wy],
            [wz,  0, -wx],
            [-wy, wx, 0]]
    we have ω = [ω̂[2,1], ω̂[0,2], ω̂[1,0]].
    """
    wx = omega_hat[..., 2, 1]
    wy = omega_hat[..., 0, 2]
    wz = omega_hat[..., 1, 0]
    return np.stack([wx, wy, wz], axis=-1)

def decode_state_from_geom_phi(Phi: np.ndarray, p_max: int, nx: int = 12) -> np.ndarray:
    """
    Decode x = [pos(3), eul_xyz(rad)(3), v(3), ω(3)] from Φ built by your basis:
      Φ = [ pos(3), lin_vel(3), vec(R)(9), vec(ω̂)(9), vec(R ω̂^p)_{p=1..p_max} ]
    Works on (N, Nφ) or (1, Nφ).
    """
    Phi = np.asarray(Phi, float)
    if Phi.ndim == 1:
        Phi = Phi.reshape(1, -1)
    N = Phi.shape[0]

    i = 0
    pos = Phi[:, i:i+3];           i += 3
    vlin = Phi[:, i:i+3];          i += 3
    Rvec = Phi[:, i:i+9];          i += 9
    ohvec = Phi[:, i:i+9];         i += 9
    # remainder (9*p_max) are ψ̄ terms we don't need for decoding

    Rmat = Rvec.reshape(N, 3, 3)
    # Optional: re-orthonormalize R via SVD for robustness
    # U, _, Vt = np.linalg.svd(Rmat)
    # Rmat = U @ Vt

    eul = R.from_matrix(Rmat).as_euler("xyz", degrees=False)  # (N,3)
    omega_hat = ohvec.reshape(N, 3, 3)
    omega = _vee_from_skew(omega_hat)                         # (N,3)

    x = np.zeros((N, nx), dtype=float)
    x[:, 0:3] = pos
    x[:, 3:6] = eul
    x[:, 6:9] = vlin
    x[:, 9:12] = omega
    return x

# =========================================================
# 2) EDMD with inputs (covariance form; generic Φ)
# =========================================================
def edmd_with_inputs(
    X: np.ndarray,
    Y: np.ndarray,
    U: np.ndarray,
    lift_fn,
    l2_reg: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Learn Koopman matrices A, B such that:
        Φ(Y) ≈ A Φ(X) + B U
    Shapes:
        X: (M, nx)   states at t
        Y: (M, nx)   states at t+1
        U: (M, nu)   inputs at t
    Notes:
      - Uses empirical covariances G1, G2 and ridge on G2 for numerical stability.
      - Returns A ∈ R^{N×N}, B ∈ R^{N×nu} where N = dim(Φ(x)).
      - With row vectors, prediction uses: φ_next = φ @ A.T + u @ B.T
    """
    # Lift sample-wise to ensure lift_fn can handle 1×nx inputs
    PhiX = np.vstack([lift_fn(xi) for xi in X])   # (M, N)
    PhiY = np.vstack([lift_fn(yi) for yi in Y])   # (M, N)

    # Augmented regressor: [ Φ(X)  U ]
    Phihat = np.hstack([PhiX, U])                 # (M, N + nu)

    # Empirical covariances (normalized by M)
    M = PhiX.shape[0]
    scale = 1.0 / max(1, M)
    G1 = scale * (PhiY.T @ Phihat)                # (N, N+nu)
    G2 = scale * (Phihat.T @ Phihat)              # (N+nu, N+nu)

    # Ridge for stability
    if l2_reg > 0.0:
        G2 = G2 + l2_reg * np.eye(G2.shape[0], dtype=G2.dtype)

    # K = G1 * G2^{-1}
    K = G1 @ np.linalg.solve(G2, np.eye(G2.shape[0], dtype=G2.dtype))  # (N, N+nu)

    Nphi = PhiX.shape[1]
    A = K[:, :Nphi]                 # (N, N)
    B = K[:, Nphi:]                 # (N, nu)
    return A, B

# =========================================================
# 3) Train/Val split (contiguous)
# =========================================================
def train_val_split(X0: np.ndarray, X1: np.ndarray, U0: np.ndarray, val_frac: float = 0.2) -> Dict[str, Dict[str, np.ndarray]]:
    N = X0.shape[0]
    N_val = max(1, int(round(val_frac * N)))
    N_tr  = max(1, N - N_val)
    return {
        "train": {"X0": X0[:N_tr], "X1": X1[:N_tr], "U0": U0[:N_tr]},
        "val":   {"X0": X0[N_tr:], "X1": X1[N_tr:], "U0": U0[N_tr:]},
    }

# =========================================================
# 4) Rollout in lifted space (geometric decode)
# =========================================================
def rollout_open_loop_geom(
    A: np.ndarray, B: np.ndarray,
    p_max: int,
    x0: np.ndarray, U_seq: np.ndarray,
) -> np.ndarray:
    """
    Open-loop rollout in Φ-space, decoding x_k from Φ(x_k) using geometry.
    - Starts from x0 (12,), applies Φ(x) → φ, evolves φ_{k+1} = φ A^T + u_k B^T.
    - Decodes each φ_k to x_k via (R,euler) and (ω̂→ω).
    Returns X_pred of shape (H, 12).
    """
    H = U_seq.shape[0]
    phi = lift_row(x0.reshape(1, -1), p_max=p_max)  # (1, Nφ)
    X_pred = np.zeros((H, 12), float)
    for k in range(H):
        u = U_seq[k].reshape(1, -1)
        phi = (phi @ A.T) + (u @ B.T if B.size else 0.0)  # (1, Nφ)
        X_pred[k] = decode_state_from_geom_phi(phi, p_max=p_max, nx=12)[0]
    return X_pred

# =========================================================
# 5) HDF5: discover raw datasets and build (X0,X1,U0)
# =========================================================
STATE_KEYS = ["X", "x", "state", "states", "obs", "observation", "observations", "state_seq", "X_all", "state_history"]
INPUT_KEYS = ["U", "u", "input", "inputs", "control", "controls", "action", "actions", "act", "U_all", "control_seq"]

def _iter_numeric_datasets(h5: h5py.File) -> Iterable[Tuple[str, np.ndarray]]:
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            try:
                arr = np.array(obj)
                if _is_numeric(arr):
                    ds.append((f"/{name}", arr))
            except Exception:
                pass
    ds: List[Tuple[str, np.ndarray]] = []
    h5.visititems(visitor)
    return ds

def _pick_dataset_by_name(datasets: List[Tuple[str, np.ndarray]], candidates: List[str]) -> Optional[Tuple[str, np.ndarray]]:
    # Score by whether any candidate token appears in the path (case-insensitive), preferring 2D arrays with width>=2
    best = None
    best_score = -1
    for path, arr in datasets:
        s = path.lower()
        score = 0
        for tok in candidates:
            if re.search(rf"(^|/|_)({re.escape(tok)})($|/|_)", s):
                score += 1
        if arr.ndim == 2 and arr.shape[1] >= 2:
            score += 1
        if score > best_score:
            best = (path, arr)
            best_score = score
    return best

def _labels_from_h5(h5: h5py.File) -> Optional[List[str]]:
    for k in ["/labels", "/state_labels", "/dataset/labels", "/data/labels"]:
        if k in h5:
            try:
                return _as_str_list(np.array(h5[k]).tolist())
            except Exception:
                pass
    return None

def _build_X0X1U0_from_raw(h5: h5py.File) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    # Try to pick state & input datasets by name
    datasets = list(_iter_numeric_datasets(h5))
    picked_state = _pick_dataset_by_name(datasets, STATE_KEYS)
    picked_input = _pick_dataset_by_name(datasets, INPUT_KEYS)

    if picked_state is None:
        raise RuntimeError("Could not locate a state time-series (tried common names like /X, /states, /obs, ...).")

    path_x, X = picked_state
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    U = None
    if picked_input is not None:
        path_u, U = picked_input
        U = np.asarray(U, float)
        if U.ndim == 1:
            U = U.reshape(-1, 1)

    # Align lengths to form transitions
    Tx = X.shape[0]
    if Tx < 2:
        raise RuntimeError(f"State sequence too short for transitions: {path_x} has length {Tx}.")

    if U is None:
        # No inputs -> treat as autonomous (nu=0)
        X0 = X[:-1].copy()
        X1 = X[1:].copy()
        U0 = np.zeros((Tx-1, 0), float)
    else:
        Tu = U.shape[0]
        if Tu == Tx:
            X0 = X[:-1].copy()
            X1 = X[1:].copy()
            U0 = U[:-1].copy()
        elif Tu == Tx - 1:
            X0 = X[:-1].copy()
            X1 = X[1:].copy()
            U0 = U.copy()
        else:
            N = min(Tx-1, Tu)
            X0 = X[:N].copy()
            X1 = X[1:N+1].copy()
            U0 = U[:N].copy()

    labels = _labels_from_h5(h5)
    if labels is None:
        labels = [f"x{i}" for i in range(X.shape[1])]
    return X0, X1, U0, labels

def load_X0_X1_U0_from_h5(h5_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    1) Try explicit datasets: /X0, /X1, /U0 (also under /data or /dataset)
    2) Otherwise, auto-build from raw sequences (e.g., /X and /U) with alignment rules.
    """
    with h5py.File(h5_path, "r") as h5:
        def _read_first(names: List[str]) -> Optional[np.ndarray]:
            for nm in names:
                if nm in h5:
                    try:
                        arr = np.array(h5[nm])
                        if _is_numeric(arr):
                            return arr
                    except Exception:
                        pass
            return None

        X0 = _read_first(["/X0", "/data/X0", "/dataset/X0"])
        X1 = _read_first(["/X1", "/data/X1", "/dataset/X1"])
        U0 = _read_first(["/U0", "/data/U0", "/dataset/U0"])

        if X0 is not None and X1 is not None:
            # Use explicit, even if U0 missing (autonomous fallback)
            X0 = np.asarray(X0, float).reshape((-1, X0.shape[-1]))
            X1 = np.asarray(X1, float).reshape((-1, X1.shape[-1]))
            if U0 is None:
                U0 = np.zeros((X0.shape[0], 0), float)
            else:
                U0 = np.asarray(U0, float).reshape((X0.shape[0], -1))

            labels = _labels_from_h5(h5) or [f"x{i}" for i in range(X0.shape[1])]
            return X0, X1, U0, labels

        # Build from raw
        return _build_X0X1U0_from_raw(h5)

# =========================================================
# 6) Main runner
# =========================================================
def run_edmd_and_save(
    h5_path: Path | str,
    out_group: str = "/eval",
    p_max: int = 5,
    l2_reg: float = 1e-6,
    val_frac: float = 0.2,
    H: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Loads (or builds) X0/X1/U0 from an HDF5 file, fits EDMD with inputs on TRAIN split
    using YOUR geometric basis, evaluates an H-step open-loop rollout on a contiguous
    window in VAL split (decoded via geometry), writes results under `out_group`,
    and returns a summary dict.
    """
    h5_path = Path(h5_path)
    X0, X1, U0, labels = load_X0_X1_U0_from_h5(h5_path)
    N, nx = X0.shape
    nu = U0.shape[1]

    # Split
    splits = train_val_split(X0, X1, U0, val_frac=val_frac)
    tr, va = splits["train"], splits["val"]
    N_train = tr["X0"].shape[0]

    # Fit EDMD with your geometric basis (per-sample 1-D φ)
    A, B = edmd_with_inputs(
        tr["X0"], tr["X1"], tr["U0"],
        lift_fn=lambda xi: lift_1d(xi, p_max),
        l2_reg=l2_reg,
    )

    # Validation rollout (geometric decode)
    Nv = va["X0"].shape[0]
    if Nv < 2:
        raise RuntimeError("Validation split too short to evaluate.")
    H_eff = min(H, Nv)
    x0   = va["X0"][0]
    Useq = va["U0"][:H_eff]
    Xtru = va["X1"][:H_eff]

    Xpred = rollout_open_loop_geom(A, B, p_max=p_max, x0=x0, U_seq=Useq)

    # Error metrics with angle wrapping if 12D SRB convention
    err = Xpred - Xtru
    if nx == 12:
        err[:, 3:6] = _wrap_to_pi(err[:, 3:6])  # residual wrap (radians)

    rmse_per_state = np.sqrt(np.mean(err**2, axis=0))
    overall_rmse   = float(np.sqrt(np.mean(rmse_per_state**2)))

    meta = {
        "N": int(N_train), "nx": int(nx), "nu": int(nu),
        "H": int(H_eff), "p_max": int(p_max), "l2_reg": float(l2_reg),
        "val_frac": float(val_frac), "seed": int(seed),
        "timestamp": time.time(),
        "notes": "Geometric basis Φ=[pos, v, vec(R), vec(ω̂), vec(R ω̂^p)]; EDMD covariance form; radians.",
    }

    # Write back to H5
    group = out_group if str(out_group).startswith("/") else f"/{out_group}"
    with h5py.File(h5_path, "a") as h5:
        if group in h5:
            del h5[group]
        g = h5.create_group(group)
        g.create_dataset("X_true", data=Xtru, compression="gzip")
        g.create_dataset("X_pred", data=Xpred, compression="gzip")
        dt = h5py.string_dtype(encoding="utf-8")
        g.create_dataset("labels", data=np.array(labels, dtype=dt))
        for k, v in meta.items():
            try:
                g.attrs[k] = v
            except Exception:
                g.attrs[k] = json.dumps(v)

        # Store Koopman matrices
        kg = h5.create_group("/koopman") if "/koopman" not in h5 else h5["/koopman"]
        for nm in ["A", "B"]:
            if nm in kg: del kg[nm]
        kg.create_dataset("A", data=A, compression="gzip")
        kg.create_dataset("B", data=B, compression="gzip")

    return {
        "X_true": Xtru, "X_pred": Xpred, "labels": labels,
        "rmse_per_state": rmse_per_state, "overall_rmse": overall_rmse,
        "meta": meta, "A": A, "B": B,
    }
