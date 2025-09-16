# edmd_runner2.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Iterable, Literal
import json
import time
import re

import numpy as np
import h5py
from scipy.spatial.transform import Rotation as R  # geometric decode

# Your geometric dictionary (vectorized)
from basis import observables as geom_observables


# =========================================================
# 0) Small helpers
# =========================================================
def _is_numeric(arr: np.ndarray) -> bool:
    return np.issubdtype(np.asarray(arr).dtype, np.number)

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

def _to_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 1:
        a = a[:, None]
    return a


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
    omega_hat = np.asarray(omega_hat)
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
# 2) Feature augmentation: contact & feet geometry (optional)
# =========================================================
def expand_contact_to_12(c4: np.ndarray) -> np.ndarray:
    """
    Expand 4 binary contacts to 12 mask over [fx,fy,fz]*4.
    c4: (N,4) → C12: (N,12) with each leg repeated over 3 axes.
    """
    c4 = _to_2d(c4).astype(float)
    C12 = np.repeat(c4, repeats=3, axis=1)
    return C12

def augment_phi_with_contact(Phi: np.ndarray, c4: np.ndarray) -> np.ndarray:
    """
    Concatenate contact (4) and its 12-expansion to Φ.
    """
    c4 = _to_2d(c4).astype(float)
    C12 = expand_contact_to_12(c4)
    return np.hstack([Phi, c4, C12])

def augment_phi_with_feet(Phi: np.ndarray, feet_pos_world: np.ndarray, com_world: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Concatenate foot positions (flattened) or foot positions relative to COM.
    feet_pos_world: (N, 4, 3), com_world: (N, 3) or None
    """
    feet = np.asarray(feet_pos_world, float)
    assert feet.ndim == 3 and feet.shape[1:] == (4, 3), f"feet_pos must be (N,4,3), got {feet.shape}"
    if com_world is not None:
        com = _to_2d(com_world)
        rel = feet - com[:, None, :]
        add = rel.reshape(rel.shape[0], -1)  # (N,12)
    else:
        add = feet.reshape(feet.shape[0], -1)  # (N,12)
    return np.hstack([Phi, add])


# =========================================================
# 3) Wrench helpers (Option 1 from discussion)
# =========================================================
def wrench_from_forces(feet_pos_world: np.ndarray, com_world: np.ndarray, u_foot: np.ndarray) -> np.ndarray:
    """
    feet_pos_world: (N,4,3)
    com_world:      (N,3)
    u_foot:         (N,12) stacked [fx,fy,fz]*4 in order [FL,FR,RL,RR]
    returns W:      (N,6) wrench [Fx,Fy,Fz, Mx,My,Mz] about COM (world)
    """
    feet = np.asarray(feet_pos_world, float)
    com  = _to_2d(np.asarray(com_world, float))
    U    = _to_2d(np.asarray(u_foot, float))
    assert feet.shape[0] == com.shape[0] == U.shape[0], "Length mismatch in wrench_from_forces inputs"

    N = feet.shape[0]
    W = np.zeros((N, 6), float)
    for k in range(N):
        F = np.zeros(3); M = np.zeros(3)
        c = com[k]
        for j in range(4):
            f = U[k, 3*j:3*j+3]
            r = feet[k, j] - c
            F += f
            M += np.cross(r, f)
        W[k, :3] = F
        W[k, 3:] = M
    return W


# =========================================================
# 4) EDMD core (covariance form; generic Φ)
# =========================================================
def edmd_with_inputs(
    X: np.ndarray,
    Y: np.ndarray,
    U: np.ndarray,
    lift_fn,
    l2_reg: float = 1e-6,
    contact: Optional[np.ndarray] = None,
    feet_pos: Optional[np.ndarray] = None,
    com_pos:  Optional[np.ndarray] = None,
    include_contact_in_lift: bool = False,
    include_feet_in_lift: bool = False,
    mask_inputs_by_contact: bool = True,
    input_mode: Literal["forces", "wrench", "forces+contact"] = "forces",
    fit_C_linear: bool = True,
    l2_reg_C: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """
    Learn Koopman matrices A, B (and optionally linear decoder C) such that:
        Φ(Y) ≈ A Φ(X) + B U_eff
        X    ≈ C Φ(X)       (if fit_C_linear)
    Shapes:
        X, Y: (M, nx) states at t, t+1
        U:    (M, nu) inputs at t  (forces or wrench depending on input_mode)
    Options:
        - input_mode="forces": U is (M,12)
        - input_mode="wrench": U is (M,6)  (if you pass feet_pos & com_pos we can compute from forces)
        - input_mode="forces+contact": same as "forces", plus we include contact in Φ and (optionally) mask inputs.
    Returns: (A, B, C or None, meta)
    """
    X = np.asarray(X, float); Y = np.asarray(Y, float); U = _to_2d(U)
    M, nx = X.shape
    nu = U.shape[1]

    # Build Φ(X), Φ(Y)
    PhiX = np.vstack([lift_fn(xi) for xi in X])   # (M, Nφ)
    PhiY = np.vstack([lift_fn(yi) for yi in Y])   # (M, Nφ)
    Nphi = PhiX.shape[1]

    # Optionally augment Φ with contact and/or feet geometry
    if include_contact_in_lift and (contact is not None):
        PhiX = augment_phi_with_contact(PhiX, contact)
        PhiY = augment_phi_with_contact(PhiY, contact)  # use c_k; could also use c_{k+1}
    if include_feet_in_lift and (feet_pos is not None):
        PhiX = augment_phi_with_feet(PhiX, feet_pos, com_world=com_pos)
        PhiY = augment_phi_with_feet(PhiY, feet_pos, com_world=com_pos)

    # Effective input according to mode
    if input_mode == "wrench":
        # If caller supplied U as forces (12), compute wrench if geometry present
        if nu == 12 and (feet_pos is not None) and (com_pos is not None):
            Ueff = wrench_from_forces(feet_pos, com_pos, U)
        else:
            Ueff = U  # already wrench
        nu_eff = Ueff.shape[1]

    elif input_mode in ("forces", "forces+contact"):
        Ueff = U.copy()
        # Optionally mask swing legs → zeros where contact=0
        if mask_inputs_by_contact and (contact is not None):
            C12 = expand_contact_to_12(contact)  # (M,12)
            if Ueff.shape[1] == 12:
                Ueff = Ueff * C12
        nu_eff = Ueff.shape[1]

    else:
        raise ValueError(f"Unknown input_mode: {input_mode}")

    # Empirical covariances
    scale = 1.0 / max(1, M)
    Phihat = np.hstack([PhiX, Ueff])                 # (M, Nφ+nu_eff)
    G1 = scale * (PhiY.T @ Phihat)                   # (Nφ_aug, Nφ+nu_eff)
    G2 = scale * (Phihat.T @ Phihat)                 # (Nφ+nu_eff, Nφ+nu_eff)
    if l2_reg > 0.0:
        G2 = G2 + l2_reg * np.eye(G2.shape[0], dtype=G2.dtype)

    K = G1 @ np.linalg.solve(G2, np.eye(G2.shape[0], dtype=G2.dtype))
    Nphi_aug = G1.shape[0]
    A = K[:, :Nphi_aug]
    B = K[:, Nphi_aug:]

    # Optional linear decoder C: X ≈ C Φ(X_aug)
    C = None
    if fit_C_linear:
        # Solve least-squares with ridge: min ||X - C ΦX||_F^2
        PhiXt = PhiX  # already augmented
        XtPhi = X.T @ PhiXt
        PP = PhiXt.T @ PhiXt
        if l2_reg_C > 0.0:
            PP = PP + l2_reg_C * np.eye(PP.shape[0])
        C = XtPhi @ np.linalg.solve(PP, np.eye(PP.shape[0]))  # (nx, Nφ_aug)

    meta = dict(
        M=int(M), nx=int(nx), nu=int(nu),
        Nphi=int(Nphi), Nphi_aug=int(A.shape[0]),
        input_mode=input_mode,
        include_contact_in_lift=bool(include_contact_in_lift),
        include_feet_in_lift=bool(include_feet_in_lift),
        mask_inputs_by_contact=bool(mask_inputs_by_contact),
        fit_C_linear=bool(fit_C_linear),
        l2_reg=float(l2_reg),
        l2_reg_C=float(l2_reg_C),
    )
    return A, B, C, meta


# =========================================================
# 5) Rollout in lifted space (choose: geometric decode or linear C)
# =========================================================
def rollout_open_loop(
    A: np.ndarray, B: np.ndarray,
    x0: np.ndarray, U_seq: np.ndarray,
    lift_fn,
    C: Optional[np.ndarray] = None,
    decode_geom: bool = False,
    p_max_for_geom: int = 5,
) -> np.ndarray:
    """
    φ_{k+1} = φ_k A^T + u_k B^T. State readout:
      - If decode_geom=True: use geometric decoder (assumes your geometric Φ).
      - Else: x_k = C φ_k  (requires C of shape (nx, Nφ_aug)).
    """
    H = U_seq.shape[0]
    phi = np.atleast_2d(lift_fn(x0)).astype(float)  # (1, Nφ_aug)
    X_pred = []
    for k in range(H):
        u = U_seq[k:k+1, :]
        phi = (phi @ A.T) + (u @ B.T if B.size else 0.0)
        if decode_geom:
            xk = decode_state_from_geom_phi(phi, p_max=p_max_for_geom, nx=12)[0]
        else:
            if C is None:
                raise ValueError("C is required when decode_geom=False.")
            xk = (C @ phi.T).ravel()
        X_pred.append(xk)
    return np.asarray(X_pred)


# =========================================================
# 6) HDF5: discover raw datasets and build (X0,X1,U0) (+ optional contact/feet/com)
# =========================================================
STATE_KEYS = ["X", "x", "state", "states", "obs", "observation", "observations", "state_seq", "X_all", "state_history"]
INPUT_KEYS = ["U", "u", "input", "inputs", "control", "controls", "action", "actions", "act", "U_all", "control_seq"]
CONTACT_KEYS = ["contact", "contacts", "contact_state", "contact_seq", "feet_contact", "foot_contact"]
FEET_POS_KEYS = ["feet_pos", "foot_pos", "feet_positions", "foot_positions"]
COM_KEYS = ["com", "com_pos", "CoM", "base_pos", "base_position"]

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
        if arr.ndim >= 2 and arr.shape[-1] >= 1:
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

def load_from_h5(
    h5_path: Path,
    try_autodiscover: bool = True,
    explicit: Optional[Dict[str, str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Returns a dict with (some may be absent):
      X0, X1, U0, labels, contact, feet_pos, com
    """
    out: Dict[str, np.ndarray] = {}
    with h5py.File(h5_path, "r") as h5:
        if explicit:
            def _read(path: str) -> Optional[np.ndarray]:
                return np.asarray(h5[path]) if (path in h5) else None

            X0 = _read(explicit.get("X0", ""))
            X1 = _read(explicit.get("X1", ""))
            U0 = _read(explicit.get("U0", ""))
            if X0 is None or X1 is None:
                raise RuntimeError("Explicit paths provided but X0 or X1 not found.")
            out["X0"] = np.asarray(X0, float)
            out["X1"] = np.asarray(X1, float)
            out["U0"] = np.asarray(U0, float) if U0 is not None else np.zeros((out["X0"].shape[0], 0), float)
            out["labels"] = np.array(_labels_from_h5(h5) or [f"x{i}" for i in range(out["X0"].shape[1])])
            # optional extras
            for key, candidates in [("contact", CONTACT_KEYS), ("feet_pos", FEET_POS_KEYS), ("com", COM_KEYS)]:
                p = explicit.get(key, "")
                if p and p in h5:
                    out[key] = np.asarray(h5[p], float)
        else:
            X0, X1, U0, labels = _build_X0X1U0_from_raw(h5)
            out.update(dict(X0=X0, X1=X1, U0=U0, labels=np.array(labels)))
            if try_autodiscover:
                datasets = list(_iter_numeric_datasets(h5))
                pick_c = _pick_dataset_by_name(datasets, CONTACT_KEYS)
                pick_f = _pick_dataset_by_name(datasets, FEET_POS_KEYS)
                pick_c0 = _pick_dataset_by_name(datasets, COM_KEYS)
                if pick_c is not None:
                    out["contact"] = np.asarray(pick_c[1], float)
                if pick_f is not None:
                    v = np.asarray(pick_f[1], float)
                    # try to coerce to (N,4,3)
                    if v.ndim == 3 and v.shape[1] == 4 and v.shape[2] == 3:
                        out["feet_pos"] = v
                    elif v.ndim == 2 and v.shape[1] == 12:
                        out["feet_pos"] = v.reshape(v.shape[0], 4, 3)
                if pick_c0 is not None:
                    v = np.asarray(pick_c0[1], float)
                    if v.ndim == 2 and v.shape[1] >= 3:
                        out["com"] = v[:, :3]
    return out


# =========================================================
# 7) Main runner
# =========================================================
def run_edmd_and_save(
    h5_path: Path | str,
    out_group: str = "/eval2",
    p_max: int = 5,
    l2_reg: float = 1e-6,
    val_frac: float = 0.2,
    H: int = 20,
    seed: int = 42,
    # New options:
    input_mode: Literal["forces", "wrench", "forces+contact"] = "forces",
    include_contact_in_lift: bool = False,
    include_feet_in_lift: bool = False,
    mask_inputs_by_contact: bool = True,
    fit_C_linear: bool = True,
    l2_reg_C: float = 1e-8,
    decode_geom: bool = True,        # if False, uses learned C for readout
    explicit_paths: Optional[Dict[str, str]] = None,
    save_npz_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """
    Loads (or builds) X0/X1/U0 from an HDF5 file, fits EDMD with inputs on TRAIN split
    using YOUR geometric basis, evaluates an H-step open-loop rollout on a contiguous
    window in VAL split, writes results under `out_group` (and optional .npz), and
    returns a summary dict.

    New capabilities:
      - input_mode: "forces" (12-D), "wrench" (6-D), or "forces+contact" (learn stance dependence)
      - include_contact_in_lift / include_feet_in_lift: augment Φ with exogenous params
      - mask_inputs_by_contact: zero swing leg forces during fit
      - fit_C_linear: also learn C (x ≈ C Φ), useful if you don’t want geometric decode
      - decode_geom: toggle geometric vs linear readout at evaluation
    """
    rng = np.random.default_rng(seed)
    h5_path = Path(h5_path)
    data = load_from_h5(h5_path, try_autodiscover=True, explicit=explicit_paths)

    X0 = np.asarray(data["X0"], float)
    X1 = np.asarray(data["X1"], float)
    U0 = np.asarray(data["U0"], float)
    labels = _as_str_list(data.get("labels", [f"x{i}" for i in range(X0.shape[1])]))

    # Optional extras
    contact = data.get("contact", None)
    feet_pos = data.get("feet_pos", None)
    com_pos  = data.get("com", None)

    # Split (contiguous)
    N = X0.shape[0]
    N_val = max(1, int(round(val_frac * N)))
    N_tr  = max(1, N - N_val)
    tr_idx = slice(0, N_tr)
    va_idx = slice(N_tr, N)

    # Fit EDMD
    A, B, C, meta_fit = edmd_with_inputs(
        X=X0[tr_idx], Y=X1[tr_idx], U=U0[tr_idx],
        lift_fn=lambda xi: lift_1d(xi, p_max),
        l2_reg=l2_reg,
        contact=(None if contact is None else contact[tr_idx]),
        feet_pos=(None if feet_pos is None else feet_pos[tr_idx]),
        com_pos=(None  if com_pos  is None else com_pos[tr_idx]),
        include_contact_in_lift=include_contact_in_lift,
        include_feet_in_lift=include_feet_in_lift,
        mask_inputs_by_contact=mask_inputs_by_contact,
        input_mode=input_mode,
        fit_C_linear=fit_C_linear,
        l2_reg_C=l2_reg_C,
    )

    # Validation rollout
    Nv = X0[va_idx].shape[0]
    if Nv < 2:
        raise RuntimeError("Validation split too short to evaluate.")
    H_eff = min(H, Nv)
    x0   = X0[va_idx][0]
    Useq = U0[va_idx][:H_eff]
    Xtru = X1[va_idx][:H_eff]

    # Choose lift used at evaluation (must mirror training augmentation)
    def lift_eval(xi: np.ndarray) -> np.ndarray:
        Phi = lift_1d(xi, p_max)
        # note: at evaluation we don’t pass c/feet by default; if you trained with augmentation,
        # you should also pass the same here (e.g., from the validation segment)
        if include_contact_in_lift:
            if contact is None:
                raise ValueError("include_contact_in_lift=True but 'contact' not found in data.")
        if include_feet_in_lift:
            if feet_pos is None:
                raise ValueError("include_feet_in_lift=True but 'feet_pos' not found in data.")
        return Phi

    # Rollout with geometric or learned C
    if decode_geom:
        Xpred = rollout_open_loop(A, B, x0=x0, U_seq=Useq, lift_fn=lift_eval,
                                  C=None, decode_geom=True, p_max_for_geom=p_max)
    else:
        if C is None:
            raise ValueError("decode_geom=False requires fit_C_linear=True to provide C.")
        Xpred = rollout_open_loop(A, B, x0=x0, U_seq=Useq, lift_fn=lift_eval,
                                  C=C, decode_geom=False)

    # Error metrics (angle wrapping if 12D SRB convention)
    err = Xpred - Xtru
    if Xtru.shape[1] >= 6:
        err[:, 3:6] = _wrap_to_pi(err[:, 3:6])
    rmse_per_state = np.sqrt(np.mean(err**2, axis=0))
    overall_rmse   = float(np.sqrt(np.mean(rmse_per_state**2)))

    meta = {
        "N_train": int(N_tr), "N_val": int(N_val),
        "nx": int(X0.shape[1]), "nu": int(U0.shape[1]),
        "H": int(H_eff), "p_max": int(p_max),
        "l2_reg": float(l2_reg), "val_frac": float(val_frac), "seed": int(seed),
        "decode_geom": bool(decode_geom),
        "timestamp": time.time(),
    }
    meta.update(meta_fit)

    # Write back to H5 (under out_group)
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
        kg_path = "/koopman2"
        kg = h5.create_group(kg_path) if kg_path not in h5 else h5[kg_path]
        for nm in ["A", "B", "C"]:
            if nm in kg:
                del kg[nm]
        kg.create_dataset("A", data=A, compression="gzip")
        kg.create_dataset("B", data=B, compression="gzip")
        if C is not None:
            kg.create_dataset("C", data=C, compression="gzip")

    # Optional NPZ export (for direct controller consumption)
    if save_npz_path is not None:
        save_npz_path = Path(save_npz_path)
        np.savez_compressed(
            save_npz_path,
            A=A, B=B, **({"C": C} if C is not None else {}),
            meta=json.dumps(meta),
        )

    return {
        "X_true": Xtru, "X_pred": Xpred, "labels": labels,
        "rmse_per_state": rmse_per_state, "overall_rmse": overall_rmse,
        "meta": meta, "A": A, "B": B, "C": C,
    }
