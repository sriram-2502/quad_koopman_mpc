# eval_nstep.py
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Dict, Optional, Tuple

# ---- helpers consistent with your basis.observables layout ----
def obs_slices(p_max: int) -> Tuple[slice, slice, slice, slice]:
    """
    Your observables = [pos(3), lin_vel(3), R(9), omega_hat(9), psi_bar(...)]
    Return slices to extract those first 4 blocks from a lifted vector.
    """
    i = 0
    sl_pos = slice(i, i+3); i += 3
    sl_lin = slice(i, i+3); i += 3
    sl_R   = slice(i, i+9); i += 9
    sl_oh  = slice(i, i+9); i += 9
    return sl_pos, sl_lin, sl_R, sl_oh

def vee(S: np.ndarray) -> np.ndarray:
    """Inverse of hat: so(3)->R^3. Expects a 3x3 skew-symmetric."""
    return np.array([S[2,1], S[0,2], S[1,0]], dtype=float)

def nearest_SO3(M: np.ndarray) -> np.ndarray:
    """Project a 3x3 matrix to the closest rotation (det=+1) via SVD."""
    U, _, Vt = np.linalg.svd(M)
    Rproj = U @ Vt
    if np.linalg.det(Rproj) < 0:
        U[:, -1] *= -1
        Rproj = U @ Vt
    return Rproj

def wrap_to_pi(a: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi]."""
    return (a + np.pi) % (2*np.pi) - np.pi

def z_to_state12(z: np.ndarray, sl_pos, sl_lin, sl_R, sl_oh, degrees: bool=False) -> np.ndarray:
    """
    Decode 12-dim state [pos(3), eul(3), lin_vel(3), omega(3)]
    from a lifted feature vector z using your observable layout.
    """
    pos   = z[sl_pos]
    lin_v = z[sl_lin]
    Rm = nearest_SO3(z[sl_R].reshape(3,3))
    eul = R.from_matrix(Rm).as_euler('xyz', degrees=degrees)
    S = z[sl_oh].reshape(3,3)
    S = 0.5*(S - S.T)  # enforce skew
    omg = vee(S)
    return np.concatenate([pos, eul, lin_v, omg], axis=0)

def n_step_predict_open_loop(A: np.ndarray,
                             B: np.ndarray,
                             lift_fn,
                             x0: np.ndarray,
                             U_seq: np.ndarray,
                             p_max: int,
                             degrees: bool=False) -> np.ndarray:
    """
    Roll z_{k+1} = A z_k + B u_k for len(U_seq) steps and decode to 12-dim state each step.
    Returns: (H,12) predicted states.
    """
    sl_pos, sl_lin, sl_R, sl_oh = obs_slices(p_max)
    z = lift_fn(x0)
    preds = []
    for ut in U_seq:
        z = A @ z + B @ ut
        preds.append(z_to_state12(z, sl_pos, sl_lin, sl_R, sl_oh, degrees=degrees))
    return np.vstack(preds)  # (H,12)

def build_val_windows(ds,
                      val_mask: np.ndarray,
                      H: int) -> np.ndarray:
    """
    Return starting global indices for contiguous validation windows of length H.
    - ds.traj_ids: (N,) trajectory id per concatenated row
    - val_mask: boolean mask selecting validation rows in the concatenated arrays
    """
    idx_all = np.where(val_mask)[0]
    traj_ids = ds.traj_ids[val_mask]
    starts = []
    for tid in np.unique(traj_ids):
        idxs = idx_all[traj_ids == tid]
        if len(idxs) == 0:
            continue
        # split idxs into consecutive runs
        runs = np.split(idxs, np.where(np.diff(idxs) != 1)[0] + 1)
        for run in runs:
            if len(run) >= H:
                # all start indices that admit a window of length H
                for s in range(0, len(run) - H + 1):
                    starts.append(run[s])
    return np.array(starts, dtype=int)

def eval_nstep_window(A: np.ndarray,
                      B: np.ndarray,
                      lift_fn,
                      ds,
                      splits: Dict,
                      X0: np.ndarray,
                      X1: np.ndarray,
                      U0: np.ndarray,
                      p_max: int,
                      H: int,
                      start_index: Optional[int]=None,
                      degrees: bool=False,
                      rng: Optional[np.random.Generator]=None) -> Dict[str, np.ndarray]:
    """
    Select a contiguous validation window (length H), run open-loop N-step prediction,
    and return arrays ready for plotting.

    Returns dict:
      {
        't': (H,), 'X_true': (H,12), 'X_pred': (H,12),
        'X_true_plot': (H,12), 'X_pred_plot': (H,12),
        'labels': list[str], 'start_index': int
      }
    """
    if 'val_mask' not in splits:
        raise ValueError("splits must contain 'val_mask'.")
    val_mask = splits['val_mask']

    starts = build_val_windows(ds, val_mask, H)
    if len(starts) == 0:
        raise RuntimeError("No contiguous validation windows of length H found. Reduce H or adjust split.")

    if start_index is None:
        rng = np.random.default_rng() if rng is None else rng
        t0 = int(rng.choice(starts))
    else:
        t0 = int(start_index)

    # assemble window data
    x0 = X0[t0]                # (12,)
    U_seq = U0[t0:t0+H]        # (H, nu)
    X_true = X1[t0:t0+H]       # (H, 12)
    X_pred = n_step_predict_open_loop(A, B, lift_fn, x0, U_seq, p_max, degrees=degrees)  # (H,12)

    # unwrap Euler for smooth plotting (align pred to GT trajectory)
    X_true_plot = X_true.copy()
    X_pred_plot = X_pred.copy()
    X_true_plot[:, 3:6] = np.unwrap(X_true_plot[:, 3:6], axis=0)
    eul_diff = wrap_to_pi(X_pred[:, 3:6] - X_true[:, 3:6])
    X_pred_plot[:, 3:6] = X_true_plot[:, 3:6] + np.unwrap(eul_diff, axis=0)

    labels = [
        "pos.x [m]", "pos.y [m]", "pos.z [m]",
        "roll [rad]", "pitch [rad]", "yaw [rad]",
        "v.x [m/s]", "v.y [m/s]", "v.z [m/s]",
        "ω.x [rad/s]", "ω.y [rad/s]", "ω.z [rad/s]"
    ]

    out = {
        "t": np.arange(1, H+1),
        "X_true": X_true,
        "X_pred": X_pred,
        "X_true_plot": X_true_plot,
        "X_pred_plot": X_pred_plot,
        "labels": labels,
        "start_index": np.array([t0], dtype=int),
    }
    return out

# -------- optional: batch evaluation (RMSE vs horizon over many windows) --------
def _rmse_acc_init(H: int, d: int) -> Dict[str, np.ndarray]:
    return {"se": np.zeros((H, d)), "n": np.zeros(H, dtype=int)}

def eval_nstep_batch(A: np.ndarray,
                     B: np.ndarray,
                     lift_fn,
                     ds,
                     splits: Dict,
                     X0: np.ndarray,
                     X1: np.ndarray,
                     U0: np.ndarray,
                     p_max: int,
                     H: int,
                     max_windows: int = 200,
                     degrees: bool=False,
                     seed: Optional[int]=0) -> Dict[str, np.ndarray]:
    """
    Evaluate N-step open-loop prediction across many validation windows.
    Returns per-horizon RMSE arrays for the 4 state groups and their scalar summaries.

    Output dict keys:
      'pos_rmse_H', 'eul_rmse_H', 'lin_rmse_H', 'omg_rmse_H'  -> shape (H,3)
      'pos_rmse_scalar', 'eul_rmse_scalar', 'lin_rmse_scalar', 'omg_rmse_scalar' -> (H,)
    """
    if 'val_mask' not in splits:
        raise ValueError("splits must contain 'val_mask'.")
    val_mask = splits['val_mask']

    starts = build_val_windows(ds, val_mask, H)
    if len(starts) == 0:
        raise RuntimeError("No contiguous validation windows of length H found. Reduce H or adjust split.")

    rng = np.random.default_rng(seed)
    rng.shuffle(starts)
    starts = starts[:min(max_windows, len(starts))]

    acc_pos = _rmse_acc_init(H, 3)
    acc_eul = _rmse_acc_init(H, 3)
    acc_lin = _rmse_acc_init(H, 3)
    acc_omg = _rmse_acc_init(H, 3)

    for t0 in starts:
        x0 = X0[t0]
        U_seq = U0[t0:t0+H]
        X_true = X1[t0:t0+H]
        X_pred = n_step_predict_open_loop(A, B, lift_fn, x0, U_seq, p_max, degrees=degrees)

        e_pos = X_pred[:, 0:3] - X_true[:, 0:3]
        e_eul = wrap_to_pi(X_pred[:, 3:6] - X_true[:, 3:6])
        e_lin = X_pred[:, 6:9] - X_true[:, 6:9]
        e_omg = X_pred[:, 9:12] - X_true[:, 9:12]

        acc_pos["se"] += e_pos**2; acc_pos["n"] += 1
        acc_eul["se"] += e_eul**2; acc_eul["n"] += 1
        acc_lin["se"] += e_lin**2; acc_lin["n"] += 1
        acc_omg["se"] += e_omg**2; acc_omg["n"] += 1

    # per-horizon vector RMSE (H,3)
    pos_rmse_H = np.sqrt(acc_pos["se"] / acc_pos["n"][:, None])
    eul_rmse_H = np.sqrt(acc_eul["se"] / acc_eul["n"][:, None])
    lin_rmse_H = np.sqrt(acc_lin["se"] / acc_lin["n"][:, None])
    omg_rmse_H = np.sqrt(acc_omg["se"] / acc_omg["n"][:, None])

    # scalar RMSE per horizon (combine axes)
    pos_rmse_scalar = np.sqrt(np.mean(pos_rmse_H**2, axis=1))
    eul_rmse_scalar = np.sqrt(np.mean(eul_rmse_H**2, axis=1))
    lin_rmse_scalar = np.sqrt(np.mean(lin_rmse_H**2, axis=1))
    omg_rmse_scalar = np.sqrt(np.mean(omg_rmse_H**2, axis=1))

    return {
        "pos_rmse_H": pos_rmse_H,
        "eul_rmse_H": eul_rmse_H,
        "lin_rmse_H": lin_rmse_H,
        "omg_rmse_H": omg_rmse_H,
        "pos_rmse_scalar": pos_rmse_scalar,
        "eul_rmse_scalar": eul_rmse_scalar,
        "lin_rmse_scalar": lin_rmse_scalar,
        "omg_rmse_scalar": omg_rmse_scalar,
    }
