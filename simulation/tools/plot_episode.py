# simulation/tools/plot_episode.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import h5py

def _to_1d(a):
    a = np.asarray(a)
    return a.reshape(-1)

def _same_len(*arrs):
    L = min(len(a) for a in arrs)
    return [a[:L] for a in arrs]

def _load_episode_array(f: h5py.File, name: str, ep: int) -> Optional[np.ndarray]:
    """Return episode array (T,...) or None if not present."""
    try:
        ds = f["recordings"][name]
        return np.asarray(ds[ep])
    except Exception:
        return None

def _contiguous_spans(mask_1d: np.ndarray, t: np.ndarray):
    """Return [(t_start, t_end), ...] for contiguous True regions in mask_1d."""
    spans = []
    in_run = False
    start_i = 0
    for i, val in enumerate(mask_1d.astype(bool)):
        if val and not in_run:
            in_run = True; start_i = i
        elif not val and in_run:
            in_run = False; spans.append((t[start_i], t[i-1]))
    if in_run:
        spans.append((t[start_i], t[len(mask_1d)-1]))
    return spans

def _reconstruct_u_ref_from_stance(stance_mask: np.ndarray, mg: float) -> np.ndarray:
    """
    stance_mask: (T,4) 1/0
    Returns Uref: (T,4,3) with only fz populated as mg/#stance per step.
    """
    T = stance_mask.shape[0]
    Uref = np.zeros((T, 4, 3), float)
    for t in range(T):
        s = stance_mask[t].astype(int)
        nstance = int(s.sum())
        if nstance > 0:
            fz_each = mg / nstance
            for leg in range(4):
                if s[leg] == 1:
                    Uref[t, leg, 2] = fz_each
    return Uref

def plot_episode_grfs(
    h5_path: Path | str,
    episode_idx: int = 0,
    prefer_cmd: bool = True,
    fz_threshold: float = 5.0,
    fz_ylim: Optional[Tuple[float, float]] = None,
    title_suffix: str = "",
):
    """
    Plot per-leg GRFs (fx,fy,fz) vs reconstructed reference (mg/#stance) with stance shading.

    - Uses 'nmpc_GRFs' if available (prefer_cmd=True), else 'contact_forces'.
    - Stance comes from 'contact_state' if present (T,4) booleans/ints, otherwise fz>fz_threshold.
    - Robust handling for (T,1) time, non-monotonic time (sort), and array length mismatches.
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    with h5py.File(h5_path, "r") as f:
        t = _load_episode_array(f, "time", episode_idx)
        if t is None:
            raise RuntimeError("Missing '/recordings/time' for the selected episode.")
        t = _to_1d(t)

        # Choose GRFs to plot
        U = None
        if prefer_cmd:
            U = _load_episode_array(f, "nmpc_GRFs", episode_idx)
        if U is None:
            # fallback to measured forces
            U = _load_episode_array(f, "contact_forces", episode_idx)
        if U is None:
            raise RuntimeError("Neither 'nmpc_GRFs' nor 'contact_forces' found in H5.")

        U = np.asarray(U, float)
        if U.ndim == 2 and U.shape[1] == 12:
            # (T,12) -> (T,4,3)
            U = np.stack([U[:, 0:3], U[:, 3:6], U[:, 6:9], U[:, 9:12]], axis=1)
        elif U.ndim != 3 or U.shape[1:] != (4, 3):
            raise ValueError(f"Unexpected GRF array shape: {U.shape} (expect (T,4,3) or (T,12))")

        # Stance mask: prefer dataset if present
        stance = _load_episode_array(f, "contact_state", episode_idx)
        if stance is None:
            stance = _load_episode_array(f, "feet_contact_state", episode_idx)
        if stance is not None:
            stance = (np.asarray(stance) > 0).astype(int)
            if stance.shape[1] != 4:
                # Try to fix transposed shapes (T,4) expected
                if stance.shape[0] == 4:
                    stance = stance.T
        else:
            # derive from fz threshold
            stance = (U[:, :, 2] > fz_threshold).astype(int)  # (T,4)

    # Make time strictly 1D and monotonic; if not, sort
    order = np.argsort(t)
    t = t[order]
    U = U[order]
    stance = stance[order]

    # Trim to same length
    T = len(t)
    U = U[:T]
    stance = stance[:T]

    # Build Uref from stance
    try:
        # lazy import to avoid hard dep here
        from quadruped_pympc import config as cfg
        mg = cfg.mass * cfg.gravity_constant
    except Exception:
        mg = 0.0
    Uref = _reconstruct_u_ref_from_stance(stance, mg)

    # Axis limits for fz (if not provided)
    if fz_ylim is None:
        # Ignore zeros when all are swing
        fz_vals = U[:, :, 2].reshape(-1)
        # safeguard if all zero
        hi = np.nanpercentile(fz_vals[fz_vals > 1e-6], 99) if np.any(fz_vals > 1e-6) else 50.0
        fz_ylim = (-10.0, max(10.0, hi * 1.1))

    # Plot
    legs = ["FL", "FR", "RL", "RR"]
    comps = ["fx", "fy", "fz"]

    fig, axes = plt.subplots(4, 3, figsize=(14, 9), sharex=True)
    for i_leg, leg in enumerate(legs):
        # build stance spans per leg for clean shading
        spans = _contiguous_spans(stance[:, i_leg].astype(bool), t)
        for j_comp, comp in enumerate(comps):
            ax = axes[i_leg, j_comp]
            y = U[:, i_leg, j_comp]
            yref = Uref[:, i_leg, j_comp]
            ax.plot(t, y, label=f"{leg} {comp}", linewidth=1.2)
            ax.plot(t, yref, "--", label=f"{leg} {comp}_ref", linewidth=1.0)

            # shade stance spans
            for (ts, te) in spans:
                ax.axvspan(ts, te, color="0.9", alpha=0.35)

            if comp == "fz":
                ax.set_ylim(*fz_ylim)

            ax.grid(True, linestyle=":")
            if i_leg == 0:
                ax.set_title(comp)
            if j_comp == 0:
                ax.set_ylabel(f"{leg}  (N)")
            if i_leg == 0 and j_comp == 0:
                ax.legend(fontsize=8, loc="best")

    axes[-1, 1].set_xlabel("time (s)")
    ttl = "Per-leg GRFs vs reconstructed reference (stance shaded)"
    if title_suffix:
        ttl += f" — {title_suffix}"
    fig.suptitle(ttl)
    plt.tight_layout()
    plt.show()

    # Total vertical support
    fz_sum = U[:, :, 2].sum(axis=1)
    fz_ref_sum = Uref[:, :, 2].sum(axis=1)
    plt.figure(figsize=(10, 3))
    plt.plot(t, fz_sum, label="Σ fz")
    plt.plot(t, fz_ref_sum, "--", label="Σ fz_ref")
    if mg > 0:
        plt.plot(t, np.full_like(t, mg), ":", label="mg")
    plt.grid(True, linestyle=":")
    plt.xlabel("time (s)")
    plt.ylabel("N")
    plt.title("Total vertical support")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_episode_states(
    h5_path: Path | str,
    episode_idx: int = 0,
    *,
    center_xy: bool = False,
    t0: Optional[float] = None,
    duration: Optional[float] = None,
    unwrap_yaw: bool = False,
    deg: bool = False,
    title_suffix: str = "",
):
    """
    Plot base states (x,y,z), Euler angles (roll,pitch,yaw), linear and angular velocities for one episode.

    Args:
        h5_path: HDF5 file path.
        episode_idx: episode index to load.
        center_xy: if True, subtract (x0,y0) so position starts at (0,0).
        t0: optional start time (s) for slicing.
        duration: optional duration (s) from t0; if None, uses full range.
        unwrap_yaw: if True, unwrap yaw for continuity (good for long turns).
        deg: if True, plot angles in degrees (default radians).
        title_suffix: extra text to append to the figure title.

    Notes:
        - Requires datasets under '/recordings': time, base_pos, base_ori_euler_xyz,
          base_lin_vel, base_ang_vel.
        - Time is sorted if non-monotonic; all arrays are trimmed to the shortest length.
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    with h5py.File(h5_path, "r") as f:
        t = _load_episode_array(f, "time", episode_idx)
        if t is None:
            raise RuntimeError("Missing '/recordings/time' for the selected episode.")
        t = _to_1d(t)

        pos  = _load_episode_array(f, "base_pos", episode_idx)              # (T,3)
        eul  = _load_episode_array(f, "base_ori_euler_xyz", episode_idx)    # (T,3)
        vlin = _load_episode_array(f, "base_lin_vel", episode_idx)          # (T,3)
        vang = _load_episode_array(f, "base_ang_vel", episode_idx)          # (T,3)

        for name, arr in [("base_pos", pos), ("base_ori_euler_xyz", eul),
                          ("base_lin_vel", vlin), ("base_ang_vel", vang)]:
            if arr is None:
                raise RuntimeError(f"Missing '/recordings/{name}' for episode {episode_idx}.")

        # Ensure 2D shapes
        pos  = np.asarray(pos,  float).reshape(-1, 3)
        eul  = np.asarray(eul,  float).reshape(-1, 3)
        vlin = np.asarray(vlin, float).reshape(-1, 3)
        vang = np.asarray(vang, float).reshape(-1, 3)

    # Sort by time if needed
    order = np.argsort(t)
    t     = t[order]
    pos   = pos[order]
    eul   = eul[order]
    vlin  = vlin[order]
    vang  = vang[order]

    # Trim to common length
    t, pos, eul, vlin, vang = _same_len(t, pos, eul, vlin, vang)

    # Apply optional time window
    tmin, tmax = float(t[0]), float(t[-1])
    if duration is None and t0 is None:
        mask = np.ones_like(t, dtype=bool)
        t0_show, t1_show = tmin, tmax
    else:
        if t0 is None:
            t0 = tmin
        t1 = t0 + (duration if duration is not None else (tmax - t0))
        t0_show, t1_show = max(tmin, t0), min(tmax, t1)
        mask = (t >= t0_show) & (t <= t1_show)
        if not np.any(mask):
            raise ValueError(f"No samples in requested window [{t0_show:.3f}, {t1_show:.3f}] s")

    t     = t[mask]
    pos   = pos[mask]
    eul   = eul[mask]
    vlin  = vlin[mask]
    vang  = vang[mask]

    # Optional centering of x,y to start at zero
    if center_xy and len(pos) > 0:
        anchor_xy = pos[0, :2].copy()
        pos[:, 0] -= anchor_xy[0]
        pos[:, 1] -= anchor_xy[1]

    # Optional yaw unwrap and angle units
    if unwrap_yaw and len(eul) > 0:
        eul[:, 2] = np.unwrap(eul[:, 2], discont=np.pi)
    if deg:
        eul = np.degrees(eul)

    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    # 1) Position
    axes[0].plot(t, pos[:, 0], label="x")
    axes[0].plot(t, pos[:, 1], label="y")
    axes[0].plot(t, pos[:, 2], label="z")
    axes[0].set_ylabel("pos (m)")
    axes[0].grid(True, linestyle=":")
    axes[0].legend(ncol=3)

    # 2) Euler angles
    aunit = "deg" if deg else "rad"
    axes[1].plot(t, eul[:, 0], label=f"roll ({aunit})")
    axes[1].plot(t, eul[:, 1], label=f"pitch ({aunit})")
    axes[1].plot(t, eul[:, 2], label=f"yaw ({aunit})")
    axes[1].set_ylabel(f"Euler ({aunit})")
    axes[1].grid(True, linestyle=":")
    axes[1].legend(ncol=3)

    # 3) Linear velocity
    axes[2].plot(t, vlin[:, 0], label="vx")
    axes[2].plot(t, vlin[:, 1], label="vy")
    axes[2].plot(t, vlin[:, 2], label="vz")
    axes[2].set_ylabel("lin vel (m/s)")
    axes[2].grid(True, linestyle=":")
    axes[2].legend(ncol=3)

    # 4) Angular velocity
    axes[3].plot(t, vang[:, 0], label="wx")
    axes[3].plot(t, vang[:, 1], label="wy")
    axes[3].plot(t, vang[:, 2], label="wz")
    axes[3].set_ylabel("ang vel (rad/s)")
    axes[3].set_xlabel("time (s)")
    axes[3].grid(True, linestyle=":")
    axes[3].legend(ncol=3)

    ttl = f"Episode {episode_idx} — base states"
    if center_xy:
        ttl += " (centered x,y)"
    if (duration is not None) or (t0 is not None):
        ttl += f" [{t0_show:.2f}, {t1_show:.2f}] s"
    if title_suffix:
        ttl += f" — {title_suffix}"
    fig.suptitle(ttl)
    plt.tight_layout()
    plt.show()
