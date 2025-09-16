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
