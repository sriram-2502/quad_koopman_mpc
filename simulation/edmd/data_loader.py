# loader.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List, Any, Sequence, Literal
from pathlib import Path
from h5_reader import H5Reader


def _repeat4_to_12(c4: np.ndarray) -> np.ndarray:
    """(T,4)->(T,12) expand leg mask across [fx,fy,fz]"""
    c4 = np.asarray(c4, float)
    # Accept (4,) and (T,4) and repeat per-force axis.
    if c4.ndim == 1:
        c4 = c4.reshape(-1, 4)
    return np.repeat(c4, repeats=3, axis=1)

def _ensure_feet_shape(feet: np.ndarray) -> np.ndarray:
    """Coerce feet to (T,4,3) from (T,12) if needed."""
    feet = np.asarray(feet, float)
    # H5 logs sometimes flatten feet positions to 12.
    if feet.ndim == 2 and feet.shape[1] == 12:
        return feet.reshape(feet.shape[0], 4, 3)
    assert feet.ndim == 3 and feet.shape[1:] == (4, 3), f"feet_pos must be (T,4,3) or (T,12), got {feet.shape}"
    return feet


def _ensure_grf_shape(grfs: np.ndarray) -> np.ndarray:
    """Coerce GRFs to (T,12) from (T,4,3) if needed."""
    grfs = np.asarray(grfs, float)
    # Accept (T,4,3) and flatten to (T,12) for learning.
    if grfs.ndim == 3 and grfs.shape[1:] == (4, 3):
        return grfs.reshape(grfs.shape[0], 12)
    assert grfs.ndim == 2 and grfs.shape[1] == 12, f"GRFs must be (T,12) or (T,4,3), got {grfs.shape}"
    return grfs


def _downsample(arr: Optional[np.ndarray], ds: int) -> Optional[np.ndarray]:
    """Downsample 1D/2D/3D arrays by stride ds (no-op if ds <= 1 or None)."""
    # Keep caller logic simple by handling None here.
    if arr is None or ds <= 1:
        return arr
    return arr[::ds]


def _trim_to_min_length(arrs: Sequence[Optional[np.ndarray]]) -> Tuple[int, List[Optional[np.ndarray]]]:
    """Trim all non-None arrays to the shortest length and return (T, trimmed)."""
    # Enforce consistent T across signals per trajectory.
    lens = [len(a) for a in arrs if a is not None]
    if not lens:
        return 0, list(arrs)
    T = min(lens)
    return T, [a[:T] if a is not None else None for a in arrs]


def _apply_mask(arr: Optional[np.ndarray], mask: np.ndarray) -> Optional[np.ndarray]:
    """Apply a boolean mask to an array if it exists."""
    # Useful for masking optional extras alongside X/U.
    return arr[mask] if arr is not None else None


def _wrench_from_forces(feet_pos_world_t: np.ndarray, com_world_t: np.ndarray, u_12_t: np.ndarray) -> np.ndarray:
    """
    One-timestep wrench: feet_pos_world_t (4,3), com_world_t (3,), u_12_t (12,)
    returns w = [Fx,Fy,Fz, Mx,My,Mz]
    """
    # Sum forces and moments around COM.
    F = np.zeros(3)
    M = np.zeros(3)
    for j in range(4):
        f = u_12_t[3*j:3*j+3]
        r = feet_pos_world_t[j] - com_world_t
        F += f
        M += np.cross(r, f)
    return np.concatenate([F, M], axis=0)


@dataclass
class QuadrupedEDMDDataset:
    file_path: Optional[str] = None
    dataset: Optional[H5Reader] = None
    downsample: int = 1
    pos_zero_start: bool = True
    drop_nan: bool = True
    # Optionally skip the first N seconds of each trajectory.
    trim_head_sec: float = 0.0
    # Optional timestep (seconds) used when time is not logged.
    sample_dt: Optional[float] = None

    # What to use as inputs:
    # - "wrench": (default) 6-D centroidal wrench [Fx,Fy,Fz,Mx,My,Mz] computed from forces+feet+com
    # - "forces": 12-D GRFs [fx,fy,fz]*4
    input_mode: Literal["forces", "wrench"] = "wrench"
    # If contact is present, mask swing legs' forces before using them or computing wrench
    mask_inputs_by_contact: bool = True
    # If True, use body-frame angular velocity when available (linear remains world frame)
    use_body_ang_vel: bool = True

    # HDF5 keys used to read arrays (override as needed)
    keys: Dict[str, str] = field(default_factory=lambda: {
        "pos": "base_pos",                       # (T,3)
        "eul": "base_ori_euler_xyz",             # (T,3) radians
        "lin_vel": "base_lin_vel",               # (T,3) world frame
        "ang_vel": "base_ang_vel",               # (T,3) world frame
        "ang_vel_body": "base_ang_vel_body",     # (T,3) body frame (optional)
        "u": "nmpc_GRFs",                        # (T,4,3) control GRFs; override to "contact_forces" if desired
        "time": "time",                          # (T,) or (T,1) time stamps (optional)
        # Optional extras:
        "contact": "contact_state",              # (T,4) {0,1} stance flags (if available)
        "feet_pos": "feet_pos",                  # (T,4,3) or (T,12) feet positions in world frame
        "cmd_base_lin_vel": "cmd_base_lin_vel",  # (T,3) commanded base linear velocity
        "cmd_base_ang_vel": "cmd_base_ang_vel",  # (T,3) commanded base angular velocity
    })

    # Outputs after build()
    X0: Optional[np.ndarray] = None  # (N, 12)
    X1: Optional[np.ndarray] = None  # (N, 12)
    U0: Optional[np.ndarray] = None  # (N, 12) or (N, 6) depending on input_mode
    episode_bounds: Optional[List[Tuple[int, int]]] = None  # global index per episode
    traj_ids: Optional[np.ndarray] = None  # (N,) trajectory index per sample

    # Extras aligned with X0/U0 for evaluation
    contact0: Optional[np.ndarray] = None  # (N,4) if available
    feet0: Optional[np.ndarray] = None     # (N,4,3) if available
    com0: Optional[np.ndarray] = None      # (N,3) base_pos (COM approximation)
    x_ref0: Optional[np.ndarray] = None    # (N,6) [cmd_lin_vel, cmd_ang_vel] aligned with X0
    x_ref1: Optional[np.ndarray] = None    # (N,6) [cmd_lin_vel, cmd_ang_vel] aligned with X1
    x_ref: Optional[np.ndarray] = None     # (N,6) kept for backward compatibility (alias of x_ref0)

    def _need_reader(self):
        """Ensure the H5Reader is initialized."""
        # Lazy-open the dataset only when needed.
        if self.dataset is None:
            if self.file_path is None:
                raise ValueError("Provide either dataset=H5Reader(...) or file_path to construct one.")
            self.dataset = H5Reader(file_path=str(Path(self.file_path)))

    def _read_required(self, rec: Any, logical_key: str, traj_id: int) -> np.ndarray:
        """Read a required dataset key for a given trajectory."""
        # Missing required keys should fail fast.
        return np.asarray(rec[self.keys[logical_key]][traj_id])

    def _read_optional(self, rec: Any, logical_key: str, traj_id: int) -> Optional[np.ndarray]:
        """Read an optional dataset key; return None if missing."""
        # Optional keys are allowed to be absent in H5.
        k = self.keys.get(logical_key)
        if not isinstance(k, str):
            return None
        try:
            return np.asarray(rec[k][traj_id])
        except KeyError:
            return None

    def _get_traj_arrays(
        self, traj_id: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
            Optional[np.ndarray], Optional[np.ndarray],
            Optional[np.ndarray], Optional[np.ndarray]]:
        """Load one trajectory worth of arrays and align them to a common length."""
        rec = self.dataset.recordings  # type: ignore[attr-defined]

        # Base state (pos/eul + velocities) with optional body-frame preference.
        base_pos = self._read_required(rec, "pos", traj_id)  # world frame
        base_eul = self._read_required(rec, "eul", traj_id)  # euler angles xyz
        base_lin = self._read_required(rec, "lin_vel", traj_id)
        if self.use_body_ang_vel:
            base_ang = self._read_optional(rec, "ang_vel_body", traj_id)
            if base_ang is None:
                base_ang = self._read_required(rec, "ang_vel", traj_id)
        else:
            base_ang = self._read_required(rec, "ang_vel", traj_id)
        u        = _ensure_grf_shape(self._read_required(rec, "u", traj_id))

        # Optional extras used for masking and wrench computation.
        contact = self._read_optional(rec, "contact", traj_id)
        feet_pos = self._read_optional(rec, "feet_pos", traj_id)
        if feet_pos is not None:
            feet_pos = _ensure_feet_shape(feet_pos)  # (T,4,3) or reshape from (T,12)
        cmd_lin = self._read_optional(rec, "cmd_base_lin_vel", traj_id)
        cmd_ang = self._read_optional(rec, "cmd_base_ang_vel", traj_id)
        time = self._read_optional(rec, "time", traj_id) if self.trim_head_sec > 0 else None

        # Downsample all signals by a common stride.
        ds = max(1, int(self.downsample))
        base_pos = _downsample(base_pos, ds)
        base_eul = _downsample(base_eul, ds)
        base_lin = _downsample(base_lin, ds)
        base_ang = _downsample(base_ang, ds)
        u        = _downsample(u, ds)
        contact  = _downsample(contact, ds)
        feet_pos = _downsample(feet_pos, ds)
        cmd_lin  = _downsample(cmd_lin, ds)
        cmd_ang  = _downsample(cmd_ang, ds)
        time     = _downsample(time, ds)
        if time is not None:
            time = np.asarray(time, float).reshape(-1)

        # Trim to the shortest available signal length.
        arrays = [base_pos, base_eul, base_lin, base_ang, u, contact, feet_pos, cmd_lin, cmd_ang]
        if time is not None:
            arrays = arrays + [time]
            T, arrays = _trim_to_min_length(arrays)
            base_pos, base_eul, base_lin, base_ang, u, contact, feet_pos, cmd_lin, cmd_ang, time = arrays
        else:
            T, arrays = _trim_to_min_length(arrays)
            base_pos, base_eul, base_lin, base_ang, u, contact, feet_pos, cmd_lin, cmd_ang = arrays

        # Optional head trim (e.g., skip first 1s of each trajectory).
        if self.trim_head_sec > 0:
            if time is None and self.sample_dt is None:
                raise ValueError("trim_head_sec requires a 'time' dataset or sample_dt to be set.")
            if time is not None:
                t0 = float(time[0])
                start_idx = int(np.searchsorted(time, t0 + self.trim_head_sec, side="left"))
            else:
                start_idx = int(round(self.trim_head_sec / float(self.sample_dt)))
            if start_idx > 0:
                base_pos = base_pos[start_idx:]
                base_eul = base_eul[start_idx:]
                base_lin = base_lin[start_idx:]
                base_ang = base_ang[start_idx:]
                u        = u[start_idx:]
                contact  = contact[start_idx:] if contact is not None else None
                feet_pos = feet_pos[start_idx:] if feet_pos is not None else None
                cmd_lin  = cmd_lin[start_idx:] if cmd_lin is not None else None
                cmd_ang  = cmd_ang[start_idx:] if cmd_ang is not None else None
                if time is not None:
                    time = time[start_idx:]
                T = len(base_pos)

        # Zero-start position for more stable learning.
        if self.pos_zero_start and T > 0:
            base_pos = base_pos - base_pos[0:1, :]

        return base_pos, base_eul, base_lin, base_ang, u, contact, feet_pos, cmd_lin, cmd_ang


    def build(self) -> "QuadrupedEDMDDataset":
        """Construct time-shifted pairs X0, X1, U0 across episodes.
           Also returns extras aligned to X0/U0: contact0, feet0, com0, x_ref0/x_ref1 (if present)."""
        self._need_reader()
        n_traj: int = int(self.dataset.n_trajectories)  # type: ignore[attr-defined]

        # Accumulate per-trajectory arrays, then concat.
        X0_list: List[np.ndarray] = []
        X1_list: List[np.ndarray] = []
        U0_list: List[np.ndarray] = []
        bounds: List[Tuple[int, int]] = []
        traj_ids_joined: List[np.ndarray] = []

        contact0_list: List[np.ndarray] = []
        feet0_list: List[np.ndarray] = []
        com0_list: List[np.ndarray] = []
        x_ref0_list: List[np.ndarray] = []
        x_ref1_list: List[np.ndarray] = []

        cursor = 0
        for k in range(n_traj):
            # Load and assemble state/control per episode.
            pos, eul, lin, ang, u, contact, feet_pos, cmd_lin, cmd_ang = self._get_traj_arrays(k)
            state = np.concatenate([pos, eul, lin, ang], axis=-1)  # (T,12)

            if len(state) < 2:
                continue

            # Align time-shifted pairs for one-step dynamics.
            X0_k = state[:-1]
            X1_k = state[1:]
            U0_k = u[:-1]
            contact_prev = contact[:-1] if contact is not None else None
            cmd_lin_prev = cmd_lin[:-1] if cmd_lin is not None else None
            cmd_ang_prev = cmd_ang[:-1] if cmd_ang is not None else None
            cmd_lin_next = cmd_lin[1:] if cmd_lin is not None else None
            cmd_ang_next = cmd_ang[1:] if cmd_ang is not None else None
            x_ref0_prev = None
            x_ref1_prev = None
            if cmd_lin_prev is not None and cmd_ang_prev is not None:
                x_ref0_prev = np.concatenate([cmd_lin_prev, cmd_ang_prev], axis=-1)
            if cmd_lin_next is not None and cmd_ang_next is not None:
                x_ref1_prev = np.concatenate([cmd_lin_next, cmd_ang_next], axis=-1)

            # Optional masking of inputs using contact (swing=0).
            if self.mask_inputs_by_contact and (contact_prev is not None):
                C12k = _repeat4_to_12(contact_prev)
                U0_k = U0_k * C12k

            # If we want wrench, use base_pos as COM approximation.
            if self.input_mode == "wrench":
                if feet_pos is None:
                    raise ValueError("input_mode='wrench' requires feet_pos in dataset (key 'feet_pos').")
                W = np.zeros((U0_k.shape[0], 6), float)
                for t in range(U0_k.shape[0]):
                    W[t] = _wrench_from_forces(feet_pos[t], pos[t], U0_k[t])
                U0_k = W  # replace forces with wrench
            feet_prev = feet_pos[:-1] if feet_pos is not None else None
            com_prev = pos[:-1]

            # Drop NaNs across all aligned signals.
            if self.drop_nan:
                mask = np.isfinite(X0_k).all(axis=1) & np.isfinite(X1_k).all(axis=1) & np.isfinite(U0_k).all(axis=1)
                X0_k, X1_k, U0_k = X0_k[mask], X1_k[mask], U0_k[mask]
                contact_k = _apply_mask(contact_prev, mask)
                feet_k = _apply_mask(feet_prev, mask)
                com_k = _apply_mask(com_prev, mask)
                x_ref0_k = _apply_mask(x_ref0_prev, mask)
                x_ref1_k = _apply_mask(x_ref1_prev, mask)
            else:
                contact_k = contact_prev
                feet_k = feet_prev
                com_k = com_prev
                x_ref0_k = x_ref0_prev
                x_ref1_k = x_ref1_prev

            if X0_k.size == 0:
                continue

            X0_list.append(X0_k)
            X1_list.append(X1_k)
            U0_list.append(U0_k)

            if contact_k is not None:
                contact0_list.append(contact_k.astype(float, copy=False))
            if feet_k is not None:
                feet0_list.append(feet_k.astype(float, copy=False))
            if com_k is not None:
                com0_list.append(com_k.astype(float, copy=False))
            if x_ref0_k is not None:
                x_ref0_list.append(x_ref0_k.astype(float, copy=False))
            if x_ref1_k is not None:
                x_ref1_list.append(x_ref1_k.astype(float, copy=False))

            start = cursor
            cursor += len(X0_k)
            bounds.append((start, cursor))
            traj_ids_joined.append(np.full((len(X0_k),), k, dtype=np.int32))

        if not X0_list:
            raise RuntimeError("No valid trajectories found after preprocessing.")

        self.X0 = np.concatenate(X0_list, axis=0).astype(np.float64, copy=False)
        self.X1 = np.concatenate(X1_list, axis=0).astype(np.float64, copy=False)
        self.U0 = np.concatenate(U0_list, axis=0).astype(np.float64, copy=False)
        self.episode_bounds = bounds
        self.traj_ids = np.concatenate(traj_ids_joined, axis=0)
        self.x_ref0   = np.concatenate(x_ref0_list, axis=0) 
        self.x_ref1   = np.concatenate(x_ref1_list, axis=0) 

        # Extras aligned to X0/U0 (may be missing).
        self.contact0 = np.concatenate(contact0_list, axis=0) if contact0_list else None
        self.feet0    = np.concatenate(feet0_list, axis=0)    if feet0_list    else None
        self.com0     = np.concatenate(com0_list, axis=0)     if com0_list     else None
        
        # Backward compatibility just in case
        self.x_ref    = self.x_ref0
        
        # If com missing, use base position subset (already in X0 as first 3 dims)
        if self.com0 is None:
            self.com0 = self.X0[:, 0:3].copy()

        return self

    def train_val_split_by_traj(self, val_frac: float = 0.2, seed: int = 0
                                ) -> Dict[str, Dict[str, np.ndarray]]:
        """Split the built dataset by whole trajectories into train/val sets."""
        # Keep trajectories intact across splits.
        if self.traj_ids is None:
            raise ValueError("Call build() before splitting.")
        rng = np.random.default_rng(seed)
        all_traj = np.unique(self.traj_ids)
        rng.shuffle(all_traj)
        n_val = max(1, int(len(all_traj) * val_frac))
        val_traj = set(all_traj[:n_val])
        train_mask = ~np.isin(self.traj_ids, list(val_traj))
        val_mask = ~train_mask

        split = {
            "train": {"X0": self.X0[train_mask], "X1": self.X1[train_mask], "U0": self.U0[train_mask]},
            "val":   {"X0": self.X0[val_mask],   "X1": self.X1[val_mask],   "U0": self.U0[val_mask]},
            "train_mask": train_mask,
            "val_mask": val_mask,
        }

        # propagate extras if available
        if self.contact0 is not None:
            split["train"]["contact"] = self.contact0[train_mask]
            split["val"]["contact"]   = self.contact0[val_mask]
        if self.feet0 is not None:
            split["train"]["feet_pos"] = self.feet0[train_mask]
            split["val"]["feet_pos"]   = self.feet0[val_mask]
        if self.com0 is not None:
            split["train"]["com"] = self.com0[train_mask]
            split["val"]["com"]   = self.com0[val_mask]
        if self.x_ref0 is not None:
            split["train"]["x_ref0"] = self.x_ref0[train_mask]
            split["val"]["x_ref0"]   = self.x_ref0[val_mask]
        if self.x_ref1 is not None:
            split["train"]["x_ref1"] = self.x_ref1[train_mask]
            split["val"]["x_ref1"]   = self.x_ref1[val_mask]
        return split
