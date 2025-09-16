# loader.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List, Any, Sequence, Literal
from pathlib import Path
from h5_reader import H5Reader


# --- tiny min-max scaler ---
@dataclass
class MinMaxScaler:
    feature_range: Tuple[float, float] = (-1.0, 1.0)
    clip: bool = True
    eps: float = 1e-12
    data_min_: Optional[np.ndarray] = None
    data_max_: Optional[np.ndarray] = None
    scale_:    Optional[np.ndarray] = None
    min_:      Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        X = np.asarray(X)
        self.data_min_ = np.nanmin(X, axis=0)
        self.data_max_ = np.nanmax(X, axis=0)
        rng = np.maximum(self.data_max_ - self.data_min_, self.eps)
        a, b = self.feature_range
        self.scale_ = (b - a) / rng
        self.min_   = a - self.data_min_ * self.scale_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = X * self.scale_ + self.min_
        if self.clip:
            a, b = self.feature_range
            Z = np.clip(Z, a, b)
        return Z

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        return (Z - self.min_) / self.scale_


def _repeat4_to_12(c4: np.ndarray) -> np.ndarray:
    """(T,4)->(T,12) expand leg mask across [fx,fy,fz]"""
    c4 = np.asarray(c4, float)
    if c4.ndim == 1:
        c4 = c4.reshape(-1, 4)
    return np.repeat(c4, repeats=3, axis=1)


def _ensure_feet_shape(feet: np.ndarray) -> np.ndarray:
    """Coerce feet to (T,4,3) from (T,12) if needed."""
    feet = np.asarray(feet, float)
    if feet.ndim == 2 and feet.shape[1] == 12:
        return feet.reshape(feet.shape[0], 4, 3)
    assert feet.ndim == 3 and feet.shape[1:] == (4, 3), f"feet_pos must be (T,4,3) or (T,12), got {feet.shape}"
    return feet


def _wrench_from_forces(feet_pos_world_t: np.ndarray, com_world_t: np.ndarray, u_12_t: np.ndarray) -> np.ndarray:
    """
    One-timestep wrench: feet_pos_world_t (4,3), com_world_t (3,), u_12_t (12,)
    returns w = [Fx,Fy,Fz, Mx,My,Mz]
    """
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
    normalize: Optional[str] = None                 # None | 'minmax'
    feature_range: Tuple[float, float] = (-1.0, 1.0)

    # What to use as inputs:
    # - "forces": (default) 12-D GRFs [fx,fy,fz]*4
    # - "wrench": 6-D centroidal wrench [Fx,Fy,Fz,Mx,My,Mz] computed from forces+feet+com
    input_mode: Literal["forces", "wrench"] = "forces"
    # If contact is present, mask swing legs' forces before using them or computing wrench
    mask_inputs_by_contact: bool = True

    # HDF5 keys used to read arrays (override as needed)
    keys: Dict[str, str] = field(default_factory=lambda: {
        "pos": "base_pos",                       # (T,3)
        "eul": "base_ori_euler_xyz",             # (T,3) radians
        "lin_vel": "base_lin_vel",               # (T,3)
        "ang_vel": "base_ang_vel",               # (T,3)
        "u": "contact_forces",                   # (T,12) GRFs [FLx,FLy,FLz, FRx,...]
        # Optional extras:
        "contact": "contact_state",              # (T,4) {0,1} stance flags (if available)
        "feet_pos": "feet_pos",                  # (T,4,3) or (T,12) feet positions in world frame
        "com": "com_pos",                        # (T,3) CoM position in world (fallback to base_pos if missing)
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
    com0: Optional[np.ndarray] = None      # (N,3) if available

    # learned scalers (if normalize is enabled)
    scaler_X: Optional[MinMaxScaler] = None
    scaler_U: Optional[MinMaxScaler] = None

    def _need_reader(self):
        if self.dataset is None:
            if self.file_path is None:
                raise ValueError("Provide either dataset=H5Reader(...) or file_path to construct one.")
            self.dataset = H5Reader(file_path=str(Path(self.file_path)))

    def _get_traj_arrays(
        self, traj_id: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
            Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        rec = self.dataset.recordings  # type: ignore[attr-defined]

        base_pos = np.asarray(rec[self.keys["pos"]][traj_id])
        base_eul = np.asarray(rec[self.keys["eul"]][traj_id])
        base_lin = np.asarray(rec[self.keys["lin_vel"]][traj_id])
        base_ang = np.asarray(rec[self.keys["ang_vel"]][traj_id])
        u        = np.asarray(rec[self.keys["u"]][traj_id])

        # Optional datasets — use try/except, not "in rec"
        contact = None
        k = self.keys.get("contact")
        if isinstance(k, str):
            try:
                contact = np.asarray(rec[k][traj_id])
            except KeyError:
                contact = None

        feet_pos = None
        k = self.keys.get("feet_pos")
        if isinstance(k, str):
            try:
                feet_pos = np.asarray(rec[k][traj_id])
                feet_pos = _ensure_feet_shape(feet_pos)  # (T,4,3) or reshape from (T,12)
            except KeyError:
                feet_pos = None

        com_pos = None
        k = self.keys.get("com")
        if isinstance(k, str):
            try:
                com_pos = np.asarray(rec[k][traj_id])
            except KeyError:
                com_pos = None

        # Downsample
        ds = max(1, int(self.downsample))
        if ds > 1:
            base_pos = base_pos[::ds]
            base_eul = base_eul[::ds]
            base_lin = base_lin[::ds]
            base_ang = base_ang[::ds]
            u        = u[::ds]
            if contact is not None:
                contact = contact[::ds]
            if feet_pos is not None:
                feet_pos = feet_pos[::ds]
            if com_pos is not None:
                com_pos = com_pos[::ds]

        # Trim to common length
        lens = [len(base_pos), len(base_eul), len(base_lin), len(base_ang), len(u)]
        if contact is not None: lens.append(len(contact))
        if feet_pos is not None: lens.append(len(feet_pos))
        if com_pos is not None: lens.append(len(com_pos))
        T = min(lens)

        base_pos = base_pos[:T]
        base_eul = base_eul[:T]
        base_lin = base_lin[:T]
        base_ang = base_ang[:T]
        u        = u[:T]
        if contact is not None:
            contact = contact[:T]
        if feet_pos is not None:
            feet_pos = feet_pos[:T]
        if com_pos is not None:
            com_pos = com_pos[:T]

        # Position zeroing (and COM if provided)
        if self.pos_zero_start and T > 0:
            base_pos = base_pos - base_pos[0:1, :]
            if com_pos is not None:
                com_pos = com_pos - com_pos[0:1, :]

        return base_pos, base_eul, base_lin, base_ang, u, contact, feet_pos, com_pos


    def build(self) -> "QuadrupedEDMDDataset":
        """Construct time-shifted pairs X0, X1, U0 across episodes (optionally normalized).
           Also returns extras aligned to X0/U0: contact0, feet0, com0 (if present)."""
        self._need_reader()
        n_traj: int = int(self.dataset.n_trajectories)  # type: ignore[attr-defined]

        X0_list: List[np.ndarray] = []
        X1_list: List[np.ndarray] = []
        U0_list: List[np.ndarray] = []
        bounds: List[Tuple[int, int]] = []
        traj_ids_joined: List[np.ndarray] = []

        contact0_list: List[np.ndarray] = []
        feet0_list: List[np.ndarray] = []
        com0_list: List[np.ndarray] = []

        cursor = 0
        for k in range(n_traj):
            pos, eul, lin, ang, u, contact, feet_pos, com_pos = self._get_traj_arrays(k)
            state = np.concatenate([pos, eul, lin, ang], axis=-1)  # (T,12)

            if len(state) < 2:
                continue

            # Align time-shifted pairs
            X0_k = state[:-1]
            X1_k = state[1:]
            U0_k = u[:-1]

            # Optional masking of inputs using contact (swing=0)
            if self.mask_inputs_by_contact and (contact is not None):
                C12k = _repeat4_to_12(contact[:-1])
                U0_k = U0_k * C12k

            # If we want wrench, need feet + com (fallback com to base_pos if absent)
            if self.input_mode == "wrench":
                if feet_pos is None:
                    raise ValueError("input_mode='wrench' requires feet_pos in dataset (key 'feet_pos').")
                if com_pos is None:
                    # fallback to base pos as COM approximation
                    com_pos = pos
                W = np.zeros((U0_k.shape[0], 6), float)
                for t in range(U0_k.shape[0]):
                    W[t] = _wrench_from_forces(feet_pos[t], com_pos[t], U0_k[t])
                U0_k = W  # replace forces with wrench

            # Drop NaNs
            if self.drop_nan:
                mask = np.isfinite(X0_k).all(axis=1) & np.isfinite(X1_k).all(axis=1) & np.isfinite(U0_k).all(axis=1)
                X0_k, X1_k, U0_k = X0_k[mask], X1_k[mask], U0_k[mask]
                if contact is not None:
                    contact_k = contact[:-1][mask]
                else:
                    contact_k = None
                if feet_pos is not None:
                    feet_k = feet_pos[:-1][mask]
                else:
                    feet_k = None
                if com_pos is not None:
                    com_k = com_pos[:-1][mask]
                else:
                    com_k = None
            else:
                contact_k = contact[:-1] if contact is not None else None
                feet_k = feet_pos[:-1] if feet_pos is not None else None
                com_k = com_pos[:-1] if com_pos is not None else None

            if X0_k.size == 0:
                continue

            X0_list.append(X0_k)
            X1_list.append(X1_k)
            U0_list.append(U0_k)

            if contact_k is not None:
                contact0_list.append(contact_k.astype(float))
            if feet_k is not None:
                feet0_list.append(feet_k.astype(float))
            if com_k is not None:
                com0_list.append(com_k.astype(float))

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

        # Extras aligned to X0/U0 (may be missing)
        self.contact0 = np.concatenate(contact0_list, axis=0) if contact0_list else None
        self.feet0    = np.concatenate(feet0_list, axis=0)    if feet0_list    else None
        self.com0     = np.concatenate(com0_list, axis=0)     if com0_list     else None
        # If com missing, use base position subset (already in X0 as first 3 dims)
        if self.com0 is None:
            self.com0 = self.X0[:, 0:3].copy()

        # --- optional normalization (on all rows) ---
        if self.normalize is not None:
            if self.normalize.lower() == "minmax":
                self.scaler_X = MinMaxScaler(feature_range=self.feature_range).fit(
                    np.vstack([self.X0, self.X1])
                )
                self.X0 = self.scaler_X.transform(self.X0)
                self.X1 = self.scaler_X.transform(self.X1)

                self.scaler_U = MinMaxScaler(feature_range=self.feature_range).fit(self.U0)
                self.U0 = self.scaler_U.transform(self.U0)
            else:
                raise ValueError(f"Unknown normalize option: {self.normalize}")

        return self

    def train_val_split_by_traj(self, val_frac: float = 0.2, seed: int = 0
                                ) -> Dict[str, Dict[str, np.ndarray]]:
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
        return split

    def refit_normalizers_on_train(self, train_mask: np.ndarray):
        """
        Re-fit scalers using only training samples, then re-transform ALL arrays.
        Use this AFTER you build() and split.
        """
        if self.normalize is None:
            return
        if self.normalize.lower() != "minmax":
            raise ValueError("Only supported for 'minmax' in this helper.")

        # If already normalized, go back to raw first
        if self.scaler_X is not None:
            self.X0 = self.scaler_X.inverse_transform(self.X0)
            self.X1 = self.scaler_X.inverse_transform(self.X1)
        if self.scaler_U is not None:
            self.U0 = self.scaler_U.inverse_transform(self.U0)

        # Fit only on train rows
        self.scaler_X = MinMaxScaler(feature_range=self.feature_range).fit(
            np.vstack([self.X0[train_mask], self.X1[train_mask]])
        )
        self.scaler_U = MinMaxScaler(feature_range=self.feature_range).fit(
            self.U0[train_mask]
        )

        # Re-apply to all
        self.X0 = self.scaler_X.transform(self.X0)
        self.X1 = self.scaler_X.transform(self.X1)
        self.U0 = self.scaler_U.transform(self.U0)
