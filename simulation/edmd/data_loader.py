# loader.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List, Any, Sequence
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



@dataclass
class QuadrupedEDMDDataset:
    file_path: Optional[str] = None
    dataset: Optional[H5Reader] = None
    downsample: int = 1
    pos_zero_start: bool = True
    drop_nan: bool = True
    normalize: Optional[str] = None                 # None | 'minmax'
    feature_range: Tuple[float, float] = (-1.0, 1.0)

    keys: Dict[str, str] = field(default_factory=lambda: {
        "pos": "base_pos",                       # (T,3)
        "eul": "base_ori_euler_xyz",             # (T,3) radians
        "lin_vel": "base_lin_vel",               # (T,3)
        "ang_vel": "base_ang_vel",               # (T,3)
        "u": "contact_forces",                   # (T,12) GRFs [FLx,FLy,FLz, FRx,...]
    })

    # Outputs after build()
    X0: Optional[np.ndarray] = None  # (N, 12)
    X1: Optional[np.ndarray] = None  # (N, 12)
    U0: Optional[np.ndarray] = None  # (N, 12)
    episode_bounds: Optional[List[Tuple[int, int]]] = None  # global index per episode
    traj_ids: Optional[np.ndarray] = None  # (N,) trajectory index per sample

    # learned scalers (if normalize is enabled)
    scaler_X: Optional[MinMaxScaler] = None
    scaler_U: Optional[MinMaxScaler] = None

    def _need_reader(self):
        if self.dataset is None:
            if self.file_path is None:
                raise ValueError("Provide either dataset=H5Reader(...) or file_path to construct one.")
            self.dataset = H5Reader(file_path=str(Path(self.file_path)))

    def _get_traj_arrays(self, traj_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rec = self.dataset.recordings  # type: ignore[attr-defined]
        base_pos = np.asarray(rec[self.keys["pos"]][traj_id])
        base_eul = np.asarray(rec[self.keys["eul"]][traj_id])
        base_lin = np.asarray(rec[self.keys["lin_vel"]][traj_id])
        base_ang = np.asarray(rec[self.keys["ang_vel"]][traj_id])
        u = np.asarray(rec[self.keys["u"]][traj_id])

        # Downsample
        ds = max(1, int(self.downsample))
        if ds > 1:
            base_pos = base_pos[::ds]
            base_eul = base_eul[::ds]
            base_lin = base_lin[::ds]
            base_ang = base_ang[::ds]
            u = u[::ds]

        # Trim to common length
        T = min(len(base_pos), len(base_eul), len(base_lin), len(base_ang), len(u))
        base_pos = base_pos[:T]
        base_eul = base_eul[:T]
        base_lin = base_lin[:T]
        base_ang = base_ang[:T]
        u = u[:T]

        if self.pos_zero_start and T > 0:
            base_pos = base_pos - base_pos[0:1, :]

        return base_pos, base_eul, base_lin, base_ang, u

    def build(self) -> "QuadrupedEDMDDataset":
        """Construct time-shifted pairs X0, X1, U0 across episodes (optionally normalized)."""
        self._need_reader()
        n_traj: int = int(self.dataset.n_trajectories)  # type: ignore[attr-defined]

        X0_list: List[np.ndarray] = []
        X1_list: List[np.ndarray] = []
        U0_list: List[np.ndarray] = []
        bounds: List[Tuple[int, int]] = []
        traj_ids_joined: List[np.ndarray] = []

        cursor = 0
        for k in range(n_traj):
            pos, eul, lin, ang, u = self._get_traj_arrays(k)
            state = np.concatenate([pos, eul, lin, ang], axis=-1)  # (T,12)

            if len(state) < 2:
                continue

            X0_k = state[:-1]
            X1_k = state[1:]
            U0_k = u[:-1]

            if self.drop_nan:
                mask = np.isfinite(X0_k).all(axis=1) & np.isfinite(X1_k).all(axis=1) & np.isfinite(U0_k).all(axis=1)
                X0_k, X1_k, U0_k = X0_k[mask], X1_k[mask], U0_k[mask]

            if X0_k.size == 0:
                continue

            X0_list.append(X0_k)
            X1_list.append(X1_k)
            U0_list.append(U0_k)

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

        # --- optional normalization ---
        if self.normalize is not None:
            if self.normalize.lower() == "minmax":
                self.scaler_X = MinMaxScaler(feature_range=self.feature_range).fit(
                    np.vstack([self.X0, self.X1])
                )
                self.scaler_U = MinMaxScaler(feature_range=self.feature_range).fit(self.U0)

                self.X0 = self.scaler_X.transform(self.X0)
                self.X1 = self.scaler_X.transform(self.X1)
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
