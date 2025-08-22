# h5_reader.py
from __future__ import annotations
import h5py
import numpy as np
from pathlib import Path
from typing import Optional

class _H5TrajIndexable:
    """Wrapper so you can do recordings[key][traj_id] -> (T, dim) ndarray lazily."""
    def __init__(self, ds: h5py.Dataset):
        self._ds = ds

    def __getitem__(self, traj_id: int) -> np.ndarray:
        # h5py slices return numpy arrays
        return self._ds[traj_id, ...]

    def __len__(self) -> int:
        return self._ds.shape[0]

class _RecordingDict:
    """Dict-like access to group 'recordings' (lazy by default)."""
    def __init__(self, f: h5py.File, group: str = "recordings", lazy: bool = True):
        if group not in f:
            raise KeyError(f"Group '{group}' not found in file.")
        self._grp = f[group]
        self._lazy = lazy

    def __getitem__(self, key: str):
        if key not in self._grp:
            raise KeyError(f"Dataset '{key}' not found in /{self._grp.name}.")
        ds = self._grp[key]
        if isinstance(ds, h5py.Dataset):
            return _H5TrajIndexable(ds) if self._lazy else np.array(ds)
        return ds  # nested group if present

    def keys(self):
        return list(self._grp.keys())

class H5Reader:
    """
    Minimal reader expected by QuadrupedEDMDDataset:
      - .recordings[...] is indexable by trajectory: recordings['base_pos'][traj_id] -> (T,3)
      - .n_trajectories gives number of episodes
    """
    def __init__(self, file_path: str, group: str = "recordings", lazy: bool = True):
        self.path = str(Path(file_path))
        self._f: Optional[h5py.File] = h5py.File(self.path, "r")
        self.recordings = _RecordingDict(self._f, group=group, lazy=lazy)

        # Pick a reliable dataset to infer number of trajectories
        if "time" in self.recordings.keys():
            time_ds = self._f[group]["time"]
            self.n_trajectories = int(time_ds.shape[0])
        elif "base_pos" in self.recordings.keys():
            base_pos_ds = self._f[group]["base_pos"]
            self.n_trajectories = int(base_pos_ds.shape[0])
        else:
            # fallback: first dataset in group
            first_key = self.recordings.keys()[0]
            self.n_trajectories = int(self._f[group][first_key].shape[0])

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None

    def __enter__(self):
        if self._f is None:
            self._f = h5py.File(self.path, "r")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
