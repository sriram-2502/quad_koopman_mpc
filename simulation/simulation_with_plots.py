# simulation/simulation_baseline.py
# Run the default simulation with the baseline (nominal) MPC, then plot GRFs.

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path
import numpy as np

# ---------- Repo root & default runner ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNNER = REPO_ROOT / "simulation" / "simulation.py"
if not DEFAULT_RUNNER.exists():
    raise FileNotFoundError(f"Default runner not found at: {DEFAULT_RUNNER}")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------- Configure BASELINE (nominal) MPC ----------
from quadruped_pympc import config as cfg

cfg.mpc_params["type"] = "nominal"      # <- use the repo's nominal SRBD MPC
# Timing (feel free to keep defaults if you prefer)
cfg.simulation_params["dt"] = 0.002     # simulator step
cfg.simulation_params["mpc_frequency"] = 100  # MPC solve every 10 ms

# (Optional) tweak baseline weights here if you want:
# cfg.mpc_params["Q"] = np.diag([...])
# cfg.mpc_params["R"] = np.diag([...])

# ---------- Import default run_simulation by path ----------
_spec = importlib.util.spec_from_file_location("default_runner", str(DEFAULT_RUNNER))
_default_runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_default_runner)
run_simulation = _default_runner.run_simulation

# ---------- Import plot helper ----------
from tools.plot_episode import plot_episode_grfs, plot_episode_states

if __name__ == "__main__":
    # Save to simulation/baseline_logs
    logdir = REPO_ROOT / "simulation" / "baseline_logs"
    logdir.mkdir(parents=True, exist_ok=True)

    # Run one episode with simple refs
    out_h5 = run_simulation(
        qpympc_cfg=cfg,
        num_episodes=1,
        num_seconds_per_episode=5,
        ref_base_lin_vel=(0.0, 1.0),   # scaled by hip height inside env
        ref_base_ang_vel=(0.0, 0.2),
        friction_coeff=(0.5, 1.0),
        base_vel_command_type="forward+rotate",
        seed=0,
        render=True,
        recording_path=str(logdir),
        recording_filename="baseline_experiment.h5",
    )

    print(f"\nSaved trajectory to: {out_h5}")

    # Full episode, radians:
    plot_episode_states(
        Path(out_h5),
        episode_idx=0,
        center_xy=False,        # set True to start x,y at 0 (subtract first sample)
        t0=None,                # e.g., 0.0 to start at 0s
        duration=None,          # e.g., 5.0 to plot first 5 seconds
        unwrap_yaw=False,       # True to unwrap yaw for continuity
        deg=False,              # True to plot angles in degrees
        title_suffix="Baseline MPC",
    )

    # Plot episode GRFs (robust time handling & stance shading)
    plot_episode_grfs(
        Path(out_h5),
        episode_idx=0,
        prefer_cmd=True,        # plot controller-commanded GRFs if logged as 'nmpc_GRFs'
        fz_threshold=5.0,       # used only if stance is inferred from forces
        fz_ylim=None,           # auto-pick vertical axis limits for fz
        title_suffix="Baseline MPC",
    )


