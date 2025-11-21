# simulation/simulation_convex.py
# Run default simulation with Convex (forces-only) MPC, then plot GRFs & states.

from __future__ import annotations
import sys, importlib.util
from pathlib import Path
import numpy as np

# ---------- Repo root & default runner ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNNER = REPO_ROOT / "simulation" / "simulation.py"
if not DEFAULT_RUNNER.exists():
    raise FileNotFoundError(f"Default runner not found at: {DEFAULT_RUNNER}")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------- Configure Convex MPC ----------
from quadruped_pympc import config as cfg

cfg.mpc_params["type"] = "koopman"
cfg.mpc_params["horizon"] = 10
# Sim timing
cfg.simulation_params["dt"] = 0.002
cfg.simulation_params["mpc_frequency"] = 200    # run MPC every 10 ms

# ---------- Import default run_simulation by path ----------
_spec = importlib.util.spec_from_file_location("default_runner", str(DEFAULT_RUNNER))
_default_runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_default_runner)
run_simulation = _default_runner.run_simulation

# ---------- Plot helpers ----------
from tools.plot_episode import plot_episode_grfs, plot_episode_states

if __name__ == "__main__":
    logdir = REPO_ROOT / "simulation" / "koopman_logs"
    logdir.mkdir(parents=True, exist_ok=True)

    out_h5 = run_simulation(
        qpympc_cfg=cfg,
        num_episodes=1,
        num_seconds_per_episode=0.5,
        ref_base_lin_vel=0.5,
        ref_base_ang_vel=(-0.1, 0.1),
        friction_coeff=1.0,
        base_vel_command_type="forward+rotate",
        seed=0,
        render=True,
        recording_path=str(logdir),
        recording_filename="koopman_experiment.h5",
    )

    print(f"\nSaved trajectory to: {out_h5}")

    plot_episode_states(
        Path(out_h5), episode_idx=0,
        center_xy=False, t0=None, duration=None,
        unwrap_yaw=False, deg=False,
        title_suffix="Koopman MPC",
    )

    plot_episode_grfs(
        Path(out_h5), episode_idx=0,
        prefer_cmd=True, fz_threshold=5.0, fz_ylim=None,
        title_suffix="Koopman MPC",
    )
