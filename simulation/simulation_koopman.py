# simulation/simulation_koopman.py
# Run the default simulation with Koopman MPC, then plot GRFs using the helper.

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

# ---------- Configure Koopman MPC ----------
from quadruped_pympc import config as cfg

cfg.mpc_params["type"] = "koopman"

cfg.koopman_mpc_params["log_history"] = True
cfg.koopman_mpc_params["debug"] = True
cfg.koopman_mpc_params["debug_every"] = 200
cfg.koopman_mpc_params["log_history"] = True
cfg.koopman_mpc_params["dt"] = 0.02             # EDMD model step (match training dt)
cfg.koopman_mpc_params["p_max"] = 6           # num basis used in edmd model

cfg.simulation_params["dt"] = 0.002
cfg.simulation_params["mpc_frequency"] = 100    # MPC every 10ms

# Given per-axis weights
Q_position          = np.array([0,   0,   1500])   # x, y, z
Q_velocity          = np.array([200, 200, 200])    # xdot, ydot, zdot
Q_base_angle        = np.array([500, 500, 0])      # roll, pitch, yaw
Q_base_angle_rates  = np.array([20,  20,  50])     # wx, wy, wz

# Q_position          = np.array([100,   100,   1500])   # x, y, z
# Q_velocity          = np.array([200, 200, 200])    # xdot, ydot, zdot
# Q_base_angle        = np.array([500, 500, 500])      # roll, pitch, yaw
# Q_base_angle_rates  = np.array([20,  20,  50])     # wx, wy, wz

# State layout (nx = 30):
# [0:3]   base position p
# [3:6]   base linear velocity pdot
# [6:9]   base angles (roll, pitch, yaw)
# [9:12]  base angular velocity (wx, wy, wz)
# [12:24] feet positions (FL, FR, RL, RR)  <-- leave zero if you don’t want to penalize
# [24:30] integral states (z, xdot, ydot, zdot, roll, pitch) <-- leave zero

nx = 12
Q = np.zeros((nx, nx), dtype=float)

# Fill the four blocks you care about
Q[0:3,   0:3]   = np.diag(Q_position)         # position
Q[3:6,   3:6]   = np.diag(Q_base_angle)         # linear velocity
Q[6:9,   6:9]   = np.diag(Q_velocity)       # angles
Q[9:12,  9:12]  = np.diag(Q_base_angle_rates) # angular velocity

R_leg = np.diag([0.01, 0.01, 0.005]).astype(float)
R = np.kron(np.eye(4), R_leg)
cfg.koopman_mpc_params["Q"] = Q
cfg.koopman_mpc_params["R"] = R

# ---------- Import default run_simulation by path ----------
_spec = importlib.util.spec_from_file_location("default_runner", str(DEFAULT_RUNNER))
_default_runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_default_runner)
run_simulation = _default_runner.run_simulation

# ---------- Import plot helper (module path OR by file path fallback) ----------
from tools.plot_episode import plot_episode_grfs, plot_episode_states

if __name__ == "__main__":
    # Save to simulation/kmpc_logs
    logdir = REPO_ROOT / "simulation" / "kmpc_logs"
    logdir.mkdir(parents=True, exist_ok=True)

    # Run one episode
    out_h5 = run_simulation(
        qpympc_cfg=cfg,
        num_episodes=1,
        num_seconds_per_episode=5,
        ref_base_lin_vel=(0.0, 0.1),   # height hold only; controller builds x_ref accordingly
        ref_base_ang_vel=(0.0, 0.1),
        friction_coeff=(0.5, 1.0),
        base_vel_command_type="forward+rotate",
        seed=0,
        render=True,
        recording_path=str(logdir),
        recording_filename="koopman_experiment.h5",
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
        title_suffix="Koopman MPC",
    )

    # Plot episode GRFs (robust time handling & stance shading)
    plot_episode_grfs(
        Path(out_h5),
        episode_idx=0,
        prefer_cmd=True,        # plot controller-commanded GRFs if logged as 'nmpc_GRFs'
        fz_threshold=5.0,       # used only if stance is inferred from forces
        fz_ylim=None,           # auto-pick vertical axis limits for fz
        title_suffix="Koopman MPC",
    )