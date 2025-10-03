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

cfg.mpc_params["type"] = "mit_convex"

# Horizon (match or exceed gait planner length; controller will pad/trim)
N  = cfg.mpc_params.get("horizon", 12)
dt = 0.02  # SRB discretization step inside MPC (~100 Hz)

# State weights for [p(3), v(3), theta(3), omega(3)]
Q_position = np.array([0, 0, 1500])  # x, y, z
Q_velocity = np.array([200, 200, 200])  # x_vel, y_vel, z_vel
Q_base_angle = np.array([500, 500, 500])  # roll, pitch, yaw
Q_base_angle_rates = np.array([20, 20, 50])  # roll_rate, pitch_rate, yaw_rate
Q = np.zeros((12,12))

# Q[0:3,0:3]   = Q_position   # z high
# Q[3:6,3:6]   = Q_velocity
# Q[6:9,6:9]   = Q_base_angle   # roll/pitch/yaw
# Q[9:12,9:12] = Q_base_angle_rates

# Q[0:3,0:3]   = np.diag([   5.0,    5.0, 400.0])   # position with z high
# Q[3:6,3:6]   = np.diag([  10.0,   10.0,  10.0])   # velocity  
# Q[6:9,6:9]   = np.diag([ 150.0,  150.0, 300.0])   # roll/pitch/yaw
# Q[9:12,9:12] = np.diag([   5.0,    5.0,  20.0])   # angular rates  

R_leg = np.diag([1, 1, 1])    # cheap forces
R = np.kron(np.eye(4), R_leg)
S = 1e-4 * np.eye(3*4)                  # moderate smoothing

cfg.convex_mpc_params = {
    "horizon": N,
    "dt": dt,
    "Q": Q,
    "R": R,
    "S": S,
    "mu": 1.0,
    "fz_min": 0.0,
    "fz_max": 600.0,
    "log_history": True,
    "debug": True,
}

# Sim timing
cfg.simulation_params["dt"] = 0.002
cfg.simulation_params["mpc_frequency"] = 100    # run MPC every 10 ms

# ---------- Import default run_simulation by path ----------
_spec = importlib.util.spec_from_file_location("default_runner", str(DEFAULT_RUNNER))
_default_runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_default_runner)
run_simulation = _default_runner.run_simulation

# ---------- Plot helpers ----------
from tools.plot_episode import plot_episode_grfs, plot_episode_states

if __name__ == "__main__":
    logdir = REPO_ROOT / "simulation" / "convex_logs"
    logdir.mkdir(parents=True, exist_ok=True)

    out_h5 = run_simulation(
        qpympc_cfg=cfg,
        num_episodes=1,
        num_seconds_per_episode=5,
        ref_base_lin_vel=(0.0, 0.5),
        ref_base_ang_vel=(0.0, 0.0),
        friction_coeff=1.0,
        base_vel_command_type="forward",
        seed=0,
        render=True,
        recording_path=str(logdir),
        recording_filename="convex_experiment.h5",
    )

    print(f"\nSaved trajectory to: {out_h5}")

    plot_episode_states(
        Path(out_h5), episode_idx=0,
        center_xy=False, t0=None, duration=None,
        unwrap_yaw=False, deg=False,
        title_suffix="Convex MPC",
    )

    plot_episode_grfs(
        Path(out_h5), episode_idx=0,
        prefer_cmd=True, fz_threshold=5.0, fz_ylim=None,
        title_suffix="Convex MPC",
    )
