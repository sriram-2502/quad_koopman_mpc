# rollout.py
import numpy as np
from scipy.spatial.transform import Rotation as R
from basis import observables  # uses your p_max

# --- indices inside your observables vector ---
# observables = [pos(3), lin_vel(3), R(9), omega_hat(9), psi_bar(p_max*9)]
def _slices(p_max: int):
    i = 0
    sl_pos = slice(i, i+3); i += 3
    sl_lin = slice(i, i+3); i += 3
    sl_R   = slice(i, i+9); i += 9
    sl_oh  = slice(i, i+9); i += 9
    # remainder is psi_bar, not needed for state reconstruction
    return sl_pos, sl_lin, sl_R, sl_oh

def _vee(S):
    """Inverse of hat: so(3)->R^3."""
    return np.array([S[2,1], S[0,2], S[1,0]], dtype=float)

def _project_to_SO3(M):
    """Nearest rotation via SVD (Procrustes). Ensures det=+1."""
    U, _, Vt = np.linalg.svd(M)
    Rproj = U @ Vt
    if np.linalg.det(Rproj) < 0:
        U[:, -1] *= -1
        Rproj = U @ Vt
    return Rproj

def rollout_open_loop(A, B, x0, U_seq, p_max=2, return_euler=False, degrees=False, enforce_SO3=True):
    """
    Open-loop simulate the learned lifted dynamics:
        z_{t+1} = A z_t + B u_t
    using your dictionary from basis.observables.

    Args:
      A: (N,N)
      B: (N,nu)
      x0: (12,) base state [pos(3), eul(3), lin_vel(3), ang_vel(3)]
      U_seq: (T, nu) sequence of inputs
      p_max: same p_max you used in basis.observables
      return_euler: if True, also return euler angles reconstructed from R
      degrees: euler units if return_euler=True
      enforce_SO3: project predicted 3x3 block back to SO(3)

    Returns:
      dict with keys:
        'pos': (T,3), 'lin_vel': (T,3),
        'R': (T,3,3), 'omega': (T,3),
        (optional) 'eul': (T,3) if return_euler
        'z': (T,N) lifted predictions
    """
    # initial lift
    z = observables(x0[None, :], p_max=p_max)[0]  # (N,)
    T = U_seq.shape[0]
    N = z.shape[0]
    sl_pos, sl_lin, sl_R, sl_oh = _slices(p_max)

    pos_traj   = np.zeros((T, 3))
    lin_traj   = np.zeros((T, 3))
    R_traj     = np.zeros((T, 3, 3))
    omega_traj = np.zeros((T, 3))
    z_traj     = np.zeros((T, N))

    for t in range(T):
        # propagate in lifted space
        z = A @ z + B @ U_seq[t]
        z_traj[t] = z

        # decode pieces
        pos   = z[sl_pos]
        lin_v = z[sl_lin]

        R_flat = z[sl_R]
        R_pred = R_flat.reshape(3, 3)
        if enforce_SO3:
            R_pred = _project_to_SO3(R_pred)

        oh_flat = z[sl_oh]
        S = oh_flat.reshape(3, 3)
        # make sure it's skew (numerical symmetry)
        S = 0.5 * (S - S.T)
        omega = _vee(S)

        pos_traj[t]   = pos
        lin_traj[t]   = lin_v
        R_traj[t]     = R_pred
        omega_traj[t] = omega

    out = {
        "pos": pos_traj,
        "lin_vel": lin_traj,
        "R": R_traj,
        "omega": omega_traj,
        "z": z_traj,
    }
    if return_euler:
        eul = np.zeros((T, 3))
        for t in range(T):
            eul[t] = R.from_matrix(R_traj[t]).as_euler('xyz', degrees=degrees)
        out["eul"] = eul
    return out
