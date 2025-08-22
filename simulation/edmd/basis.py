import numpy as np
from scipy.spatial.transform import Rotation as R

def skew_symmetric(omega):
    # omega: (..., 3)
    zero = np.zeros_like(omega[..., 0])
    wx, wy, wz = omega[..., 0], omega[..., 1], omega[..., 2]
    return np.stack([
        np.stack([zero, -wz, wy], axis=-1),
        np.stack([wz, zero, -wx], axis=-1),
        np.stack([-wy, wx, zero], axis=-1)
    ], axis=-2)  # (..., 3, 3)

def observables(x, p_max=2):
    # x: (N, 12) [pos(3), eul(3), lin_vel(3), ang_vel(3)]
    pos = x[:, 0:3]
    eul = x[:, 3:6]
    lin_vel = x[:, 6:9]
    omega = x[:, 9:12]
    N = x.shape[0]
    # Rotation matrices: (N, 3, 3)
    Rmat = R.from_euler('xyz', eul).as_matrix()
    # Omega hat: (N, 3, 3)
    omega_hat = skew_symmetric(omega)
    # psi_bar: [vec(R @ omega_hat^p) for p in 1..p_max]
    psi_bar = []
    omega_hat_p = omega_hat.copy()
    for p in range(1, p_max + 1):
        R_omega_hat_p = np.matmul(Rmat, omega_hat_p)
        psi_bar.append(R_omega_hat_p.reshape(N, -1))
        omega_hat_p = np.matmul(omega_hat_p, omega_hat)  # Next power
    psi_bar = np.concatenate(psi_bar, axis=-1) if psi_bar else np.zeros((N, 0))
    # Flatten Rmat and omega_hat
    Rmat_vec = Rmat.reshape(N, -1)
    omega_hat_vec = omega_hat.reshape(N, -1)
    # Concatenate all observables
    return np.concatenate([pos, lin_vel, Rmat_vec, omega_hat_vec, psi_bar], axis=-1)