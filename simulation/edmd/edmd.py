import numpy as np

# ---------- EDMD with inputs ----------
def edmd_with_inputs(X, Y, U, lift_fn, l2_reg=1e-6):
    """
    Learn Koopman matrices A, B s.t. Π(Y) ≈ A Π(X) + B U.
    Shapes:
      X: (M, nx)   states at t
      Y: (M, nx)   states at t+1
      U: (M, nu)   inputs at t
    """
    PhiX = np.vstack([lift_fn(xi) for xi in X])        # (M, N)
    PhiY = np.vstack([lift_fn(yi) for yi in Y])        # (M, N)
    M = PhiX.shape[0]
    ones_over_M = 1.0 / max(1, M)

    # augmented regressor Π̂ = [Π(x); u]
    Phihat = np.hstack([PhiX, U])                      # (M, N+nu)

    # G1, G2 (empirical covariances), then K = G1 G2^{-1}
    G1 = ones_over_M * (PhiY.T @ Phihat)
    G2 = ones_over_M * (Phihat.T @ Phihat)

    # ridge regularization for numerical stability
    G2 += l2_reg * np.eye(G2.shape[0])

    K = G1 @ np.linalg.solve(G2, np.eye(G2.shape[0]))  # (N x (N+nu))

    N = PhiX.shape[1]
    A = K[:, :N]
    B = K[:, N:]
    return A, B