# quadruped_pympc/controllers/convex/mit_convex_mpc.py
import numpy as np
import osqp
from scipy import sparse
import copy
import quadruped_pympc.config as config

_LEG_ORDER = ("FL", "FR", "RL", "RR")

def _skew(v):
    x, y, z = v
    return np.array([[0, -z,  y],
                     [z,  0, -x],
                     [-y, x,  0]], dtype=float)

class MITConvexCentroidalMPC:
    """
    Minimal convex MPC per:
      Di Carlo et al., "Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive Control".

    Decision variables: stacked ground reaction forces F = [f_FL, f_FR, f_RL, f_RR]_k for k = 0..N-1 (each f in R^3).

    Discrete SRBD centroidal dynamics (Euler), convexified with frozen-yaw inertia:
        p_{k+1}     = p_k + dt * v_k
        v_{k+1}     = v_k + dt * ( g + (1/m) * sum_i f_{i,k} )
        theta_{k+1} = theta_k + dt * omega_k
        omega_{k+1} = omega_k + dt * I_W(yaw0)^{-1} * sum_i ( (r_i[k] - p0) x f_{i,k} )

    Assumptions:
      - Foot positions r_i[k] are in WORLD frame; torque arm uses (r_i[k] - p0) with p0 = current CoM (frozen).
      - Inertia I is provided in BODY frame; map to WORLD using yaw0 only (Rz yaw).
      - Contact schedule contact_sequence has shape (4, N), 1=stance, 0=swing, in _LEG_ORDER.
      - Friction constraints: pyramid (|fx|<=mu fz, |fy|<=mu fz, fz in [fz_min,fz_max]).
    """

    def __init__(
        self,
        mass: float,
        inertia: np.ndarray,  # 3x3 body-frame inertia
        N: int,
        dt: float,
        g: float = 9.81,
        mu: float = 1.0,
        fz_min: float = 0.0,
        fz_max: float = 1e3,
        Qp: np.ndarray | float = 1.0,
        Qv: np.ndarray | float = 10.0,
        Qt: np.ndarray | float = 1.0,
        Qw: np.ndarray | float = 10.0,
        Rf: float = 1e-3,
    ):
        self.m = float(mass)
        self.Ib = np.array(inertia, dtype=float)
        self.N = int(N)
        # IMPORTANT: use the dt passed in — do not override from a global config
        self.dt = float(dt)
        self.gvec = np.array([0.0, 0.0, -float(g)], dtype=float)
        self.mu = float(mu)
        self.fz_min = float(fz_min)
        self.fz_max = float(fz_max)

        # weights (3x3 each if arrays; else scalar*I)
        self.Qp = np.eye(3)*Qp if np.isscalar(Qp) else np.array(Qp, dtype=float)
        self.Qv = np.eye(3)*Qv if np.isscalar(Qv) else np.array(Qv, dtype=float)
        self.Qt = np.eye(3)*Qt if np.isscalar(Qt) else np.array(Qt, dtype=float)
        self.Qw = np.eye(3)*Qw if np.isscalar(Qw) else np.array(Qw, dtype=float)
        self.Rf = float(Rf)

        # OSQP workspace (reused)
        self._osqp = osqp.OSQP()
        self._is_warm = False

        # (kept for parity if you re-enable scaling later)
        self.initial_base_position = np.zeros(3)


    def set_weight(self, nx: int = 12, nu: int = 12):
        """
        Minimal weights for convex centroidal MPC.
        State order: [p(3), v(3), theta(3), omega(3)]  -> Q is (12x12)
        Control:     [FL(3), FR(3), RL(3), RR(3)]      -> R is (12x12)

        Returns:
            Q (12x12), R (12x12)
        """
        import numpy as np

        # --- State weights (your numbers) ---
        # Q_position          = np.array([   0.0,    0.0, 1500.0])  # x,y,z
        # Q_velocity          = np.array([ 200.0,  200.0,  200.0])  # vx,vy,vz
        # Q_base_angle        = np.array([ 500.0,  500.0,    0.0])  # roll,pitch,yaw
        # Q_base_angle_rates  = np.array([  20.0,   20.0,   50.0])  # wx,wy,wz

        Q_position          = np.array([   0.0,    0.0, 1000.0])  # x,y,z
        Q_velocity          = np.array([ 1000.0,  1000.0,  1000.0])  # vx,vy,vz
        Q_base_angle        = np.array([ 500.0,  500.0,    0.0])  # roll,pitch,yaw
        Q_base_angle_rates  = np.array([  20.0,   20.0,   1000.0])  # wx,wy,wz

        Q = np.diag(
            np.concatenate([Q_position, Q_velocity, Q_base_angle, Q_base_angle_rates]).astype(float)
        )

        # --- Control (force) weights per foot axis ---
        R_per_leg = np.diag([1e-3, 1e-3, 1e-3])       # fx, fy, fz
        R = np.kron(np.eye(4), R_per_leg)             # 4 feet -> (12x12)

        # (Optional) sanity checks
        assert Q.shape == (nx, nx), f"Q must be {nx}x{nx}, got {Q.shape}"
        assert R.shape == (nu, nu), f"R must be {nu}x{nu}, got {R.shape}"

        return Q, R

    # ---- internal helpers ----
    def _build_selection(self, contacts_k):
        """Return a diagonal block selector S_k (12x12) that zeros swing leg forces at stage k."""
        S = np.zeros((12, 12))
        for i, _ in enumerate(_LEG_ORDER):
            on = 1.0 if contacts_k[i] else 0.0
            S[3*i:3*i+3, 3*i:3*i+3] = on*np.eye(3)
        return S

    def _R_from_euler(self, euler_xyz):
        """
        Return Rz(yaw) only (MIT convex simplification).
        The inputs are (roll, pitch, yaw); we only use yaw.
        """
        rx, ry, rz = euler_xyz
        cz, sz = np.cos(rz), np.sin(rz)
        Rz = np.array([[cz,-sz,0],
                       [sz, cz,0],
                       [ 0,  0,1]])
        return Rz

    def _friction_pyramid_blocks(self, S: np.ndarray):
        """
        Build inequality rows for a single stage (12 force vars):
            -mu*fz <= fx <=  mu*fz
            -mu*fz <= fy <=  mu*fz
            fz_min <= fz <= fz_max
        The selector S (12x12) zeroes swing-leg forces.
        Returns:
            A_ineq: (5*4, 12)   l: (5*4,)   u: (5*4,)
        """
        mu = self.mu
        A_rows, l, u = [], [], []
        for i in range(4):
            row_fx_le_mu_fz = np.zeros((1, 12)); row_fx_le_mu_fz[:, 3*i:3*i+3] = np.array([[ 1.0,  0.0, -mu]])
            row_fx_ge_mn_mu = np.zeros((1, 12)); row_fx_ge_mn_mu[:, 3*i:3*i+3] = np.array([[-1.0,  0.0, -mu]])
            row_fy_le_mu_fz = np.zeros((1, 12)); row_fy_le_mu_fz[:, 3*i:3*i+3] = np.array([[ 0.0,  1.0, -mu]])
            row_fy_ge_mn_mu = np.zeros((1, 12)); row_fy_ge_mn_mu[:, 3*i:3*i+3] = np.array([[ 0.0, -1.0, -mu]])
            row_fz_bounds   = np.zeros((1, 12)); row_fz_bounds[:,   3*i:3*i+3] = np.array([[ 0.0,  0.0,  1.0]])
            # zero-out swing legs
            row_fx_le_mu_fz = row_fx_le_mu_fz @ S
            row_fx_ge_mn_mu = row_fx_ge_mn_mu @ S
            row_fy_le_mu_fz = row_fy_le_mu_fz @ S
            row_fy_ge_mn_mu = row_fy_ge_mn_mu @ S
            row_fz_bounds   = row_fz_bounds   @ S
            A_rows.extend([row_fx_le_mu_fz, row_fx_ge_mn_mu, row_fy_le_mu_fz, row_fy_ge_mn_mu, row_fz_bounds])
            l.extend([-np.inf, -np.inf, -np.inf, -np.inf, self.fz_min])
            u.extend([0.0,      0.0,     0.0,     0.0,     self.fz_max])
        A_ineq = np.vstack(A_rows)
        return A_ineq, np.asarray(l, dtype=float), np.asarray(u, dtype=float)

    def _feet_refs_over_horizon(self, ref_state, contacts):
        """
        Build per-stage foot references r_i[k] (i in {FL,FR,RL,RR}), size (N,4,3).
        Uses the stepping rule: advance leg's waypoint when it transitions stance->swing.
        """
        N = self.N
        rFL_all = np.atleast_2d(np.array(ref_state["ref_foot_FL"], dtype=float))
        rFR_all = np.atleast_2d(np.array(ref_state["ref_foot_FR"], dtype=float))
        rRL_all = np.atleast_2d(np.array(ref_state["ref_foot_RL"], dtype=float))
        rRR_all = np.atleast_2d(np.array(ref_state["ref_foot_RR"], dtype=float))

        FLc = contacts[0, :].astype(int)
        FRc = contacts[1, :].astype(int)
        RLc = contacts[2, :].astype(int)
        RRc = contacts[3, :].astype(int)

        idx = np.array([0, 0, 0, 0], dtype=int)
        r_over_h = np.zeros((N, 4, 3), dtype=float)

        for j in range(N):
            r_over_h[j, 0, :] = rFL_all[min(idx[0], rFL_all.shape[0]-1)]
            r_over_h[j, 1, :] = rFR_all[min(idx[1], rFR_all.shape[0]-1)]
            r_over_h[j, 2, :] = rRL_all[min(idx[2], rRL_all.shape[0]-1)]
            r_over_h[j, 3, :] = rRR_all[min(idx[3], rRR_all.shape[0]-1)]

            if j < N-1:  # advance after assignment if stance->swing at next step
                if FLc[j] == 1 and FLc[j+1] == 0 and idx[0] + 1 < rFL_all.shape[0]: idx[0] += 1
                if FRc[j] == 1 and FRc[j+1] == 0 and idx[1] + 1 < rFR_all.shape[0]: idx[1] += 1
                if RLc[j] == 1 and RLc[j+1] == 0 and idx[2] + 1 < rRL_all.shape[0]: idx[2] += 1
                if RRc[j] == 1 and RRc[j+1] == 0 and idx[3] + 1 < rRR_all.shape[0]: idx[3] += 1

        return r_over_h  # (N,4,3)

    def _stack_dynamics_maps(self, p0, v0, w0, r_over_h, contacts, Iinv_world):
        """
        Build affine maps from stacked forces to linear and angular rates over the horizon:

           v_{k+1}     = v0 + dt*(k+1)*g + (dt/m) * sum_{j=0..k} sum_i f_{i,j}
           omega_{k+1} = w0 + dt * sum_{j=0..k} I^{-1} * sum_i ( (r_i[j] - p0) x f_{i,j} )

        Returns:
            Bv, Bw: (3N x 12N) maps
            gv, gw: (3N,) offsets
            S_block: (12N x 12N) contact selection block-diag
        """
        N = self.N
        dt = self.dt
        m  = self.m
        Iinv = Iinv_world

        # Build stacked contact selector S (12N x 12N)
        S_block = sparse.block_diag([self._build_selection(contacts[:, k]) for k in range(N)]).toarray()

        # Pre-allocate block rows
        Bv_blocks, Bw_blocks = [], []
        gv_list, gw_list = [], []

        # Linear "sum over feet" map for a single stage: [I I I I]
        Sv_allfeet = np.zeros((3, 12))
        eye3 = np.eye(3)
        for i in range(4):
            Sv_allfeet[:, 3*i:3*i+3] = eye3

        for k in range(N):
            row_blocks_v, row_blocks_w = [], []
            for j in range(N):
                if j <= k:
                    Sv = Sv_allfeet
                    Sw = np.zeros((3, 12))
                    for i in range(4):
                        arm = r_over_h[j, i, :] - p0
                        Sw[:, 3*i:3*i+3] = _skew(arm)
                else:
                    Sv = np.zeros((3, 12))
                    Sw = np.zeros((3, 12))
                row_blocks_v.append((dt/m) * Sv)
                row_blocks_w.append(dt * (Iinv @ Sw))

            Bv_blocks.append(np.hstack(row_blocks_v))  # 3 x (12N)
            Bw_blocks.append(np.hstack(row_blocks_w))  # 3 x (12N)

            gv_list.append(v0 + (k+1)*dt*self.gvec)  # gravity accumulation
            gw_list.append(w0)

        Bv = np.vstack(Bv_blocks)  # (3N x 12N)
        Bw = np.vstack(Bw_blocks)  # (3N x 12N)

        # Apply contact selection (zeros swing forces)
        Bv = Bv @ S_block
        Bw = Bw @ S_block

        gv = np.hstack(gv_list)    # (3N,)
        gw = np.hstack(gw_list)    # (3N,)

        return Bv, Bw, gv, gw, S_block

    # ---- main API ----
    def compute_control(self, state_current, ref_state, contact_sequence, inertia=None):
        """
        Keys follow Acados_NMPC_Nominal:
        state_current:
            - position            : (3,)
            - linear_velocity     : (3,)
            - angular_velocity    : (3,)
            - orientation         : (3,)  # roll, pitch, yaw
        ref_state:
            - ref_position        : (N x 3) or (1 x 3)
            - ref_linear_velocity : (N x 3) or (1 x 3)
            - ref_orientation     : (N x 3) or (1 x 3)
            - ref_angular_velocity: (N x 3) or (1 x 3)
            - ref_foot_FL, ref_foot_FR, ref_foot_RL, ref_foot_RR : (T_i x 3), WORLD frame
        contact_sequence: (4 x N) array of {0,1}
        Returns:
            f0: (12,) forces at k=0 (fx,fy,fz for FL,FR,RL,RR)
            footholds: list of 4 (3,) foot world positions at stage 0
            pred: (N+1, 12) [p, v, theta, omega]
        """
        N = self.N
        dt = self.dt

        # --- unpack current state ---
        p0 = np.array(state_current["position"], dtype=float).reshape(3)
        v0 = np.array(state_current["linear_velocity"], dtype=float).reshape(3)
        w0 = np.array(state_current["angular_velocity"], dtype=float).reshape(3)
        theta0 = np.array(state_current.get("orientation", np.zeros(3)), dtype=float).reshape(3)

        # Helper to tile/clip to horizon
        def _horizonize(x):
            arr = np.array(x, dtype=float).reshape(-1, 3)
            if arr.shape[0] == 1:
                arr = np.tile(arr, (N, 1))
            elif arr.shape[0] < N:
                tail = np.repeat(arr[-1:, :], N - arr.shape[0], axis=0)
                arr = np.vstack([arr, tail])
            return arr[:N, :]

        pref  = _horizonize(ref_state.get("ref_position", np.array(p0)[None, :]))
        vref  = _horizonize(ref_state["ref_linear_velocity"])
        oref  = _horizonize(ref_state.get("ref_orientation", np.array(theta0)[None, :]))
        wref  = _horizonize(ref_state["ref_angular_velocity"])

        contacts = np.array(contact_sequence, dtype=float)  # (4, N)

        # Map inertia to WORLD using yaw0 only (MIT convex simplification)
        Rz = self._R_from_euler(theta0)      # uses yaw component only
        I_world0 = Rz @ self.Ib @ Rz.T
        I_world0_inv = np.linalg.inv(I_world0)

        # Per-stage foot references (N,4,3) in WORLD
        r_over_h = self._feet_refs_over_horizon(ref_state, contacts)

        # --- build v, omega maps (core dynamics) ---
        Bv, Bw, gv, gw, S_block = self._stack_dynamics_maps(p0, v0, w0, r_over_h, contacts, I_world0_inv)
        Bv_s = sparse.csc_matrix(Bv)
        Bw_s = sparse.csc_matrix(Bw)

        # --- integrate to get position (p) and orientation (theta) maps ---
        # Strictly-lower block accumulator C (row k sums cols 0..k-1).
        I3 = np.eye(3)
        C_blocks = []
        for k in range(N):
            row_blocks = []
            for j in range(N):
                row_blocks.append(I3 if j < k else np.zeros((3,3)))
            C_blocks.append(np.hstack(row_blocks))
        C_s = sparse.csc_matrix(np.vstack(C_blocks))  # (3N x 3N)
        oneN = np.ones((N,1))

        # Stacked references
        Pref = pref.reshape(3*N)
        Vref = vref.reshape(3*N)
        Oref = oref.reshape(3*N)
        Wref = wref.reshape(3*N)

        # p map: Pstack = (dt*C)* (Bv F + gv) + kron(1_N, p0)
        Bp_s = (self.dt * C_s) @ Bv_s
        gp = self.dt * (C_s @ gv)               # integrate gv
        gp = np.asarray(gp).reshape(3*N) + np.kron(oneN, p0).reshape(3*N)

        # theta map: Tstack = (dt*C)* (Bw F + gw) + kron(1_N, theta0)
        Btheta_s = (self.dt * C_s) @ Bw_s
        gtheta = self.dt * (C_s @ gw)           # integrate gw
        gtheta = np.asarray(gtheta).reshape(3*N) + np.kron(oneN, theta0).reshape(3*N)

        # === Build force reference from contact schedule (fx=fy=0; fz = mg / #stance) ===
        g_mag = float(abs(self.gvec[2]))
        m = self.m
        Fref_list = []
        for k in range(N):
            c = contacts[:, k].astype(float)  # [FL, FR, RL, RR] in {0,1}
            legs_in_stance = int(c.sum())
            fz_each = (m * g_mag / legs_in_stance) if legs_in_stance > 0 else 0.0
            f_FL = np.array([0.0, 0.0, fz_each * c[0]])
            f_FR = np.array([0.0, 0.0, fz_each * c[1]])
            f_RL = np.array([0.0, 0.0, fz_each * c[2]])
            f_RR = np.array([0.0, 0.0, fz_each * c[3]])
            Fref_list.append(np.hstack([f_FL, f_FR, f_RL, f_RR]))
        Fref = np.hstack(Fref_list)  # (12*N,)

        # === Stacked state map: x_stack = Sx * F + gx ===
        # Order matches state: [p(3N); v(3N); theta(3N); omega(3N)]
        Sx_s = sparse.vstack([Bp_s, Bv_s, Btheta_s, Bw_s], format="csc")  # (12N x 12N)
        gx   = np.hstack([gp,    gv,    gtheta,    gw   ])                # (12N,)
        Xref = np.hstack([Pref,  Vref,  Oref,      Wref ])                # (12N,)

        # === Weights from your simple setter ===
        Q_state12, R_force12 = self.set_weight(nx=12, nu=12)
        Q_big = sparse.kron(sparse.eye(N, format="csc"), sparse.csc_matrix(Q_state12), format="csc")  # (12N x 12N)
        R_big = sparse.kron(sparse.eye(N, format="csc"), sparse.csc_matrix(R_force12), format="csc")  # (12N x 12N)

        # === Quadratic program: 0.5 F^T P F + q^T F
        P = (Sx_s.T @ Q_big @ Sx_s) + R_big
        P = 0.5 * (P + P.T)  # symmetrize for OSQP
        P = P.tocsc()        # ensure CSC for OSQP

        # Build q as dense 1-D vector (avoid sparse/matrix types)
        delta_x = (gx - Xref)                      # (12N,)
        t1 = Sx_s.T.dot(Q_big.dot(delta_x))        # may be np.matrix / sparse
        t2 = R_big.dot(Fref)                       # may be np.matrix / sparse

        # Convert both to flat dense arrays
        if hasattr(t1, "toarray"):  # sparse -> dense
            t1 = t1.toarray()
        t1 = np.asarray(t1, dtype=np.float64).ravel()

        if hasattr(t2, "toarray"):
            t2 = t2.toarray()
        t2 = np.asarray(t2, dtype=np.float64).ravel()

        q = (t1 - t2).astype(np.float64)           # (12N,)

        # === Inequalities: friction pyramid + normal force bounds, per stage ===
        A_rows, l_list, u_list = [], [], []
        for k in range(N):
            S_k = S_block[12 * k : 12 * (k + 1), 12 * k : 12 * (k + 1)]
            A_ineq_k, l_k, u_k = self._friction_pyramid_blocks(S_k)
            A_block = np.zeros((A_ineq_k.shape[0], 12 * N))
            A_block[:, 12 * k : 12 * (k + 1)] = A_ineq_k
            A_rows.append(A_block); l_list.append(l_k); u_list.append(u_k)

        if A_rows:
            A = sparse.csc_matrix(np.vstack(A_rows))
            l = np.hstack(l_list).astype(float)
            u = np.hstack(u_list).astype(float)
        else:
            A = sparse.csc_matrix((0, 12 * N)); l = np.array([], float); u = np.array([], float)

        # === Solve QP ===
        prob = self._osqp
        prob.setup(P=P, q=q, A=A, l=l, u=u, warm_start=self._is_warm, verbose=False)
        res = prob.solve()
        if res.info.status_val not in (1, 2):
            # Infeasible: return zeros (and you’ll see residual print below)
            F = np.zeros(12 * N)
            self._is_warm = False
        else:
            F = res.x
            self._is_warm = True

        # First control (12,) — return GRFs for stage 0
        f0 = F[:12]

        # --------- quick residual debug (current first) ----------
        def _debug_residuals(p0, feet_world_4x3, f12, mass, grav=np.array([0,0,-9.81], float)):
            p0 = np.asarray(p0, float).reshape(3,)
            Fk = np.asarray(f12, float).reshape(4,3)
            feet = np.asarray(feet_world_4x3, float).reshape(4,3)
            m = float(mass)
            g = np.asarray(grav, float).reshape(3,)
            netF = Fk.sum(axis=0) + m*g
            tau = np.zeros(3)
            for i in range(4):
                tau += np.cross(feet[i] - p0, Fk[i])
            print("[resid] netF:", netF, " tau:", tau)

        # _debug_residuals(p0, r_over_h[0,:,:], f0.reshape(4,3), self.m, self.gvec)

        # --- rollout for logging/preview (p, v, theta, omega) ---
        pred = np.zeros((N + 1, 12))
        p = p0.copy(); v = v0.copy(); th = theta0.copy(); w = w0.copy()
        pred[0, :] = np.hstack([p, v, th, w])
        for k in range(N):
            fk = F[12 * k : 12 * (k + 1)].reshape(4, 3)
            acc = self.gvec + (1.0 / self.m) * fk.sum(axis=0)
            tau = np.zeros(3)
            for i in range(4):
                tau += np.cross(r_over_h[k, i, :] - p0, fk[i])  # torque arm uses p0 (frozen)
            alpha = I_world0_inv @ tau
            p = p + dt * v
            v = v + dt * acc
            th = th + dt * w
            w = w + dt * alpha
            pred[k + 1, :] = np.hstack([p, v, th, w])

        footholds = [r_over_h[0, i, :] for i in range(4)]
        return f0, footholds, pred

    def reset(self):
        """
        Reset the QP solver state (NMPC parity).
        Clears warm start and reinitializes the OSQP workspace.
        """
        self._osqp = osqp.OSQP()
        self._is_warm = False
