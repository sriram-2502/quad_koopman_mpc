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

    Centroidal SRBD dynamics (discrete Euler; state uses names p, v, theta, omega):
        p_{k+1}     = p_k + dt * v_k
        v_{k+1}     = v_k + dt * ( g + (1/m) * sum_i f_{i,k} )
        theta_{k+1} = theta_k + dt * omega_k
        omega_{k+1} = omega_k + dt * I^{-1} * sum_i ( (r_i[k] - p_ref) x f_{i,k} )

    We track desired p, v, theta, omega over the horizon, and regularize forces.

    Assumptions (unchanged):
      - Foot positions r_i are piecewise-constant (held at the current waypoint during stance/swing).
      - Inertia I is constant in body frame (diagonal is fine).
      - Short-horizon: torque arm uses p_ref ≈ current base p0.
      - Contact schedule provided as contact_sequence (shape (4, N)), 1=stance, 0=swing.
      - Friction constraints: pyramid (|fx| <= mu fz, |fy| <= mu fz, fz ∈ [fz_min, fz_max]).
    """

    def __init__(
        self,
        mass: float,
        inertia: np.ndarray,  # 3x3
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
        self.I = np.array(inertia, dtype=float)
        self.I_inv = np.linalg.inv(self.I)
        self.N = int(N)
        self.dt = config.mpc_params['dt']
        self.gvec = np.array([0.0, 0.0, -float(g)], dtype=float)
        self.mu = float(mu)
        self.fz_min = float(fz_min)
        self.fz_max = float(fz_max)

        # weights (3x3 each if arrays; otherwise scalar*I)
        self.Qp = np.eye(3)*Qp if np.isscalar(Qp) else np.array(Qp, dtype=float)
        self.Qv = np.eye(3)*Qv if np.isscalar(Qv) else np.array(Qv, dtype=float)
        self.Qt = np.eye(3)*Qt if np.isscalar(Qt) else np.array(Qt, dtype=float)
        self.Qw = np.eye(3)*Qw if np.isscalar(Qw) else np.array(Qw, dtype=float)
        self.Rf = float(Rf)

        # OSQP workspace (reused)
        self._osqp = osqp.OSQP()
        self._is_warm = False
        self._P = None
        self._A = None
        self._l = None
        self._u = None

        # (kept for parity if you re-enable scaling later)
        self.initial_base_position = np.zeros(3)

    # ---- internal helpers ----
    def _build_selection(self, contacts_k):
        """Return a diagonal block selector S_k (12x12) that zeros swing leg forces at stage k."""
        S = np.zeros((12, 12))
        for i, _ in enumerate(_LEG_ORDER):
            on = 1.0 if contacts_k[i] else 0.0
            S[3*i:3*i+3, 3*i:3*i+3] = on*np.eye(3)
        return S
    
    def _R_from_euler(self, euler_xyz):
        rx, ry, rz = euler_xyz
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
        Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
        Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
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
        Build per-stage foot references r_i[k] (i in {FL,FR,RL,RR}), size (N,4,3),
        mimicking the NMPC logic:
          - At each stage j, use the current index for each leg's ref_foot_*.
          - When a leg transitions stance->swing (1->0) between j and j+1,
            advance that leg's index (if another waypoint exists).
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
    
    def debug_residuals(p0, feet_world, f, mass, g=np.array([0,0,-9.81])):
        # p0, feet_world: (3,), (4,3) in WORLD frame at k=0
        # f: (12,) stacked [FL,FR,RL,RR] with (fx,fy,fz) per foot at k=0
        F = f.reshape(4,3)
        netF = F.sum(axis=0) + mass * g
        tau = np.zeros(3)
        for i in range(4):
            r = feet_world[i] - p0
            tau += np.cross(r, F[i])
        print("[debug] net force (should be ~0):", netF)
        print("[debug] net moment about COM (should be ~0):", tau)
        # Useful projections
        print("[debug] pitch moment (y):", tau[1])
        print("[debug] total fz and mg:", F[:,2].sum(), mass*abs(g[2]))

    # ---- main API ----
    def compute_control(self, state_current, ref_state, contact_sequence, inertia=None):
        """
        Keys follow Acados_NMPC_Nominal:
        state_current:
            - position            : (3,)
            - linear_velocity     : (3,)
            - angular_velocity    : (3,)
            - orientation         : (3,)  # used if available for theta0
        ref_state:
            - ref_position        : (N x 3) or (1 x 3)
            - ref_linear_velocity : (N x 3) or (1 x 3)
            - ref_orientation     : (N x 3) or (1 x 3)
            - ref_angular_velocity: (N x 3) or (1 x 3)
            - ref_foot_FL, ref_foot_FR, ref_foot_RL, ref_foot_RR : (T_i x 3)
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

        # assuming yaw is zero or constant
        R0 = self._R_from_euler(ref_state["ref_orientation"])
        I_world0 = R0 @ self.I @ R0.T
        I_world0_inv = np.linalg.inv(I_world0)

        # Per-stage foot references (N,4,3)
        r_over_h = self._feet_refs_over_horizon(ref_state, contacts)

        # --- build v, omega maps (core dynamics) ---
        Bv, Bw, gv, gw, S_block = self._stack_dynamics_maps(p0, v0, w0, r_over_h, contacts, I_world0_inv)
        Bv_s = sparse.csc_matrix(Bv)
        Bw_s = sparse.csc_matrix(Bw)

        # --- integrate to get position (p) and orientation (theta) maps ---
        # Build strictly-lower block accumulator C (row k sums cols 0..k-1).
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

        # p map: Pstack = (dt*C)* (Bv F + gv) + kron(1_N, p0 + dt*v0)
        Bp_s = (self.dt * C_s) @ Bv_s
        gp = self.dt * (C_s @ gv)  # gv is (3N,)
        gp = np.asarray(gp).reshape(3*N) + np.kron(oneN, (p0)).reshape(3*N)

        # theta map: Tstack = (dt*C)* (Bw F + gw) + kron(1_N, theta0 + dt*w0)
        Btheta_s = (self.dt * C_s) @ Bw_s
        gtheta = self.dt * (C_s @ gw)  # gw is (3N,)
        gtheta = np.asarray(gtheta).reshape(3*N) + np.kron(oneN, (theta0)).reshape(3*N)

        # === Quadratic cost: track p, v, theta, omega; regularize forces ===
        Qp_blk = sparse.kron(sparse.eye(N, format="csc"), sparse.csc_matrix(self.Qp))
        Qv_blk = sparse.kron(sparse.eye(N, format="csc"), sparse.csc_matrix(self.Qv))
        Qt_blk = sparse.kron(sparse.eye(N, format="csc"), sparse.csc_matrix(self.Qt))
        Qw_blk = sparse.kron(sparse.eye(N, format="csc"), sparse.csc_matrix(self.Qw))

        P = (
            (Bp_s.T @ Qp_blk @ Bp_s)
            + (Bv_s.T @ Qv_blk @ Bv_s)
            + (Btheta_s.T @ Qt_blk @ Btheta_s)
            + (Bw_s.T @ Qw_blk @ Bw_s)
            + self.Rf * sparse.eye(12 * N, format="csc")
        )
        P = 0.5 * (P + P.T)

        q = (
            (Bp_s.T @ (Qp_blk @ (gp - Pref)))
            + (Bv_s.T @ (Qv_blk @ (gv - Vref)))
            + (Btheta_s.T @ (Qt_blk @ (gtheta - Oref)))
            + (Bw_s.T @ (Qw_blk @ (gw - Wref)))
        )
        q = np.asarray(q).reshape(-1)

        # === Force references from contact schedule (fx=fy=0; fz=m*g/legs_in_stance) ===
        g_mag = float(abs(self.gvec[2])) if self.gvec.shape[0] == 3 else 9.81
        Fref_list = []
        for k in range(N):
            c = contacts[:, k].astype(float)  # [FL, FR, RL, RR] in {0,1}
            legs_in_stance = int(c.sum())
            fz_each = (self.m * g_mag / legs_in_stance) if legs_in_stance > 0 else 0.0
            f_FL = np.array([0.0, 0.0, fz_each * c[0]])
            f_FR = np.array([0.0, 0.0, fz_each * c[1]])
            f_RL = np.array([0.0, 0.0, fz_each * c[2]])
            f_RR = np.array([0.0, 0.0, fz_each * c[3]])
            Fref_list.append(np.hstack([f_FL, f_FR, f_RL, f_RR]))
        Fref = np.hstack(Fref_list)  # (12*N,)

        # Regularization shift: 0.5 F^T R F -> q := q - R * Fref  (R = Rf*I here)
        if np.isscalar(self.Rf):
            q = q - float(self.Rf) * Fref
        else:
            R_blk = sparse.kron(sparse.eye(N, format="csc"), sparse.csc_matrix(self.Rf))
            q = q - np.asarray(R_blk @ Fref).reshape(-1)

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
            F = np.zeros(12 * N)
        else:
            F = res.x
            self._is_warm = True

        # First control (12,) — return GRFs for stage 0
        f0 = F[:12]

        # --------- quick debug (current vs reference) ----------
        curr_fx = np.asarray(f0)[0::3]
        curr_fy = np.asarray(f0)[1::3]
        curr_fz = np.asarray(f0)[2::3]
        ref_fx0 = np.asarray(Fref[:12])[0::3]
        ref_fy0 = np.asarray(Fref[:12])[1::3]
        ref_fz0 = np.asarray(Fref[:12])[2::3]

        # --------- debug (current first, then reference) ----------
        # print(f"[DEBUG] euler={theta0} | f0(x,y,z)=({curr_fx}, {curr_fy}, {curr_fz})")
        # print(f"[DEBUG] ref={oref[0]} | ref(x,y,z)=({ref_fx0}, {ref_fy0}, {ref_fz0})")
        # print(f"[DEBUG] euler={theta0} | f0(x,y,z)=({curr_fx}")
        # print(f"[DEBUG] ref={oref[0]} | ref(x,y,z)=({ref_fx0}")
        # ----------------------------------------------------------

        # -------------------------------------------------------

        # --- rollout for logging/preview (p, v, theta, omega) ---
        pred = np.zeros((N + 1, 12))
        p = p0.copy(); v = v0.copy(); th = theta0.copy(); w = w0.copy()
        pred[0, :] = np.hstack([p, v, th, w])
        for k in range(N):
            fk = F[12 * k : 12 * (k + 1)].reshape(4, 3)
            acc = self.gvec + (1.0 / self.m) * fk.sum(axis=0)
            tau = np.zeros(3)
            # use per-stage feet for torque arm
            for i in range(4):
                tau += np.cross(r_over_h[k, i, :] - p, fk[i])
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
