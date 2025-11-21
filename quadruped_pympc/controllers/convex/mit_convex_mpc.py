import numpy as np
from scipy import sparse
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
from casadi import MX, vertcat, horzcat, reshape, cos, sin

_LEG_ORDER = ("FL", "FR", "RL", "RR")


class MITConvexCentroidalMPC:
    """
    MIT-cheetah-style convex centroidal MPC (affine, time-varying) with your state order:
      x = [p(3); v(3); theta(3); omega(3)], u = [f_FL; f_FR; f_RL; f_RR] \in R^{12}.

      Discrete dynamics:
        p_{k+1}     = p_k + dt * v_k
        v_{k+1}     = v_k + dt * ( g + (1/m) * Σ_i f_{i,k} )
        theta_{k+1} = theta_k + dt * Rz(psi_k)^T * omega_k
        omega_{k+1} = omega_k + dt * Iw^{-1}(psi_k) * Σ_i S(r_{i,k}-pbar_k) f_{i,k}

      Stage cost:    ||x - xref||_Q^2 + ||u - uref||_R^2 + ||u - u_last||_{Rdu}^2
      Terminal cost: ||x_N - xref_N||_Q^2
      Constraints: friction pyramid + fz upper bound via general ineq;
                   stance fz_min & swing u=0 via per-stage box bounds.
    """

    def __init__(
        self,
        mass: float,
        inertia: np.ndarray,
        N: int,
        dt: float,
        g: float = 9.81,
        mu: float = 1.0,
        fz_min: float = 0.0,
        fz_max: float = 1e3,
        Qp: np.ndarray | float = (200.0, 200.0, 200.0),
        Qv: np.ndarray | float = (250.0, 250.0, 250.0),
        Qt: np.ndarray | float = (300.0, 200.0, 300.0),
        Qw: np.ndarray | float = (200.0, 200.0, 100.0),
        Rf: float | np.ndarray = None,
        Rdu: float | np.ndarray = None,
    ):
        # plant
        self.m = float(mass)
        self.Ib = np.array(inertia, dtype=float)   # 3x3 body-frame inertia at CoM
        self.N = int(N)
        self.dt = float(dt)
        self.Tf = self.N * self.dt
        self.g = float(g)
        self.gvec = np.array([0.0, 0.0, -self.g])
        self.mu = float(mu)
        self.fz_min = float(fz_min)
        self.fz_max = float(fz_max)

        # weights (anisotropic on forces so fx,fy can develop)
        self.Qp, self.Qv, self.Qt, self.Qw = Qp, Qv, Qt, Qw
        if Rf is None:
            Rf = np.kron(np.eye(4), np.diag([1e-6, 1e-6, 5e-4]))
        if Rdu is None:
            # slightly stronger Δu penalty on fz (stabilize contact transitions)
            Rdu = np.kron(np.eye(4), np.diag([2e-9, 2e-9, 5e-9]))
        self.Rf, self.Rdu = Rf, Rdu

        self.u_last = None  # Δu anchor
        self._cmd_prev = {"vx": 0.0, "vy": 0.0, "yr": 0.0}  # for optional slew-limit

        # ---------------- Build dynamics x+ = A(psi) x + B(psi,r,pbar) u + c ----------------
        nx, nu = 12, 12
        x = MX.sym("x", nx)          # [p, v, theta, omega]
        u = MX.sym("u", nu)          # GRFs stacked

        # per-stage params
        pbar = MX.sym("pbar", 3)     # reference CoM for moment arms
        Iinv = MX.sym("Iinv", 3, 3)  # world-frame inverse inertia (from yaw)
        rFL = MX.sym("rFL", 3); rFR = MX.sym("rFR", 3)
        rRL = MX.sym("rRL", 3); rRR = MX.sym("rRR", 3)
        psi = MX.sym("psi")          # yaw angle parameter

        dt = self.dt

        # yaw rotation
        cz, sz = cos(psi), sin(psi)
        Rz = vertcat(
            horzcat( cz, -sz, 0),
            horzcat( sz,  cz, 0),
            horzcat(  0,   0, 1)
        )

        # A(psi): rows [p; v; theta; omega], cols in same order
        A_row_p     = horzcat(MX.eye(3),    dt*MX.eye(3), MX.zeros(3,3),       MX.zeros(3,3))
        A_row_v     = horzcat(MX.zeros(3,3), MX.eye(3),   MX.zeros(3,3),       MX.zeros(3,3))
        A_row_theta = horzcat(MX.zeros(3,3), MX.zeros(3,3), MX.eye(3),         dt*Rz.T)
        A_row_omega = horzcat(MX.zeros(3,3), MX.zeros(3,3), MX.zeros(3,3),     MX.eye(3))
        A = vertcat(A_row_p, A_row_v, A_row_theta, A_row_omega)

        # c = [0; dt*g; 0; 0]
        c = MX(np.concatenate([np.zeros(3), dt*self.gvec, np.zeros(3), np.zeros(3)]))

        # helper: skew matrix S(a) so S(a)@b = a × b
        def _skew(a: MX) -> MX:
            ax, ay, az = a[0], a[1], a[2]
            row1 = horzcat( MX(0), -az,    ay)
            row2 = horzcat(  az,   MX(0), -ax)
            row3 = horzcat( -ay,    ax,   MX(0))
            return vertcat(row1, row2, row3)

        # per-leg B block with state order [p; v; theta; omega]
        def _leg_block(arm: MX) -> MX:
            Bp     = MX.zeros(3,3)
            Bv     = (dt / self.m) * MX.eye(3)
            Btheta = MX.zeros(3,3)
            Bomega = dt * (Iinv @ _skew(arm))
            return vertcat(Bp, Bv, Btheta, Bomega)

        aFL = rFL - pbar
        aFR = rFR - pbar
        aRL = rRL - pbar
        aRR = rRR - pbar
        B = horzcat(_leg_block(aFL), _leg_block(aFR),
                    _leg_block(aRL), _leg_block(aRR))

        # register model
        model = AcadosModel()
        model.name = "mit_convex_mpc"
        model.x = x
        model.u = u
        # params: pbar(3) + rFL(3)+rFR(3)+rRL(3)+rRR(3) + vec(Iinv)(9) + psi(1) = 25
        model.p = vertcat(pbar, rFL, rFR, rRL, rRR, reshape(Iinv, 9, 1), psi)
        model.disc_dyn_expr = A @ x + B @ u + c
        self.model = model

        # ---------------- OCP/QP ----------------
        ocp = AcadosOcp()
        ocp.model = model

        ocp.dims.nx = nx
        ocp.dims.nu = nu
        ocp.dims.ny = nx + 2 * nu     # [x; u; u_last]
        ocp.dims.ny_e = nx
        ocp.dims.np = 25

        ocp.solver_options.N_horizon = self.N
        ocp.solver_options.tf = float(self.Tf)
        ocp.parameter_values = np.zeros(25)

        # Costs
        Q, R, Rdu = self._build_weights()
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"
        ocp.cost.W   = sparse.block_diag([Q, R, Rdu]).toarray()
        ocp.cost.W_e = Q

        Vx = np.zeros((nx + 2*nu, nx))
        Vu = np.zeros((nx + 2*nu, nu))
        Vx[0:nx, 0:nx] = np.eye(nx)
        Vu[nx:nx+nu, :] = np.eye(nu)
        Vu[nx+nu:,   :] = np.eye(nu)
        ocp.cost.Vx = Vx
        ocp.cost.Vu = Vu
        ocp.cost.Vx_e = np.eye(nx)
        ocp.cost.yref = np.zeros(nx + 2*nu)
        ocp.cost.yref_e = np.zeros(nx)

        # --- Global general inequalities: friction pyramid + fz upper bound ONLY ---
        # IMPORTANT: do NOT put fz_min here; handle stance fz_min via per-stage box-bounds.
        mu = self.mu
        A_ineq = []
        lg, ug = [], []
        for i in range(4):
            blk = np.zeros((5, 12))
            blk[0, 3*i:3*i+3] = [ 1,  0, -mu]  #  fx - mu fz <= 0
            blk[1, 3*i:3*i+3] = [-1,  0, -mu]  # -fx - mu fz <= 0
            blk[2, 3*i:3*i+3] = [ 0,  1, -mu]  #  fy - mu fz <= 0
            blk[3, 3*i:3*i+3] = [ 0, -1, -mu]  # -fy - mu fz <= 0
            blk[4, 3*i:3*i+3] = [ 0,  0,  1]   #  fz <= fz_max
            A_ineq.append(blk)
            lg.extend([-1e8, -1e8, -1e8, -1e8, -1e8])    # no lower bounds here
            ug.extend([0.0,   0.0,   0.0,   0.0, self.fz_max])
        A_ineq = np.vstack(A_ineq).astype(float)
        ocp.dims.ng = A_ineq.shape[0]
        ocp.constraints.C  = np.zeros((ocp.dims.ng, nx), dtype=float)
        ocp.constraints.D  = A_ineq
        ocp.constraints.lg = np.asarray(lg, dtype=float)
        ocp.constraints.ug = np.asarray(ug, dtype=float)

        # Box bounds for u (swing legs set to 0 per stage; stance fz_min via box)
        ocp.dims.nbu = nu
        ocp.constraints.idxbu = np.arange(nu, dtype=np.int32)
        ocp.constraints.lbu = -1e8 * np.ones(nu)
        ocp.constraints.ubu =  1e8 * np.ones(nu)

        # Initial state equality x(0) = measured
        ocp.dims.nbx_0 = nx
        ocp.constraints.idxbx_0 = np.arange(nx, dtype=np.int32)
        ocp.constraints.lbx_0 = np.zeros(nx)
        ocp.constraints.ubx_0 = np.zeros(nx)

        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"

        self.ocp = ocp
        self.solver = AcadosOcpSolver(ocp, json_file="centroidal_qp_mit_yawA.json")

    # ---------- utilities ----------
    def _as_mat(self, val, n):
        if np.isscalar(val): return np.eye(n) * float(val)
        A = np.array(val, dtype=float)
        if A.shape == (n,):  return np.diag(A)
        if A.shape == (n, n): return A
        raise ValueError(f"Weight must be scalar, len-{n} vector, or {n}x{n}; got {A.shape}")

    def _build_weights(self, nx=12, nu=12):
        Qp = self._as_mat(self.Qp, 3)
        Qv = self._as_mat(self.Qv, 3)
        Qt = self._as_mat(self.Qt, 3)
        Qw = self._as_mat(self.Qw, 3)
        Q = sparse.block_diag([Qp, Qv, Qt, Qw]).toarray()

        R = self.Rf
        if np.isscalar(R):
            R = np.kron(np.eye(4), np.eye(3) * float(R))
        else:
            R = np.array(R, dtype=float)
            if R.shape == (3, 3):       R = np.kron(np.eye(4), R)
            elif R.shape == (12, 12):   pass
            else: raise ValueError(f"Rf must be scalar, 3x3, or 12x12; got {R.shape}")

        Rdu = self.Rdu
        if np.isscalar(Rdu):
            Rdu = np.kron(np.eye(4), np.eye(3) * float(Rdu))
        else:
            Rdu = np.array(Rdu, dtype=float)
            if Rdu.shape == (3, 3):       Rdu = np.kron(np.eye(4), Rdu)
            elif Rdu.shape == (12, 12):   pass
            else: raise ValueError(f"Rdu must be scalar, 3x3, or 12x12; got {Rdu.shape}")

        return Q, np.array(R, float), np.array(Rdu, float)

    def _R_from_euler_xyz(self, euler_xyz):
        rx, ry, rz = euler_xyz
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
        Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
        Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
        return Rz @ Ry @ Rx   # intrinsic XYZ

    def _contacts_to_4xN(self, contact_sequence):
        arr = np.array(contact_sequence, dtype=int)
        if arr.shape == (self.N, 4): arr = arr.T
        assert arr.shape == (4, self.N), f"contact_sequence must be (4,N) or (N,4); got {arr.shape}"
        return arr

    def _feet_refs_over_horizon(self, ref_state, contacts4xN):
        N = self.N
        rFL_all = np.atleast_2d(ref_state["ref_foot_FL"])
        rFR_all = np.atleast_2d(ref_state["ref_foot_FR"])
        rRL_all = np.atleast_2d(ref_state["ref_foot_RL"])
        rRR_all = np.atleast_2d(ref_state["ref_foot_RR"])
        idx = np.array([0, 0, 0, 0], dtype=int)
        r_over_h = np.zeros((N, 4, 3))
        FLc, FRc, RLc, RRc = contacts4xN[0], contacts4xN[1], contacts4xN[2], contacts4xN[3]
        for j in range(N):
            r_over_h[j,0] = rFL_all[min(idx[0], rFL_all.shape[0]-1)]
            r_over_h[j,1] = rFR_all[min(idx[1], rFR_all.shape[0]-1)]
            r_over_h[j,2] = rRL_all[min(idx[2], rRL_all.shape[0]-1)]
            r_over_h[j,3] = rRR_all[min(idx[3], rRR_all.shape[0]-1)]
            if j < N-1:
                if FLc[j]==1 and FLc[j+1]==0 and idx[0]+1<rFL_all.shape[0]: idx[0]+=1
                if FRc[j]==1 and FRc[j+1]==0 and idx[1]+1<rFR_all.shape[0]: idx[1]+=1
                if RLc[j]==1 and RLc[j+1]==0 and idx[2]+1<rRL_all.shape[0]: idx[2]+=1
                if RRc[j]==1 and RRc[j+1]==0 and idx[3]+1<rRR_all.shape[0]: idx[3]+=1
        return r_over_h

    def _build_u_ref_mg_split(self, contacts4xN):
        N = self.N
        u_ref = np.zeros((N, 12))
        for k in range(N):
            n_stance = int(contacts4xN[:, k].sum())
            if n_stance <= 0:
                continue
            fz = self.m * self.g / n_stance
            for i in range(4):
                if contacts4xN[i, k] == 1:
                    u_ref[k, 3*i:3*i+3] = [0.0, 0.0, fz]
        return u_ref

    # ---------- main solve ----------
    def compute_control(self, state_current, ref_state, contact_sequence):
        """
        Returns:
            nmpc_GRFs: (12,) GRFs at k=0
            nmpc_footholds: (4,3) foot positions at stage 0
            nmpc_predicted_state: dict with 'x_traj' (N+1,12)
        """
        N = self.N
        dt = self.dt

        p0 = np.array(state_current["position"]).reshape(3)
        v0 = np.array(state_current["linear_velocity"]).reshape(3)
        w0 = np.array(state_current["angular_velocity"]).reshape(3)
        theta0 = np.array(state_current.get("orientation", np.zeros(3))).reshape(3)

        # ---------- Extract commanded (vx, vy, yawrate, z) from ref_state ----------
        # Preferred keys: cmd_vxy / cmd_yawrate / cmd_z
        if "cmd_vxy" in ref_state:
            vx_cmd, vy_cmd = ref_state["cmd_vxy"]
        else:
            vref_in = np.array(ref_state["ref_linear_velocity"]).reshape(-1, 3)
            vx_cmd, vy_cmd = vref_in[0, 0], vref_in[0, 1]

        if "cmd_yawrate" in ref_state:
            yawrate_cmd = float(ref_state["cmd_yawrate"])
        else:
            wref_in = np.array(ref_state["ref_angular_velocity"]).reshape(-1, 3)
            yawrate_cmd = float(wref_in[0, 2])

        if "cmd_z" in ref_state:
            z_cmd = float(ref_state["cmd_z"])
        else:
            if "ref_position" in ref_state:
                pref_in = np.array(ref_state["ref_position"]).reshape(-1, 3)
                z_cmd = float(pref_in[0, 2])
            else:
                z_cmd = float(p0[2])

        # Optional command slew limits (operator-level accel limits)
        a_max_xy, a_max_yaw = 1.5, 2.0
        def slew(prev, target, rate):
            step = np.clip(target - prev, -rate*dt, rate*dt)
            return prev + step
        vx_cmd = slew(self._cmd_prev["vx"], vx_cmd, a_max_xy)
        vy_cmd = slew(self._cmd_prev["vy"], vy_cmd, a_max_xy)
        yawrate_cmd = slew(self._cmd_prev["yr"], yawrate_cmd, a_max_yaw)
        self._cmd_prev = {"vx": vx_cmd, "vy": vy_cmd, "yr": yawrate_cmd}

        # Piecewise-constant (vx,vy,yawrate); integrate to p and yaw
        vref = np.tile(np.array([vx_cmd, vy_cmd, 0.0]), (N, 1))
        wref = np.tile(np.array([0.0, 0.0, yawrate_cmd]), (N, 1))
        pref = np.zeros((N, 3))
        oref = np.zeros((N, 3))  # [roll, pitch, yaw]
        pref[0] = np.array([p0[0], p0[1], z_cmd])
        oref[0] = np.array([0.0, 0.0, theta0[2]])
        for k in range(N-1):
            pref[k+1] = pref[k] + dt * vref[k]
            oref[k+1] = oref[k] + dt * wref[k]

        # ---------- contact plan & footholds ----------
        contacts = self._contacts_to_4xN(contact_sequence)
        feet_over_h = self._feet_refs_over_horizon(ref_state, contacts)

        # input references: split mg across stance feet
        u_ref = self._build_u_ref_mg_split(contacts)

        # per-stage Iinv from yaw
        Iinv_all = []
        for k in range(N):
            psi_k = oref[k, 2]
            cz, sz = np.cos(psi_k), np.sin(psi_k)
            Rz = np.array([[cz, -sz, 0],
                           [sz,  cz, 0],
                           [ 0,   0, 1]])
            Iw = Rz @ self.Ib @ Rz.T
            Iinv_all.append(np.linalg.inv(Iw))
        Iinv_all = np.array(Iinv_all)

        # Δu anchor
        if self.u_last is None:
            u0_last = np.zeros(12)
            n_stance0 = int(contacts[:, 0].sum())
            if n_stance0 > 0:
                fz = self.m * self.g / n_stance0
                for i in range(4):
                    if contacts[i, 0] == 1:
                        u0_last[3*i:3*i+3] = [0.0, 0.0, fz]
            self.u_last = u0_last
        u_last = self.u_last.copy()

        # ---------- costs ----------
        nx, nu = 12, 12
        for k in range(N):
            xref_k = np.concatenate([pref[k], vref[k], oref[k], wref[k]])
            yk = np.concatenate([xref_k, u_ref[k], u_last])
            self.solver.set(k, "yref", yk.astype(float))
        yN = np.concatenate([pref[-1], vref[-1], oref[-1], wref[-1]])
        self.solver.set(N, "yref", yN.astype(float))  # terminal via yref at stage N

        # initial state equality + warm start baseline
        x0 = np.concatenate([p0, v0, theta0, w0]).astype(float)
        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)
        self.solver.set(0, "x", x0)

        # ---------- parameters & per-stage box bounds ----------
        for k in range(N):
            rFL, rFR, rRL, rRR = feet_over_h[k]
            pbar = pref[k]
            psi_k = float(oref[k, 2])
            Iinv = Iinv_all[k]
            params = np.concatenate([pbar, rFL, rFR, rRL, rRR, Iinv.reshape(-1), [psi_k]])
            self.solver.set(k, "p", params)

            # Box bounds on u
            lbu = -1e8 * np.ones(12)
            ubu =  1e8 * np.ones(12)
            for i in range(4):
                idx = 3*i
                if contacts[i, k] == 0:
                    # swing: clamp all three components to zero
                    lbu[idx:idx+3] = 0.0
                    ubu[idx:idx+3] = 0.0
                else:
                    # stance: fx,fy wide; enforce fz in [fz_min, fz_max]
                    lbu[idx+2] = self.fz_min
                    ubu[idx+2] = self.fz_max
            self.solver.set(k, "lbu", lbu)
            self.solver.set(k, "ubu", ubu)

        # ---------- warm-start u with u_ref ----------
        for k in range(N):
            self.solver.set(k, "u", u_ref[k])

        # ---------- warm-start x by rolling model with u_ref ----------
        def _skew_np(a):
            return np.array([[0, -a[2], a[1]],
                             [a[2], 0, -a[0]],
                             [-a[1], a[0], 0]])
        xk = x0.copy()
        self.solver.set(0, "x", xk)
        for k in range(N):
            psi_k = float(oref[k, 2])
            cz, sz = np.cos(psi_k), np.sin(psi_k)
            Rz = np.array([[cz, -sz, 0],
                           [sz,  cz, 0],
                           [ 0,   0, 1]])
            # A_k numeric
            A_k = np.eye(12)
            A_k[0:3, 3:6]   = dt * np.eye(3)
            A_k[6:9, 9:12]  = dt * Rz.T
            # B_k numeric
            rFL, rFR, rRL, rRR = feet_over_h[k]
            pbar = pref[k]
            arms = [rFL - pbar, rFR - pbar, rRL - pbar, rRR - pbar]
            Iinv = Iinv_all[k]
            B_k = np.zeros((12, 12))
            for i_leg, arm in enumerate(arms):
                B_k[3:6, 3*i_leg:3*i_leg+3]  = (dt / self.m) * np.eye(3)
                B_k[9:12, 3*i_leg:3*i_leg+3] = dt * (Iinv @ _skew_np(arm))
            c_k = np.concatenate([np.zeros(3), dt*self.gvec, np.zeros(3), np.zeros(3)])
            xk = A_k @ xk + B_k @ u_ref[k] + c_k
            self.solver.set(k+1, "x", xk)

        # ---------- solve ----------
        status = self.solver.solve()
        if status != 0:
            print(f"[Acados/QP] solver failed with status {status}, returning zeros")
            return np.zeros(12), feet_over_h[0], {"x_traj": np.zeros((N+1, 12))}

        u0 = np.array(self.solver.get(0, "u")).reshape(-1)
        feet0 = feet_over_h[0].copy()
        self.u_last = u0.copy()

        x_traj = np.zeros((N+1, 12))
        for k in range(N+1):
            xk = self.solver.get(k, "x")
            x_traj[k] = np.array(xk).reshape(12)

        return u0, feet0, {"x_traj": x_traj}
    
    def reset(self):
        """
        Clear warm starts and internal anchors between episodes.
        Safe to call before the first compute_control() of a new rollout.
        """
        # reset internal anchors for references & delta-u penalty
        self.u_last = None
        self._cmd_prev = {"vx": 0.0, "vy": 0.0, "yr": 0.0}

        # optional: clear acados warm-starts (ignore if solver not yet fully built)
        try:
            nx, nu, N = 12, 12, self.N
            zero_x = np.zeros(nx, dtype=float)
            zero_u = np.zeros(nu, dtype=float)
            for k in range(N):
                self.solver.set(k, "u", zero_u)
            for k in range(N + 1):
                self.solver.set(k, "x", zero_x)
        except Exception:
            pass