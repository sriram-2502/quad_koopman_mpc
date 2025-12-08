"""
Koopman convex MPC (acados) using lifted linear dynamics learned via EDMD.

Dynamics: z_{k+1} = A z_k + B̄ ū_k,  ū = [GRF(12); wrench(6)],  B̄ = [0 B].
Only the wrench drives the lifted dynamics; GRFs are constrained to match the wrench
through net force/torque equalities and friction/normal bounds.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence, Dict, Any

import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
from casadi import MX, vertcat, horzcat, reshape

try:
    from simulation.edmd.edmd_runner import lift_1d  # type: ignore
except Exception:  # pragma: no cover
    lift_1d = None  # type: ignore

class KoopmanConvexMPC:
    """
    acados-based convex MPC with Koopman lifted dynamics.

    Parameters
    ----------
    model_path : Path | str
        npz with A,B (and optional meta) from EDMD (wrench-driven).
    N : int
        Horizon length.
    dt : float
        Sample time.
    mu : float
        Friction coefficient.
    fz_min, fz_max : float
        Normal force bounds for stance legs.
    Qx : float or sequence
        Weights on first 12 physical states; zeros for remaining lifted dims.
    R_grf : float or matrix (12x12)
        GRF weight.
    R_wrench : float or matrix (6x6)
        Wrench weight.
    lift_fn : callable | None
        Function x->phi(x); defaults to edmd_runner.lift_1d if available else identity.
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
        Qp: Sequence[float] | float = (200.0, 200.0, 200.0),
        Qv: Sequence[float] | float = (2.0, 2.0, 2.0),
        Qt: Sequence[float] | float = (300.0, 200.0, 300.0),
        Qw: Sequence[float] | float = (200.0, 200.0, 100.0),
        Qx: Optional[Sequence[float] | float] = None,
        R_grf: Optional[Sequence[float] | float] = None,
        R_wrench: Sequence[float] | float = 1e-6,
        slack_wrench_force: float = 1e3,
        slack_wrench_torque: float = 1e3,
        model_path: Path | str = None,
        lift_fn=None,
    ):
        if model_path is None:
            # Default to koopman/models/wrench_model.npz
            self.model_path = Path(__file__).resolve().parent / "models" / "wrench_model.npz"
        else:
            self.model_path = Path(model_path)
        self.N = int(N)
        self.dt = float(dt)
        self.mu = float(mu)
        self.fz_min = float(fz_min)
        self.fz_max = float(fz_max)

        data = np.load(self.model_path, allow_pickle=True)
        self.A = np.asarray(data["A"], float)
        self.B = np.asarray(data["B"], float)
        self.meta = data["meta"].item() if "meta" in data.files and np.asarray(data["meta"]).size == 1 else {}

        self.nphi = self.A.shape[0]
        self.nu_w = self.B.shape[1]  # expected 6
        self.nu_f = 12
        self.nu = self.nu_f + self.nu_w
        self.B_bar = np.hstack([np.zeros((self.nphi, self.nu_f)), self.B])

        if lift_fn is not None:
            self.lift_fn = lift_fn
        elif lift_1d is not None:
            p_max = int(self.meta.get("p_max", 5)) if isinstance(self.meta, dict) else 5
            self.lift_fn = lambda x: lift_1d(np.asarray(x, float), p_max)
        else:
            self.lift_fn = lambda x: np.asarray(x, float)

        # Build Qz: if Qx provided, use generic builder; else use block weights
        if Qx is not None:
            self.Qz = self._build_Qz_generic(Qx)
        else:
            self.Qz = self._build_Qz_from_blocks(Qp, Qv, Qt, Qw)
            
        # Default anisotropic GRF weight: light fx, fy; heavier fz per leg
        if R_grf is None:
            R_grf = np.kron(np.eye(4), np.diag([1e-9, 1e-9, 5e-4]))
        self.Rf = self._as_mat(R_grf, self.nu_f)
        self.Rw = self._as_mat(R_wrench, self.nu_w)
        self.mass = float(mass)
        self.gvec = np.array([0.0, 0.0, -abs(g)], float)

        # Soft-constraint penalties for wrench equality (force/torque)
        self.slack_wrench_force = float(slack_wrench_force)
        self.slack_wrench_torque = float(slack_wrench_torque)
        self.solver = self._build_solver()

    @staticmethod
    def _as_mat(val, n: int) -> np.ndarray:
        if np.isscalar(val):
            return float(val) * np.eye(n)
        arr = np.asarray(val, float)
        if arr.shape == (n,):
            return np.diag(arr)
        if arr.shape == (n, n):
            return arr
        raise ValueError(f"Cannot form {n}x{n} matrix from shape {arr.shape}")

    def _build_Qz_from_blocks(self, Qp, Qv, Qt, Qw) -> np.ndarray:
        """
        Build lifted cost weights from per-group blocks (MIT convex MPC style).
        Each block can be scalar or len-3; concatenated in the SAME order used by
        the EDMD state (pos, ori, lin vel, ang vel) → [p, theta, v, omega].
        """
        Qz = np.zeros((self.nphi, self.nphi))
        def _as3(val):
            arr = np.asarray(val, float).flatten()
            if arr.size == 1:
                return np.full(3, arr.item(), float)
            if arr.size == 3:
                return arr
            raise ValueError("Qp/Qv/Qt/Qw entries must be scalar or len-3")
        # State ordering in EDMD lift: [p (0:3), theta (3:6), v (6:9), omega (9:12)]
        q = np.concatenate([_as3(Qp), _as3(Qt), _as3(Qv), _as3(Qw)], axis=0)
        Qz[np.arange(12), np.arange(12)] = q
        return Qz

    def _build_Qz_generic(self, Qx) -> np.ndarray:
        """
        Generic builder for backward compatibility:
          - scalar → scalar*I on first 12
          - len-12 vector → diag on first 12
          - 12x12 matrix → copied into top-left block
        """
        Qz = np.zeros((self.nphi, self.nphi))
        if np.isscalar(Qx):
            Qz[np.arange(12), np.arange(12)] = float(Qx)
            return Qz
        arr = np.asarray(Qx, float)
        if arr.shape == (12,):
            Qz[np.arange(12), np.arange(12)] = arr
            return Qz
        if arr.shape == (12, 12):
            Qz[:12, :12] = arr
            return Qz
        raise ValueError("Qx must be scalar, len-12 vector, or 12x12 matrix")

    def _build_solver(self) -> AcadosOcpSolver:
        nx = self.nphi
        nu = self.nu

        # symbols
        x = MX.sym("x", nx)
        u = MX.sym("u", nu)  # [GRF(12), wrench(6)]
        arms = MX.sym("arms", 12)  # lever arms per stage (flattened)

        # Dynamics: z+ = A z + B_bar u
        A_mx = MX(self.A)
        B_mx = MX(self.B_bar)
        x_next = A_mx @ x + B_mx @ u

        # Net wrench consistency h = 0 (6)
        f_mat = reshape(u[0 : self.nu_f], 4, 3)  # rows are legs
        w_vec = u[self.nu_f :]                   # (6,)
        # Force balance (3x1)
        f_sum = (f_mat[0, :] + f_mat[1, :] + f_mat[2, :] + f_mat[3, :]).T  # (3,1)
        force_expr = f_sum - w_vec[0:3]
        # Torque balance (3x1): sum (r x f)
        torque_terms = []
        for i in range(4):
            r = arms[3 * i : 3 * i + 3]
            skew = MX.zeros(3, 3)
            skew[0, 1] = -r[2]
            skew[0, 2] = r[1]
            skew[1, 0] = r[2]
            skew[1, 2] = -r[0]
            skew[2, 0] = -r[1]
            skew[2, 1] = r[0]
            fi = f_mat[i, :].T
            torque_terms.append(skew @ fi)
        torque_expr = torque_terms[0] + torque_terms[1] + torque_terms[2] + torque_terms[3] - w_vec[3:6]
        h_expr = vertcat(force_expr, torque_expr)

        model = AcadosModel()
        model.name = "koopman_convex_mpc"
        model.x = x
        model.u = u
        model.p = arms
        model.disc_dyn_expr = x_next
        model.con_h_expr = h_expr

        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.nx = nx
        ocp.dims.nu = nu
        ocp.dims.np = 12
        ocp.dims.N = self.N
        ocp.parameter_values = np.zeros(12)

        ny = nx + nu
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"
        W = np.zeros((ny, ny))
        W[:nx, :nx] = self.Qz
        W[nx:nx + self.nu_f, nx:nx + self.nu_f] = self.Rf
        W[nx + self.nu_f :, nx + self.nu_f :] = self.Rw
        ocp.cost.W = W
        ocp.cost.W_e = self.Qz

        Vx = np.zeros((ny, nx))
        Vu = np.zeros((ny, nu))
        Vx[:nx, :nx] = np.eye(nx)
        Vu[nx:, :] = np.eye(nu)
        ocp.cost.Vx = Vx
        ocp.cost.Vu = Vu
        ocp.cost.Vx_e = np.eye(nx)
        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(nx)

        # Friction pyramid + fz_max
        A_ineq = []
        lg = []
        ug = []
        mu = self.mu
        for i in range(4):
            blk = np.zeros((5, nu))
            blk[0, 3 * i : 3 * i + 3] = [1, 0, -mu]
            blk[1, 3 * i : 3 * i + 3] = [-1, 0, -mu]
            blk[2, 3 * i : 3 * i + 3] = [0, 1, -mu]
            blk[3, 3 * i : 3 * i + 3] = [0, -1, -mu]
            blk[4, 3 * i + 2] = 1
            A_ineq.append(blk)
            lg.extend([-1e8, -1e8, -1e8, -1e8, -1e8])
            ug.extend([0.0, 0.0, 0.0, 0.0, self.fz_max])
        A_ineq = np.vstack(A_ineq)
        ocp.dims.ng = A_ineq.shape[0]
        ocp.constraints.C = np.zeros((ocp.dims.ng, nx))
        ocp.constraints.D = A_ineq
        ocp.constraints.lg = np.asarray(lg, float)
        ocp.constraints.ug = np.asarray(ug, float)

        ocp.dims.nh = 6
        ocp.constraints.lh = np.zeros(6)
        ocp.constraints.uh = np.zeros(6)
        # Soften wrench equalities with slacks and quadratic penalties
        ocp.dims.nsh = 6
        ocp.constraints.idxsh = np.arange(6, dtype=np.int32)
        ocp.constraints.lsh = np.zeros(6)
        ocp.constraints.ush = np.zeros(6)
        Zw = np.diag([
            self.slack_wrench_force,
            self.slack_wrench_force,
            self.slack_wrench_force,
            self.slack_wrench_torque,
            self.slack_wrench_torque,
            self.slack_wrench_torque,
        ])
        ocp.cost.zl = np.zeros(6)
        ocp.cost.zu = np.zeros(6)
        ocp.cost.Zl = Zw
        ocp.cost.Zu = Zw

        ocp.dims.nbu = nu
        ocp.constraints.idxbu = np.arange(nu, dtype=np.int32)
        ocp.constraints.lbu = -1e8 * np.ones(nu)
        ocp.constraints.ubu = 1e8 * np.ones(nu)

        ocp.dims.nbx_0 = nx
        ocp.constraints.idxbx_0 = np.arange(nx, dtype=np.int32)
        ocp.constraints.lbx_0 = np.zeros(nx)
        ocp.constraints.ubx_0 = np.zeros(nx)

        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.tf = self.N * self.dt

        return AcadosOcpSolver(ocp, json_file=f"{model.name}.json")

    def solve(
        self,
        z0: np.ndarray,
        z_ref: np.ndarray,
        u_ref: np.ndarray,
        arms_seq: np.ndarray,
        contact_schedule: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        N = self.N
        nphi = self.nphi
        nu = self.nu

        z0 = np.asarray(z0, float).reshape(-1)
        z_ref = np.asarray(z_ref, float)
        u_ref = np.asarray(u_ref, float)
        arms_seq = np.asarray(arms_seq, float)

        if z_ref.shape != (N + 1, nphi):
            raise ValueError(f"z_ref must be (N+1,{nphi})")
        if u_ref.shape != (N, nu):
            raise ValueError(f"u_ref must be (N,{nu})")
        if arms_seq.shape != (N, 4, 3):
            raise ValueError("arms_seq must be (N,4,3)")

        solver = self.solver
        solver.cost_set(0, "yref", np.concatenate([z_ref[0], u_ref[0]]))
        solver.set(0, "lbx", z0)
        solver.set(0, "ubx", z0)

        lbu = -1e8 * np.ones(nu)
        ubu = 1e8 * np.ones(nu)
        if contact_schedule is not None:
            contact_schedule = np.asarray(contact_schedule, float)
            if contact_schedule.shape == (4, N):
                contact_schedule = contact_schedule.T
            if contact_schedule.shape != (N, 4):
                raise ValueError("contact_schedule must be (N,4)")

        for k in range(N):
            solver.cost_set(k, "yref", np.concatenate([z_ref[k], u_ref[k]]))
            solver.set(k, "p", arms_seq[k].reshape(-1))

            if contact_schedule is not None:
                lbu_k = lbu.copy()
                ubu_k = ubu.copy()
                for leg in range(4):
                    if contact_schedule[k, leg] <= 0.5:
                        lbu_k[3 * leg : 3 * leg + 3] = 0.0
                        ubu_k[3 * leg : 3 * leg + 3] = 0.0
                    else:
                        lbu_k[3 * leg + 2] = self.fz_min
                solver.set(k, "lbu", lbu_k)
                solver.set(k, "ubu", ubu_k)
            solver.set(k, "u", u_ref[k])

        solver.cost_set(N, "yref", z_ref[N])
        solver.set(N, "p", arms_seq[N - 1].reshape(-1))

        status = solver.solve()
        if status != 0:
            cs0 = contact_schedule[0] if contact_schedule is not None else "n/a"
            print(
                f"[KoopmanConvexMPC] acados status={status} | cs0={cs0} "
                f"| u_ref0_f={u_ref[0, :12]} | wrench_ref0={u_ref[0, 12:18]} "
                f"| arms0={arms_seq[0]}"
            )
        z_pred = np.vstack([solver.get(i, "x") for i in range(N + 1)])
        u_pred = np.vstack([solver.get(i, "u") for i in range(N)])

        return {"status": status, "Z": z_pred, "U": u_pred}

    # simple interface matching srbd_controller expectations
    def compute_control(self, state_current, ref_state, contact_sequence, **kwargs):
        """
        Close to MIT convex MPC compute_control:
          - build refs from cmd_vxy/cmd_yawrate/cmd_z (or ref_*)
          - integrate yaw/pos; keep v,w constant over horizon
          - split mg across stance legs for u_ref if not provided
          - lift refs to z_ref and solve
        Returns: (grf0 (12,), footholds (4,3), predicted_z, status)
        """
        N = self.N
        dt = self.dt
        nu = self.nu
        nphi = self.nphi

        # Current state
        p0 = np.asarray(state_current.get("position", np.zeros(3)), float).reshape(3)
        v0 = np.asarray(state_current.get("linear_velocity", np.zeros(3)), float).reshape(3)
        w0 = np.asarray(state_current.get("angular_velocity", np.zeros(3)), float).reshape(3)
        theta0 = np.asarray(state_current.get("orientation", np.zeros(3)), float).reshape(3)

        # Commands (like MIT convex)
        if "cmd_vxy" in ref_state:
            vx_cmd, vy_cmd = ref_state["cmd_vxy"]
        else:
            vref_in = np.asarray(ref_state.get("ref_linear_velocity", np.tile(v0, (N, 1)))).reshape(-1, 3)
            vx_cmd, vy_cmd = vref_in[0, 0], vref_in[0, 1]

        if "cmd_yawrate" in ref_state:
            yawrate_cmd = float(ref_state["cmd_yawrate"])
        else:
            wref_in = np.asarray(ref_state.get("ref_angular_velocity", np.tile(w0, (N, 1)))).reshape(-1, 3)
            yawrate_cmd = float(wref_in[0, 2])

        if "cmd_z" in ref_state:
            z_cmd = float(ref_state["cmd_z"])
        else:
            pref_in = np.asarray(ref_state.get("ref_position", np.tile(p0, (N, 1)))).reshape(-1, 3)
            z_cmd = float(pref_in[0, 2])

        # Build reference trajectories (piecewise-constant v,w)
        vref = np.tile(np.array([vx_cmd, vy_cmd, 0.0]), (N, 1))
        wref = np.tile(np.array([0.0, 0.0, yawrate_cmd]), (N, 1))
        pref = np.zeros((N, 3))
        oref = np.zeros((N, 3))
        pref[0] = np.array([p0[0], p0[1], z_cmd])
        oref[0] = np.array([0.0, 0.0, theta0[2]])
        for k in range(N - 1):
            pref[k + 1] = pref[k] + dt * vref[k]
            oref[k + 1] = oref[k] + dt * wref[k]

        # Contact plan: expect same leg order as convex (FL,FR,RL,RR)
        contacts = np.asarray(contact_sequence, float)
        if contacts.shape == (N, 4):
            contacts4xN = contacts.T
        elif contacts.shape == (4, N):
            contacts4xN = contacts
        else:
            contacts4xN = np.ones((4, N), float)
        contacts = contacts4xN.T  # keep (N,4) for downstream

        # Arms (lever) defaults to zeros unless provided
        arms_seq = np.asarray(ref_state.get("ref_arms", np.zeros((N, 4, 3))), float)
        if arms_seq.shape != (N, 4, 3):
            arms_seq = np.zeros((N, 4, 3))

        # Footholds best-effort (match convex MPC keys)
        if all(k in ref_state for k in ["ref_foot_FL", "ref_foot_FR", "ref_foot_RL", "ref_foot_RR"]):
            feet0 = np.stack([
                np.asarray(ref_state["ref_foot_FL"], float).reshape(-1, 3)[0],
                np.asarray(ref_state["ref_foot_FR"], float).reshape(-1, 3)[0],
                np.asarray(ref_state["ref_foot_RL"], float).reshape(-1, 3)[0],
                np.asarray(ref_state["ref_foot_RR"], float).reshape(-1, 3)[0],
            ], axis=0)
        else:
            feet0 = np.asarray(ref_state.get("ref_footholds", np.zeros((4, 3))), float)
            if feet0.shape != (4, 3):
                feet0 = np.zeros((4, 3))
        # If arms missing, derive lever arms from feet and COM refs
        if not np.any(arms_seq):
            com_ref = pref  # use position reference as COM proxy
            for k in range(N):
                for leg in range(4):
                    arms_seq[k, leg, :] = feet0[leg] - com_ref[min(k, pref.shape[0]-1)]

        # u_ref: split mg across stance if not provided; fill wrench to match GRF
        u_ref = np.asarray(ref_state.get("koopman_u_ref", np.zeros((N, nu))), float)
        if u_ref.ndim == 1:
            u_ref = np.tile(u_ref.reshape(1, -1), (N, 1))
        if u_ref.shape != (N, nu):
            u_ref = np.zeros((N, nu))

        if not np.any(u_ref):
            u_ref = np.zeros((N, nu))
            mg = self.mass * abs(self.gvec[2])
            for k in range(N):
                n_stance = int(contacts[k].sum())
                if n_stance > 0:
                    fz = mg / n_stance
                    for leg in range(4):
                        if contacts[k, leg] > 0.5:
                            u_ref[k, 3 * leg : 3 * leg + 3] = [0.0, 0.0, fz]

        # Fill wrench ref to sum GRFs (force + torque)
        for k in range(N):
            f_mat = u_ref[k, :12].reshape(4, 3)
            u_ref[k, 12:15] = f_mat.sum(axis=0)
            arms_k = arms_seq[min(k, arms_seq.shape[0] - 1)]
            torque_k = np.sum(np.cross(arms_k, f_mat), axis=0)
            u_ref[k, 15:18] = torque_k

        # Lift references
        z_ref = np.zeros((N + 1, nphi))
        for k in range(N + 1):
            idx = min(k, N - 1) if N > 0 else 0
            vk = vref[idx] if k < N else vref[-1]
            wk = wref[idx] if k < N else wref[-1]
            pk = pref[idx] if k < N else pref[-1]
            ok = oref[idx] if k < N else oref[-1]
            xk = np.concatenate([pk, ok, vk, wk], axis=0)
            zk = self.lift_fn(xk)
            zk = np.asarray(zk, float).reshape(-1)
            if zk.size != nphi:
                if zk.size == 12 and nphi > 12:
                    tmp = np.zeros(nphi, float)
                    tmp[:12] = zk
                    zk = tmp
                else:
                    raise ValueError(f"lifted ref dim mismatch: got {zk.size}, expected {nphi}")
            z_ref[k] = zk

        # Initial lift
        z0 = self.lift_fn(np.concatenate([p0, theta0, v0, w0], axis=0))
        z0 = np.asarray(z0, float).reshape(-1)
        if z0.size != nphi:
            if z0.size == 12 and nphi > 12:
                tmp = np.zeros(nphi, float)
                tmp[:12] = z0
                z0 = tmp
            else:
                raise ValueError(f"lifted dim mismatch: got {z0.size}, expected {nphi}")

        res = self.solve(z0, z_ref, u_ref, arms_seq, contact_schedule=contacts)
        u_pred = res["U"]
        grf0 = u_pred[0, :12] if u_pred.shape[0] > 0 else np.zeros(12)

        return grf0, feet0, res.get("Z", None), res.get("status", None)
    
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
