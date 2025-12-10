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

# import lifting function from correct path
def _load_lift_1d():
    """
    Try to import simulation.edmd.edmd_runner.lift_1d via sys.path or explicit path.
    Returns None if not found.
    """
    try:
        from simulation.edmd.edmd_runner import lift_1d  # type: ignore
        return lift_1d
    except Exception:
        pass
    try:
        import importlib.util, sys
        _edmd_file = Path(__file__).resolve().parents[3] / "simulation" / "edmd" / "edmd_runner.py"
        _edmd_dir = _edmd_file.parent
        if str(_edmd_dir) not in sys.path:
            sys.path.insert(0, str(_edmd_dir))
        spec = importlib.util.spec_from_file_location("edmd_runner", str(_edmd_file))
        _edmd_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_edmd_mod)  # type: ignore
        return getattr(_edmd_mod, "lift_1d", None)
    except Exception:
        return None

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
        Qp: Sequence[float] | float = (0.0, 0.0, 0.0),
        Qv: Sequence[float] | float = (0.0, 0.0, 0.0),
        QR: Sequence[float] | float | np.ndarray = (
            10.0, 10.0, 0.0,
            10.0, 10.0, 0.0,
            0.0, 0.0, 10.0,
        ),
        Qw: Sequence[float] | float = (0.0, 0.0, 0.0),
        R_grf: Sequence[float] | float = np.diag([1e-6, 1e-6, 5e-4]),
        R_wrench: Sequence[float] | float = (
            1e-2,
            1e-2,
            1e-2,
            1e-2,
            1e-2,
            1e-2,
        ),
        slack_wrench_force: float = 0.0,
        slack_wrench_torque: float = 0.0,
        model_path: Path | str = None,
        lift_fn=None,
    ):  
        # Debug toggle: if False, skip friction/box/wrench constraints
        self._use_constraints = False
        
        # load the EDMD model (A,B,meta) based on wrench inputs
        self._load_model(model_path)
        
        # load basis functions
        self._init_lift(lift_fn)

        # setup params
        self.mass = float(mass)
        self.gvec = np.array([0.0, 0.0, -abs(g)], float)
        self.N = int(N)
        self.dt = float(dt)
        self.mu = float(mu)
        self.fz_min = float(fz_min)
        self.fz_max = float(fz_max)
        self.nphi = self.A.shape[0]
        self.nu_w = self.B.shape[1]  # expected 6
        self.nu_f = 12
        self.nu = self.nu_f + self.nu_w
        self.B_bar = np.hstack([np.zeros((self.nphi, self.nu_f)), self.B])
        
        # Sanity check
        _probe_x = np.zeros(12, float)
        _phi = np.asarray(self.lift_fn(_probe_x)).reshape(-1)
        if _phi.size != self.nphi:
            raise ValueError(
                f"Lift dimension mismatch: lift(x).size={_phi.size} but Koopman A is {self.nphi}x{self.nphi}. "
                f"Check lift_fn / p_max vs. the saved model."
            )
            
        # Build Qz: if Qx provided, use generic builder; else use block weights
        self.Qz = self._build_Qz_from_blocks(Qp, Qv, QR, Qw)
            
        # Default anisotropic GRF weight: light fx, fy; heavier fz per leg
        R_grf = np.kron(np.eye(4), R_grf)
        self.Rf = self._as_mat(R_grf, self.nu_f)
        self.Rw = self._as_mat(R_wrench, self.nu_w)
        
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

    def _init_lift(self, lift_fn):
        """
        Initialize the lift function, falling back to edmd_runner.lift_1d or identity for 12x12 models.
        Performs a sanity check on the output dimension versus Koopman A rows.
        """
        if lift_fn is not None:
            self.lift_fn = lift_fn
        else:
            lift_1d = _load_lift_1d()
            if lift_1d is None:
                raise ImportError(
                    "Could not import simulation.edmd.edmd_runner.lift_1d and no lift_fn was provided. "
                    "Ensure simulation/edmd is on PYTHONPATH or pass lift_fn explicitly."
                )
            if not isinstance(self.meta, dict) or "p_max" not in self.meta:
                raise ValueError("p_max not found in Koopman model meta; cannot build lift_fn.")
            p_max = int(self.meta["p_max"])
            self.lift_fn = lambda x: lift_1d(np.asarray(x, float), p_max)

    def _build_Qz_from_blocks(self, Qp, Qv, QR, Qw) -> np.ndarray:
        """
        Build lifted cost weights for the EDMD basis:
        [pos(3), lin_vel(3), vec(R)(9), w(3), psi_bar(...)]. psi_bar is left unweighted.

        Accepted shapes:
        - Qp, Qv, Qw: scalar or len-3 → diag on their blocks.
        - QR: scalar → scalar*I_9; len-9 → diag on vec(R).
        """
        Qz = np.zeros((self.nphi, self.nphi))

        def as3(val):
            arr = np.asarray(val, float).flatten()
            if arr.size == 1:
                return np.full(3, arr.item(), float)
            if arr.size == 3:
                return arr
            raise ValueError("Expected scalar or len-3 for Qp/Qv/Qw")

        # pos (0:3)
        qp = as3(Qp)
        Qz[0:3, 0:3] = np.diag(qp)

        # lin_vel (3:6)
        qv = as3(Qv)
        Qz[3:6, 3:6] = np.diag(qv)

        # vec(R) (6:15)
        if np.isscalar(QR):
            Qz[6:15, 6:15] = float(QR) * np.eye(9)
        else:
            arrR = np.asarray(QR, float).flatten()
            if arrR.size != 9:
                raise ValueError("QR must be scalar or len-9")
            Qz[6:15, 6:15] = np.diag(arrR)

        # w (15:18)
        qw = as3(Qw)
        Qz[15:18, 15:18] = np.diag(qw)

        return Qz

    def _load_model(self, model_path: Path | str | None):
        if model_path is None:
            # Default to koopman/models/wrench_model.npz
            self.model_path = Path(__file__).resolve().parent / "models" / "wrench_model.npz"
        else:
            self.model_path = Path(model_path)
        data = np.load(self.model_path, allow_pickle=True)
        self.A = np.asarray(data["A"], float)
        self.B = np.asarray(data["B"], float)
        self.meta = data["meta"].item() if "meta" in data.files and np.asarray(data["meta"]).size == 1 else {}
   
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
        if self._use_constraints:
            model.con_h_expr = h_expr
        else:
            # No path constraints when debugging without inequalities
            model.con_h_expr = MX.zeros(0, 1)

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

        if self._use_constraints:
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
        else:
            # No inequality/box constraints for debugging
            ocp.dims.ng = 0
            ocp.constraints.C = np.zeros((0, nx))
            ocp.constraints.D = np.zeros((0, nu))
            ocp.constraints.lg = np.zeros(0)
            ocp.constraints.ug = np.zeros(0)
            ocp.dims.nh = 0
            ocp.dims.nsh = 0
            ocp.constraints.lh = np.zeros(0)
            ocp.constraints.uh = np.zeros(0)
            ocp.constraints.idxsh = np.zeros(0, dtype=np.int32)
            ocp.constraints.lsh = np.zeros(0)
            ocp.constraints.ush = np.zeros(0)
            ocp.cost.zl = np.zeros(0)
            ocp.cost.zu = np.zeros(0)
            ocp.cost.Zl = np.zeros((0, 0))
            ocp.cost.Zu = np.zeros((0, 0))
            ocp.dims.nbu = 0
            ocp.constraints.idxbu = np.zeros(0, dtype=np.int32)
            ocp.constraints.lbu = np.zeros(0)
            ocp.constraints.ubu = np.zeros(0)

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

        if self._use_constraints:
            lbu = -1e8 * np.ones(nu)
            ubu = 1e8 * np.ones(nu)
        else:
            lbu = ubu = None
        if contact_schedule is not None:
            contact_schedule = np.asarray(contact_schedule, float)
            if contact_schedule.shape == (4, N):
                contact_schedule = contact_schedule.T
            if contact_schedule.shape != (N, 4):
                raise ValueError("contact_schedule must be (N,4)")

        for k in range(N):
            solver.cost_set(k, "yref", np.concatenate([z_ref[k], u_ref[k]]))
            solver.set(k, "p", arms_seq[k].reshape(-1))

            if self._use_constraints and contact_schedule is not None:
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

    # utilites
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

        # u_ref: accept 12-D GRF-only refs (convex MPC style) or full 18-D; default to mg split
        u_ref = None
        u_ref_in = ref_state.get("koopman_u_ref", None)
        if u_ref_in is not None:
            u_arr = np.asarray(u_ref_in, float)
            if u_arr.ndim == 1:
                if u_arr.size == 12:
                    u_arr = np.tile(u_arr.reshape(1, 12), (N, 1))
                elif u_arr.size == nu:
                    u_arr = np.tile(u_arr.reshape(1, nu), (N, 1))
            if u_arr.shape == (N, 12):
                u_ref = np.zeros((N, nu), float)
                u_ref[:, :12] = u_arr
            elif u_arr.shape == (N, nu):
                u_ref = u_arr.copy()

        if u_ref is None:
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
