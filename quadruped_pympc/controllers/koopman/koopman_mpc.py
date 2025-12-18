"""
Koopman convex MPC (acados) using lifted linear dynamics learned via EDMD.

Dynamics: z_{k+1} = A z_k + B_wrench * (H(arms) * u_grf),
where u_grf ∈ R^12 are per-leg GRFs, and H maps GRFs to net wrench about the COM.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence, Dict, Any

import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
from casadi import MX, vertcat, horzcat
from scipy.spatial.transform import Rotation as R

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

def _load_decode_geom():
    """
    Try to import simulation.edmd.edmd_runner.decode_state_from_geom_phi via sys.path or explicit path.
    Returns None if not found.
    """
    try:
        from simulation.edmd.edmd_runner import decode_state_from_geom_phi  # type: ignore
        return decode_state_from_geom_phi
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
        return getattr(_edmd_mod, "decode_state_from_geom_phi", None)
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
        fz_max: float = 200.0,
        Qp: Sequence[float] | float = (1e2, 1e2, 1e3),
        Qv: Sequence[float] | float = (1e3, 1e2, 1e2),
        QR: Sequence[float] | float | np.ndarray = (
            1e2, 1e2, 0.0,
            1e2, 1e2, 0.0,
            0.0, 0.0, 1e2,
        ),
        Qw: Sequence[float] | float = (1e2, 1e2, 1e2),
        R_grf: Sequence[float] | float = np.diag([1e-2, 1e-2, 1e-2]),
        model_path: Path | str = None,
        lift_fn=None,
        debug: bool = False,
        use_constraints: bool = True,
        use_simple_dynamics: bool = False,
    ):  
        # Toggle inequality/box constraints; keep only dynamics if False
        self._use_constraints = bool(use_constraints)
        self.debug = bool(debug)
        self.use_simple_dynamics = bool(use_simple_dynamics)
        
        # load the EDMD model (A,B,meta) based on wrench inputs
        self._load_model(model_path)
        
        # load basis functions
        self._init_lift(lift_fn)
        self._decode_fn = _load_decode_geom()

        # setup params
        self.mass = float(mass)
        self.gvec = np.array([0.0, 0.0, -abs(g)], float)
        self.N = int(N)
        self.dt = float(dt)
        self.mu = float(mu)
        self.fz_min = float(fz_min)
        self.fz_max = float(fz_max)
        self.nphi = self.A.shape[0]
        # GRF-only decision variables; wrench comes from H(arms) @ GRFs inside dynamics
        self.nu = 12
        self.nu_f = self.nu
        self.B_wrench = np.array(self.B, float)  # (nphi, 6) wrench-driven EDMD input map
        # Keep last command for slew limiting inside compute_control
        self._cmd_prev = {"vx": 0.0, "vy": 0.0, "yr": 0.0}
        
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
        - Qp, Qv, Qw: scalar or len-3 -> diag on their blocks.
        - QR: scalar -> scalar*I_9; len-9 -> diag on vec(R).
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

    def _decode_z_to_x(self, Z: np.ndarray) -> np.ndarray:
        """
        Decode lifted states Z to base states x (12D).
        Falls back to the first 12 components if no decoder/p_max is available.
        """
        Z = np.asarray(Z, float)
        if Z.ndim == 1:
            Z = Z.reshape(1, -1)
        p_max = None
        if isinstance(self.meta, dict):
            p_max = self.meta.get("p_max", None)
        if Z.shape[1] < 18:
            raise ValueError(f"Cannot decode lifted state with dim {Z.shape[1]} to 12D base state.")
        pos = Z[:, 0:3]
        vlin = Z[:, 3:6]
        Rvec = Z[:, 6:15]
        omega = Z[:, 15:18]
        Rmat = Rvec.reshape(Z.shape[0], 3, 3)
        eul = R.from_matrix(Rmat).as_euler("xyz", degrees=False)
        x = np.zeros((Z.shape[0], 12), float)
        x[:, 0:3] = pos
        x[:, 3:6] = eul
        x[:, 6:9] = vlin
        x[:, 9:12] = omega
        return x

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
        nz = self.nphi
        nu = self.nu

        # symbols
        z = MX.sym("z", nz)
        u = MX.sym("u", nu)  # GRFs (12)
        arms = MX.sym("arms", 12)  # lever arms per stage (flattened)

        # Build H(arms) so wrench = H @ u_f
        def _skew_from_arm(ax, ay, az):
            return vertcat(
                horzcat(MX(0),    az,   -ay),
                horzcat(   -az, MX(0),   ax),
                horzcat(    ay,   -ax, MX(0)),
            )

        force_block = horzcat(MX.eye(3), MX.eye(3), MX.eye(3), MX.eye(3))  # (3,12)
        mom_blocks = []
        for i in range(4):
            a = arms[3 * i : 3 * i + 3]
            mom_blocks.append(_skew_from_arm(a[0], a[1], a[2]))
        H_map = vertcat(force_block, horzcat(*mom_blocks))  # (6,12)
        wrench = H_map @ u  # (6,)

        # Dynamics: Koopman with wrench-driven B; or simple dynamics if debugging
        if self.use_simple_dynamics:
            A_mx = MX.eye(nz)
            B_mx = MX.ones(nz, wrench.size1())
        else:
            A_mx = MX(self.A)
            B_mx = MX(self.B_wrench)
        z_next = A_mx @ z + B_mx @ wrench

        model = AcadosModel()
        # include nonce to avoid reusing stale codegen when constraint settings change
        model.name = f"koopman_mpc_forces"
        model.x = z
        model.u = u
        model.p = arms
        model.disc_dyn_expr = z_next

        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.nx = nz
        ocp.dims.nu = nu
        ocp.dims.np = 12
        ocp.dims.N = self.N
        ocp.parameter_values = np.zeros(12)

        ny = nz + nu
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"
        W = np.zeros((ny, ny))
        W[:nz, :nz] = self.Qz
        W[nz:nz + self.nu_f, nz:nz + self.nu_f] = self.Rf
        ocp.cost.W = W
        ocp.cost.W_e = self.Qz

        Vx = np.zeros((ny, nz))
        Vu = np.zeros((ny, nu))
        Vx[:nz, :nz] = np.eye(nz)
        Vu[nz:, :] = np.eye(nu)
        ocp.cost.Vx = Vx
        ocp.cost.Vu = Vu
        ocp.cost.Vx_e = np.eye(nz)
        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(nz)

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
            ocp.constraints.C = np.zeros((ocp.dims.ng, nz))
            ocp.constraints.D = A_ineq
            ocp.constraints.lg = np.asarray(lg, float)
            ocp.constraints.ug = np.asarray(ug, float)

            ocp.dims.nh = 0
            ocp.dims.nsh = 0
            ocp.constraints.lh = np.zeros(0)
            ocp.constraints.uh = np.zeros(0)
            ocp.constraints.idxsh = np.zeros(0, dtype=np.int32)
            ocp.constraints.lsh = np.zeros(0)
            ocp.constraints.ush = np.zeros(0)
            ocp.cost.Zl = np.zeros((0, 0))
            ocp.cost.Zu = np.zeros((0, 0))
            ocp.cost.zl = np.zeros(0)
            ocp.cost.zu = np.zeros(0)

            ocp.dims.nbu = nu
            ocp.constraints.idxbu = np.arange(nu, dtype=np.int32)
            ocp.constraints.lbu = -1e8 * np.ones(nu)
            ocp.constraints.ubu = 1e8 * np.ones(nu)
        else:
            # No inequality/box constraints for debugging
            ocp.dims.ng = 0
            ocp.constraints.C = np.zeros((0, nz))
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

        ocp.dims.nbx_0 = nz
        ocp.constraints.idxbx_0 = np.arange(nz, dtype=np.int32)
        ocp.constraints.lbx_0 = np.zeros(nz)
        ocp.constraints.ubx_0 = np.zeros(nz)

        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.tf = self.N * self.dt

        # Export generated code locally (matches other controllers)
        code_export_dir = Path(__file__).parent / "c_generated_code"
        code_export_dir.mkdir(parents=True, exist_ok=True)
        ocp.code_export_directory = str(code_export_dir)

        # Keep a handle to the OCP for debugging/inspection
        self.ocp = ocp

        # Force regenerate/build to ensure constraints/slacks reflect latest settings
        json_path = code_export_dir / f"{model.name}.json"
        return AcadosOcpSolver(ocp, json_file=str(json_path), generate=True, build=True)

    def solve(
        self,
        z0: np.ndarray,
        z_ref: np.ndarray,
        u_ref: np.ndarray,
        arms_seq: np.ndarray,
        contact_schedule: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        N = self.N
        nz = self.nphi
        nu = self.nu

        z0 = np.asarray(z0, float).reshape(-1)
        z_ref = np.asarray(z_ref, float)
        u_ref = np.asarray(u_ref, float)
        arms_seq = np.asarray(arms_seq, float)

        if self.debug:
            fmt = lambda arr: np.array2string(np.asarray(arr, float), precision=3, suppress_small=True)
            print("[Koopman MPC] solve() called")
            print(f"  z0 shape={z0.shape}, z_ref shape={z_ref.shape}, u_ref shape={u_ref.shape}")
            print(f"  arms_seq shape={arms_seq.shape}, contact_sched shape={None if contact_schedule is None else contact_schedule.shape}")
            # Sanity check on slack dimensions in the built OCP
            try:
                print(f"  ocp.nh={self.ocp.dims.nh}, ocp.nsh={self.ocp.dims.nsh}, idxsh={self.ocp.constraints.idxsh}")
            except Exception:
                print("  could not read ocp slack dims/idx")

        if z_ref.shape != (N + 1, nz):
            raise ValueError(f"z_ref must be (N+1,{nz})")
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
                if self.debug and k == 0:
                    print(f"[Koopman MPC] stage {k} contact_sched={fmt(contact_schedule[k])} -> lbu[0:12]={fmt(lbu_k[:12])}")
            solver.set(k, "u", u_ref[k])

        solver.cost_set(N, "yref", z_ref[N])
        solver.set(N, "p", arms_seq[N - 1].reshape(-1))

        status = solver.solve()
        if self.debug:
            if status == 0:
                print("[Koopman MPC] solver converged (status=0)")
            else:
                print(f"[Koopman MPC] solver failed (status={status}); outputs may be invalid")
            try:
                cost_val = solver.get_cost() if hasattr(solver, "get_cost") else solver.get(0, "cost_value")
                print(f"[Koopman MPC] cost={float(cost_val):.3g}")
            except Exception:
                print("[Koopman MPC] cost unavailable")
            try:
                stats = solver.get_stats()
                if isinstance(stats, dict):
                    print(f"[Koopman MPC] stats keys={list(stats.keys())}")
                    if "time_tot" in stats:
                        print(f"[Koopman MPC] time_tot={stats['time_tot']:.3g}")
            except Exception:
                pass
            try:
                u0 = np.asarray(solver.get(0, "u"), float).reshape(-1)
                f0 = u0.reshape(4, 3)
                print(f"[Koopman MPC] control k=0 GRFs={fmt(f0)}")
            except Exception:
                print("[Koopman MPC] could not retrieve control at k=0")
            # Dynamics residual check at k=0: x1 - (A x0 + B_wrench H u0)
            try:
                x0 = np.asarray(solver.get(0, "x"), float).reshape(-1)
                x1 = np.asarray(solver.get(1, "x"), float).reshape(-1)
                u0 = np.asarray(solver.get(0, "u"), float).reshape(-1)
                # rebuild H for stage 0 using params
                arms0 = np.asarray(solver.get(0, "p"), float).reshape(4, 3)
                def _skew_np(a):
                    return np.array([[0,    a[2], -a[1]],
                                     [-a[2], 0,    a[0]],
                                     [a[1], -a[0], 0   ]], float)
                H_force = np.hstack([np.eye(3), np.eye(3), np.eye(3), np.eye(3)])
                H_mom = np.hstack([_skew_np(a) for a in arms0])
                H = np.vstack([H_force, H_mom])
                wrench0 = H @ u0
                pred = (self.A @ x0) + (self.B_wrench @ wrench0)
                resid = x1 - pred
                if resid.size >= 18:
                    r_pos = resid[0:3]
                    r_v = resid[3:6]
                    r_R = resid[6:15]
                    r_w = resid[15:18]
                    r_psi = resid[18:] if resid.size > 18 else np.zeros(0)
                    print(f"  resid pos={fmt(r_pos)} v={fmt(r_v)} Rvec={fmt(r_R)} w={fmt(r_w)}")
                    if r_psi.size > 0:
                        print(f"  resid psi_bar (rest)={fmt(r_psi)}")
            except Exception:
                print("[Koopman MPC] could not compute dynamics residual at k=0")
        if status != 0:
            cs0 = contact_schedule[0] if contact_schedule is not None else "n/a"
            print(
                f"[KoopmanConvexMPC] acados status={status} | cs0={cs0} "
                f"| u_ref0_f={u_ref[0, :12]} "
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
            fz = self.mass * abs(self.gvec[2]) / n_stance
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

        # Footholds best-effort (match convex MPC keys)
        if all(k in ref_state for k in ["ref_foot_FL", "ref_foot_FR", "ref_foot_RL", "ref_foot_RR"]):
            feet0 = np.stack([
                np.asarray(ref_state["ref_foot_FL"], float).reshape(-1, 3)[0],
                np.asarray(ref_state["ref_foot_FR"], float).reshape(-1, 3)[0],
                np.asarray(ref_state["ref_foot_RL"], float).reshape(-1, 3)[0],
                np.asarray(ref_state["ref_foot_RR"], float).reshape(-1, 3)[0],
            ], axis=0)
            feet_over_h = self._feet_refs_over_horizon(ref_state, contacts4xN)
        else:
            feet0 = np.asarray(ref_state.get("ref_footholds", np.zeros((4, 3))), float)
            if feet0.shape != (4, 3):
                feet0 = np.zeros((4, 3))
            feet_over_h = np.tile(feet0[None, :, :], (N, 1, 1))

        # Arms (lever) from feet over horizon or provided override
        arms_seq = np.asarray(ref_state.get("ref_arms", np.zeros((N, 4, 3))), float)
        if arms_seq.shape != (N, 4, 3) or not np.any(arms_seq):
            com_ref = pref  # use position reference as COM proxy
            arms_seq = np.zeros((N, 4, 3))
            for k in range(N):
                for leg in range(4):
                    arms_seq[k, leg, :] = feet_over_h[min(k, feet_over_h.shape[0]-1), leg, :] - com_ref[min(k, pref.shape[0]-1)]

        # u_ref: GRFs only (12), default to mg split across stance
        u_ref = np.zeros((N, nu))
        mg = self.mass * abs(self.gvec[2])
        for k in range(N):
            n_stance = int(contacts[k].sum())
            if n_stance > 0:
                fz = mg / n_stance
                for leg in range(4):
                    if contacts[k, leg] > 0.5:
                        u_ref[k, 3 * leg : 3 * leg + 3] = [0.0, 0.0, fz]

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

        res = self.solve(z0, z_ref, u_ref, arms_seq, contact_schedule=contacts)
        status = res.get("status", None)
        if status is None or status != 0:
            if self.debug:
                print(f"[Koopman MPC] solver failed with status {status}, returning zeros")
            return np.zeros(12), feet0, np.zeros((N + 1, 12)), status

        u_pred = res["U"]
        grf0 = u_pred[0, :12] if u_pred.shape[0] > 0 else np.zeros(12)
        z_pred = res.get("Z", None)
        x_pred = self._decode_z_to_x(z_pred) if z_pred is not None else np.zeros((N + 1, 12))

        if self.debug:
            print(f"[Koopman MPC] return grf0={grf0}")

        return grf0, feet0, x_pred, status
    
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
