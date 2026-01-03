"""
Error-based Koopman MPC (acados) using linear DMD dynamics on velocity error.

Dynamics: e_{k+1} = A e_k + B_wrench * (H(arms) * u_grf),
where e = [v_W, w_B] - [v_ref, w_ref], u_grf ∈ R^12 are per-leg GRFs,
and H maps GRFs to net wrench about the COM.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence, Dict, Any

import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
from casadi import MX, vertcat, horzcat
from quadruped_pympc import config as cfg

class KoopmanErrorMPC:
    """
    acados-based convex MPC with Koopman lifted dynamics.

    Parameters
    ----------
    model_path : Path | str
        npz with A,B from error-state DMD (wrench-driven).
    N : int
        Horizon length.
    dt : float
        Sample time.
    mu : float
        Friction coefficient.
    fz_min, fz_max : float
        Normal force bounds for stance legs.
    Qv, Qw : float or sequence
        Weights on linear/angular velocity error (first 6 states).
    R_grf : float or matrix (12x12)
        GRF weight.
    lift_fn : callable | None
        Function x->phi(x); defaults to identity for error-state models.
    """

    def __init__(
        self,
        mass: float,
        inertia: np.ndarray,
        N: int,
        dt: float,
        g: float = 9.81,
        mu: float = 0.8,
        fz_min: float = 0.0,
        fz_max: float = 200.0,
        Qv: Sequence[float] | float = (1e3, 1e3, 1e3),
        Qw: Sequence[float] | float = (1e2, 1e2, 1e3),
        R_grf: Sequence[float] | float = np.diag([1e-2, 1e-2, 1e-2]),
        model_path: Path | str = None,
        lift_fn=None,
        debug: bool = False,
        use_constraints: bool = True,
    ):  
        # Toggle inequality/box constraints; keep only dynamics if False
        self._use_constraints = bool(use_constraints)
        self.debug = bool(debug)
        
        # load the DMD model (A,B) based on wrench inputs (error-state)
        self._load_model(model_path)
        self.nphi = self.A.shape[0]

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
        # GRF-only decision variables; wrench comes from H(arms) @ GRFs inside dynamics
        self.nu = 12
        self.nu_f = self.nu
        self.B_wrench = np.array(self.B, float)  # (nphi, 6) wrench-driven EDMD input map
        # Sanity check
        _probe_x = np.zeros(6, float)
        _phi = np.asarray(self.lift_fn(_probe_x)).reshape(-1)
        if _phi.size != self.nphi:
            raise ValueError(
                f"Lift dimension mismatch: lift(x).size={_phi.size} but Koopman A is {self.nphi}x{self.nphi}. "
                f"Check lift_fn vs. the saved model."
            )
            
        # Build Qz for error state (v,w); remaining lifted dims are unweighted
        self.Qz = self._build_Qe(Qv, Qw)
            
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
        Initialize the lift function. Defaults to identity for error-state DMD models.
        """
        if lift_fn is None:
            self.lift_fn = lambda x: np.asarray(x, float).reshape(-1)
        else:
            self.lift_fn = lift_fn

    def _build_Qe(self, Qv, Qw) -> np.ndarray:
        """Build error-state cost for e = [v_W(3), w_B(3)]."""
        def as3(val):
            arr = np.asarray(val, float).flatten()
            if arr.size == 1:
                return np.full(3, arr.item(), float)
            if arr.size == 3:
                return arr
            raise ValueError("Expected scalar or len-3 for Qv/Qw")

        Qz = np.zeros((self.nphi, self.nphi))
        qv = as3(Qv)
        qw = as3(Qw)
        Qz[0:3, 0:3] = np.diag(qv)
        Qz[3:6, 3:6] = np.diag(qw)
        return Qz

    def _load_model(self, model_path: Path | str | None):
        if model_path is None:
            # Default to error-state DMD model
            self.model_path = Path(__file__).resolve().parent / "models" / "dmd_wrench_errorstate.npz"
        else:
            self.model_path = Path(model_path)
        data = np.load(self.model_path, allow_pickle=True)
        self.A = np.asarray(data["A"], float)
        self.B = np.asarray(data["B"], float)
   
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

        # Dynamics: Koopman with wrench-driven B
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
        if status != 0:
            cs0 = contact_schedule[0] if contact_schedule is not None else "n/a"
            print(
                f"[KoopmanErrorMPC] acados status={status} | cs0={cs0} "
                f"| u_ref0_f={u_ref[0, :12]} "
                f"| arms0={arms_seq[0]}"
            )
        z_pred = np.vstack([solver.get(i, "x") for i in range(N + 1)])
        u_pred = np.vstack([solver.get(i, "u") for i in range(N)])

        return {"status": status, "Z": z_pred, "U": u_pred}

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

        # Always hold vertical command at nominal hip height from config
        z_cmd = float(cfg.hip_height)

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

        # Error-state references (target is zero error)
        z_ref = np.zeros((N + 1, nphi))

        # Initial error state e0 = [v_W, w_B] - [v_ref, w_ref]
        e0 = np.concatenate([v0 - vref[0], w0 - wref[0]], axis=0)
        z0 = self.lift_fn(e0)
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
        if z_pred is not None:
            e_pred = np.asarray(z_pred, float).reshape(-1, nphi)
            vref_all = np.vstack([vref, vref[-1]])
            wref_all = np.vstack([wref, wref[-1]])
            pref_all = np.vstack([pref, pref[-1]])
            oref_all = np.vstack([oref, oref[-1]])
            x_pred = np.zeros((N + 1, 12), float)
            x_pred[:, 0:3] = pref_all
            x_pred[:, 3:6] = oref_all
            x_pred[:, 6:9] = vref_all + e_pred[:, 0:3]
            x_pred[:, 9:12] = wref_all + e_pred[:, 3:6]
        else:
            x_pred = np.zeros((N + 1, 12))

        if self.debug:
            print(f"[Koopman MPC] return grf0={grf0}")

        return grf0, feet0, x_pred, status
    
    def reset(self):
        """
        Clear warm starts and internal anchors between episodes.
        Safe to call before the first compute_control() of a new rollout.
        """
        # optional: clear acados warm-starts (ignore if solver not yet fully built)
        try:
            nu, N = 12, self.N
            zero_x = np.zeros(self.nphi, dtype=float)
            zero_u = np.zeros(nu, dtype=float)
            for k in range(N):
                self.solver.set(k, "u", zero_u)
            for k in range(N + 1):
                self.solver.set(k, "x", zero_x)
        except Exception:
            pass
