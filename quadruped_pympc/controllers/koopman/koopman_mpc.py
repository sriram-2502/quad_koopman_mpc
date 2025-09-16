# quadruped_pympc/controllers/koopman/koopman_mpc.py
import numpy as np
import scipy.sparse as sp
from typing import Optional, Tuple, Union

from quadruped_pympc.config import mpc_params, mass, gravity_constant

# Optional dynamic import for your EDMD lift
import importlib.util, sys
from pathlib import Path
_edmd_file = Path(__file__).resolve().parents[3] / "simulation" / "edmd" / "edmd_runner.py"
_edmd_dir = _edmd_file.parent
if str(_edmd_dir) not in sys.path:
    sys.path.insert(0, str(_edmd_dir))
_spec = importlib.util.spec_from_file_location("edmd_runner", str(_edmd_file))
_edmd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_edmd)
_default_lift = getattr(_edmd, "lift_1d")


class Koopman_MPC:
    """
    Linear Koopman MPC:
        z_{k+1} = A z_k + B u_k,  x_k = C z_k
    Cost:
        sum (C z_k - x_ref_k)^T Q (C z_k - x_ref_k)
          + sum (u_k - u_ref_k)^T R (u_k - u_ref_k)
    Constraints per step:
      - dynamics equality
      - GRF bounds (per-component)
      - friction pyramid: ±fx <= mu fz, ±fy <= mu fz
      - optional contact schedule: swing leg -> u_leg = 0
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        horizon: int = 12,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        lift_fn=None,
        use_friction_pyramid: bool = True,
        solver_dense: bool = True,     # kept for compatibility
        osqp_opts: Optional[dict] = None,
    ):
        self.A = np.asarray(A)
        self.B = np.asarray(B)
        self.C = np.asarray(C)

        self.N = int(horizon)
        self.nz = self.A.shape[0]
        self.nu = self.B.shape[1]     # expected 12 = 4 legs * 3
        self.nx = self.C.shape[0]
        self.nlegs = self.nu // 3
        assert self.nu == 3 * self.nlegs, "u must stack [fx,fy,fz] per leg."

        self.Q = np.eye(self.nx) if Q is None else np.asarray(Q)
        self.R = 0.1 * np.eye(self.nu) if R is None else np.asarray(R)
        self.Q_lift = self.C.T @ self.Q @ self.C

        self.mass = float(mass)
        self.gravity = float(gravity_constant)
        self.mu = float(mpc_params["mu"])

        # Parse GRF bounds into full 12-vectors per step
        self.grf_min_1step, self.grf_max_1step = self._parse_grf_bounds(
            mpc_params["grf_min"], mpc_params["grf_max"]
        )

        self.use_friction = bool(use_friction_pyramid)
        self.lift_fn = _default_lift if lift_fn is None else lift_fn

        # Build QP matrices (H, A_all) and base bounds (no z0/contact yet)
        self._build_qp_backend_mats()

        # Native OSQP backend (simple & robust)
        self._init_native_osqp(osqp_opts)
        self.backend = "native_osqp"
        self._last_x = None  # warm start
        self._last_u_ref = None  # for debugging

    # --------------------------- public API ---------------------------
    def solve(
        self,
        x0: np.ndarray,
        x_ref: Union[np.ndarray, Tuple[np.ndarray, ...]],
        contact_schedule: Optional[np.ndarray] = None,
        u_ref: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Args:
            x0:  (nx,)
            x_ref: (nx,) or (N+1, nx)
            contact_schedule: (N, nlegs) with 1=stance, 0=swing
            u_ref: (N, nu) baseline GRFs for input cost (optional)
        Returns:
            u0: (nu,) first control
            U:  (N,nu) planned forces
            X:  (N+1,nx) predicted original states
        """
        N, nz, nu, nx = self.N, self.nz, self.nu, self.nx
        n_xblk = nz * (N + 1)

        # Lift initial and reference(s)
        z0 = self._lift1(x0)
        if x_ref.ndim == 1:
            zref_all = np.tile(self._lift1(x_ref), (N + 1, 1))
        else:
            assert x_ref.shape == (N + 1, nx), f"x_ref must be (N+1,{nx})"
            zref_all = np.vstack([self._lift1(x_ref[k]) for k in range(N + 1)])

        # Linear term g
        g = np.zeros(self.n_tot)
        #  (i) state tracking part
        for k in range(N + 1):
            g[k * nz : (k + 1) * nz] = (-2.0) * (self.Q_lift @ zref_all[k])

        #  (ii) input baseline part: penalize (u - u_ref) with R
        if u_ref is None:
            Uref = self._compute_u_ref(contact_schedule if contact_schedule is not None
                                       else np.ones((self.N, self.nlegs), int))
        else:
            Uref = np.asarray(u_ref)
            assert Uref.shape == (self.N, self.nu), f"u_ref must be (N, {self.nu})"
        for k in range(N):
            g[n_xblk + k*nu : n_xblk + (k+1)*nu] = -2.0 * (self.R @ Uref[k])
        self._last_u_ref = Uref  # save for debugging

        # Variable bounds: pin z0
        lbx = self.lbx_template.copy()
        ubx = self.ubx_template.copy()
        lbx[:nz] = z0
        ubx[:nz] = z0

        # Constraint bounds, possibly tightened by contact schedule
        lba = self.lba_base.copy()
        uba = self.uba_base.copy()
        if contact_schedule is not None:
            assert contact_schedule.shape == (N, self.nlegs)
            # A_u identity bounds live at the end of [dyn, cone, u]
            n_dyn = N * nz
            n_cone = self._n_cone_rows()
            base_u = n_dyn + n_cone
            for k in range(N):
                for j in range(self.nlegs):
                    if contact_schedule[k, j] == 0:
                        row0 = base_u + k * nu + 3 * j
                        lba[row0 : row0 + 3] = 0.0
                        uba[row0 : row0 + 3] = 0.0

        z = self._solve_native_osqp(g, lba, uba, lbx, ubx)

        Z = z[:n_xblk].reshape(N + 1, nz)
        U = z[n_xblk:].reshape(N, nu)
        X = (self.C @ Z.T).T

        return U[0], U, X

    # ------------------------- internals: math -------------------------
    def _parse_grf_bounds(self, grf_min, grf_max):
        nu = self.nu
        nlegs = self.nlegs
        fx_idx = np.arange(0, nu, 3)
        fy_idx = np.arange(1, nu, 3)
        fz_idx = np.arange(2, nu, 3)

        gmin = np.array(grf_min, dtype=float).ravel()
        gmax = np.array(grf_max, dtype=float).ravel()

        lb = -np.inf * np.ones(nu)
        ub = np.inf * np.ones(nu)

        if gmin.size == 1 and gmax.size == 1:
            lb[fz_idx] = max(0.0, float(gmin.item()))
            ub[fz_idx] = float(gmax.item())
            return lb, ub

        if gmin.size == 3 and gmax.size == 3:
            lb_leg = gmin
            ub_leg = gmax
            lb = np.tile(lb_leg, nlegs)
            ub = np.tile(ub_leg, nlegs)
            return lb, ub

        if gmin.size == nu and gmax.size == nu:
            return gmin, gmax

        raise ValueError(
            f"Unsupported GRF bounds: grf_min={gmin.shape}, grf_max={gmax.shape}. "
            f"Expected scalar, (3,), or ({nu},)."
        )

    def _n_cone_rows(self) -> int:
        if not self.use_friction:
            return 0
        return 4 * self.nlegs * self.N

    def _build_qp_backend_mats(self):
        """Build constant QP matrices: H, A_all, and base bounds."""
        N, nz, nu = self.N, self.nz, self.nu
        n_xblk = nz * (N + 1)
        n_ublk = nu * N
        self.n_tot = n_xblk + n_ublk

        # Hessian (0.5 x^T H x + g^T x). Put factor 2 on stage blocks.
        Hx = sp.block_diag([2 * self.Q_lift] * (N + 1), format="csc")
        Hu = sp.block_diag([2 * self.R] * N, format="csc")
        self.H = sp.block_diag((Hx, Hu), format="csc")

        # Dynamics equalities: z_{k+1} - A z_k - B u_k = 0
        A_dyn = sp.lil_matrix((nz * N, self.n_tot))
        for k in range(N):
            A_dyn[k * nz : (k + 1) * nz, (k + 1) * nz : (k + 2) * nz] = sp.eye(nz)
            A_dyn[k * nz : (k + 1) * nz, k * nz : (k + 1) * nz] = -self.A
            A_dyn[k * nz : (k + 1) * nz, n_xblk + k * nu : n_xblk + (k + 1) * nu] = -self.B
        lba_dyn = np.zeros(nz * N)
        uba_dyn = np.zeros(nz * N)

        # Friction pyramid
        if self.use_friction:
            rows_per_leg = 4
            n_cone = rows_per_leg * self.nlegs * N
            A_cone = sp.lil_matrix((n_cone, self.n_tot))
            lba_cone = -np.inf * np.ones(n_cone)
            uba_cone = np.zeros(n_cone)
            r = 0
            for k in range(N):
                uoff = n_xblk + k * self.nu
                for j in range(self.nlegs):
                    c = uoff + 3 * j
                    # +fx - mu fz <= 0
                    A_cone[r, c + 0] =  1.0; A_cone[r, c + 2] = -self.mu; r += 1
                    # -fx - mu fz <= 0
                    A_cone[r, c + 0] = -1.0; A_cone[r, c + 2] = -self.mu; r += 1
                    # +fy - mu fz <= 0
                    A_cone[r, c + 1] =  1.0; A_cone[r, c + 2] = -self.mu; r += 1
                    # -fy - mu fz <= 0
                    A_cone[r, c + 1] = -1.0; A_cone[r, c + 2] = -self.mu; r += 1
            assert r == n_cone
        else:
            A_cone = sp.csc_matrix((0, self.n_tot))
            lba_cone = np.zeros(0)
            uba_cone = np.zeros(0)

        # GRF box bounds via identity on u
        A_u = sp.lil_matrix((self.nu * N, self.n_tot))
        for k in range(N):
            A_u[k * self.nu : (k + 1) * self.nu, n_xblk + k * self.nu : n_xblk + (k + 1) * self.nu] = sp.eye(self.nu)
        lba_u = np.tile(self.grf_min_1step, N)
        uba_u = np.tile(self.grf_max_1step, N)

        # Stack constraints
        self.A_all = sp.vstack([A_dyn.tocsc(), A_cone.tocsc(), A_u.tocsc()], format="csc")
        self.lba_base = np.concatenate([lba_dyn, lba_cone, lba_u])
        self.uba_base = np.concatenate([uba_dyn, uba_cone, uba_u])

        # Variable-bounds templates (we fold into constraints for native OSQP)
        self.lbx_template = -np.inf * np.ones(self.n_tot)
        self.ubx_template = np.inf * np.ones(self.n_tot)

    # --------------------- Native OSQP backend ---------------------
    def _init_native_osqp(self, osqp_opts: Optional[dict]):
        import osqp

        # Fold variable bounds into constraints once: [A_all; I] x in [lba; lbx], [uba; ubx]
        I = sp.eye(self.n_tot, format="csc")
        self.A_osqp = sp.vstack([self.A_all, I], format="csc")

        # Initial placeholders (will be updated at solve time)
        l0 = np.concatenate([self.lba_base, self.lbx_template])
        u0 = np.concatenate([self.uba_base, self.ubx_template])

        # OSQP setup
        self.osqp = osqp.OSQP()
        settings = dict(
            eps_abs=1e-5,
            eps_rel=1e-5,
            max_iter=20000,
            verbose=False,
            warm_start=True,
        )
        if osqp_opts and isinstance(osqp_opts, dict):
            settings.update(osqp_opts)

        # P must be (symmetric) csc
        P = (self.H + self.H.T) * 0.5  # ensure symmetry
        self.osqp.setup(
            P=P,
            q=np.zeros(self.n_tot),
            A=self.A_osqp,
            l=l0,
            u=u0,
            **settings,
        )

    def _solve_native_osqp(self, g, lba, uba, lbx, ubx):
        # Update bounds and linear term
        l = np.concatenate([lba, lbx])
        u = np.concatenate([uba, ubx])

        # Warm-start if available
        if self._last_x is not None and self._last_x.size == self.n_tot:
            try:
                self.osqp.warm_start(x=self._last_x)
            except Exception:
                pass

        self.osqp.update(q=g, l=l, u=u)
        res = self.osqp.solve()
        if res.info.status_val not in (1, 2):  # 1=solved, 2=solved inaccurate
            raise RuntimeError(f"Koopman MPC QP failed (OSQP): {res.info.status}")
        xvec = res.x.copy()
        self._last_x = xvec
        return xvec

    # --------------------------- baseline u_ref ---------------------------
    def _compute_u_ref(self, contact_schedule: Optional[np.ndarray]) -> np.ndarray:
        """
        Baseline GRFs: distribute mg across stance legs at each k.
        Returns Uref with shape (N, nu), stacked [FL,FR,RL,RR] x [fx,fy,fz].
        """
        if contact_schedule is None:
            cs = np.ones((self.N, self.nlegs), int)
        else:
            cs = np.asarray(contact_schedule, dtype=int)
            assert cs.shape == (self.N, self.nlegs)

        Uref = np.zeros((self.N, self.nu))
        mg = self.mass * self.gravity

        for k in range(self.N):
            nstance = int(cs[k].sum())
            if nstance <= 0:  # flight
                continue
            fz_each = mg / nstance
            for j in range(self.nlegs):
                if cs[k, j] == 1:
                    base = 3 * j
                    Uref[k, base + 2] = fz_each  # fx_ref=0, fy_ref=0, fz_ref=fz_each
        return Uref

    # --------------------------- lifting ---------------------------
    def _lift1(self, x: np.ndarray) -> np.ndarray:
        """
        Always return a 1-D numpy float vector of length nz.
        Handles numpy arrays, CasADi DM, and tuple/list outputs robustly.
        """
        x = np.asarray(x, dtype=float).reshape(1, -1)
        phi = self.lift_fn(x)

        # If the lift returns (phi, *rest), keep the first
        if isinstance(phi, (tuple, list)):
            phi = phi[0]

        # Convert to numpy
        try:
            import casadi as cs  # optional, for DM support
            if isinstance(phi, cs.DM):
                phi = np.array(phi)
        except Exception:
            pass

        phi = np.asarray(phi, dtype=float)

        if phi.ndim == 1:
            return phi.reshape(-1)
        if phi.ndim == 2:
            if phi.shape[0] == 1:
                return phi.reshape(-1)
            return phi[0].reshape(-1)
        return np.ravel(phi)
