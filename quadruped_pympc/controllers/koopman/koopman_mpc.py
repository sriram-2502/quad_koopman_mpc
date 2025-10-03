# quadruped_pympc/controllers/koopman/koopman_mpc.py
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from typing import Optional, Tuple, Union

from quadruped_pympc.config import mpc_params, mass, gravity_constant

# -------------------------------------------------------------------------
# Import your EDMD lift (lift_1d) and pass p_max correctly
# -------------------------------------------------------------------------
import importlib.util, sys
from pathlib import Path

_edmd_file = Path(__file__).resolve().parents[3] / "simulation" / "edmd" / "edmd_runner.py"
_edmd_dir  = _edmd_file.parent
if str(_edmd_dir) not in sys.path:
    sys.path.insert(0, str(_edmd_dir))

_spec  = importlib.util.spec_from_file_location("edmd_runner", str(_edmd_file))
_edmd  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_edmd)

# This is YOUR function that expects (x, p_max)
_default_lift = getattr(_edmd, "lift_1d")   # signature: lift_1d(x: np.ndarray, p_max: int) -> np.ndarray

# Optional: if your edmd_runner also exposes a batched geom_observables(X, p_max),
# we’ll use it to accelerate reference lifting (falls back to per-row otherwise).
_geom_obs = getattr(_edmd, "geom_observables", None)


class Koopman_MPC:
    """
    Linear Koopman MPC:
        z_{k+1} = A z_k + B u_k,   x_k = C z_k
    Cost (not lifted):
        Σ (C z_k - x_ref_k)^T Q (C z_k - x_ref_k) + Σ (u_k - u_ref_k)^T R (u_k - u_ref_k)
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        horizon: int = 12,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        lift_fn=None,                       # keep override option
        use_friction_pyramid: bool = True,
        solver_dense: bool = True,
        osqp_opts: Optional[dict] = None,
        use_u_ref_term: bool = False,
    ):
        self.A = np.asarray(A)
        self.B = np.asarray(B)
        self.C = np.asarray(C)

        self.N  = int(horizon)
        self.nz = self.A.shape[0]
        self.nu = self.B.shape[1]
        self.nx = self.C.shape[0]
        self.nlegs = self.nu // 3
        assert self.nu == 3 * self.nlegs, "u must stack [fx, fy, fz] per leg."

        self.Q = np.eye(self.nx) if Q is None else np.asarray(Q)
        self.R = 0.1 * np.eye(self.nu) if R is None else np.asarray(R)
        self.Q_lift = self.C.T @ self.Q @ self.C
        self.use_u_ref_term = bool(use_u_ref_term)

        self.mass    = float(mass)
        self.gravity = float(gravity_constant)
        self.mu      = float(mpc_params["mu"])

        # --- LIFTING SETUP --------------------------------------------------
        # Prefer passed-in lift; else use your edmd_runner.lift_1d
        self.lift_fn = _default_lift if lift_fn is None else lift_fn

        # Pull p_max from config if present; default to 2
        try:
            self.p_max = int(mpc_params.get("koopman", {}).get("p_max", 2))
        except Exception:
            self.p_max = 2

        # Fast path: optional batched lift if geom_observables is available
        self._batched_geom = callable(_geom_obs)

        # One-time sanity: ensure lift dimension == nz
        _probe_x = np.zeros((self.C.shape[0],), float)
        _phi = self._lift1(_probe_x)
        if _phi.size != self.nz:
            raise ValueError(
                f"Lift dimension mismatch: lift(x) returns { _phi.size }, "
                f"but A is ({self.nz}×{self.nz}). Adjust p_max or A/B/C."
            )
        # --------------------------------------------------------------------

        # Parse GRF bounds into full 12-vectors per step
        self.grf_min_1step, self.grf_max_1step = self._parse_grf_bounds(
            mpc_params["grf_min"], mpc_params["grf_max"]
        )

        self.use_friction = bool(use_friction_pyramid)

        # Build QP matrices (H, A_all) and base bounds (no z0/contact yet)
        self._build_qp_backend_mats()

        # Native OSQP backend (simple & robust)
        self._init_native_osqp(osqp_opts)
        self.backend = "native_osqp"
        self._last_x = None
        self._last_u_ref = None

    # --------------------------- public API ---------------------------
    def solve(
        self,
        x0: np.ndarray,
        x_ref: Union[np.ndarray, Tuple[np.ndarray, ...]],
        contact_schedule: Optional[np.ndarray] = None,
        u_ref: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        N, nz, nu, nx = self.N, self.nz, self.nu, self.nx
        n_xblk = nz * (N + 1)

        # Lift initial state
        z0 = self._lift1(x0)

        # Lift references (batch if available)
        if x_ref.ndim == 1:
            zref_one = self._lift1(x_ref)
            zref_all = np.tile(zref_one, (N + 1, 1))
        else:
            assert x_ref.shape == (N + 1, nx)
            zref_all = self._lift_batch(x_ref)

        # Linear term g for QP
        g = np.zeros(self.n_tot)
        # (i) state tracking: -2 * Q_lift * z_ref
        for k in range(N + 1):
            g[k * nz : (k + 1) * nz] = -2.0 * (self.Q_lift @ zref_all[k])

        # (ii) input tracking if enabled
        if not self.use_u_ref_term:
            Uref = np.zeros((self.N, self.nu))
        else:
            if u_ref is None:
                Uref = self._compute_u_ref(
                    contact_schedule if contact_schedule is not None
                    else np.ones((self.N, self.nlegs), int)
                )
            else:
                Uref = np.asarray(u_ref)
                assert Uref.shape == (self.N, self.nu)
        for k in range(N):
            g[n_xblk + k*nu : n_xblk + (k+1)*nu] = -2.0 * (self.R @ Uref[k])
        self._last_u_ref = Uref

        # Variable bounds: pin z0
        lbx = self.lbx_template.copy(); ubx = self.ubx_template.copy()
        lbx[:nz] = z0;                  ubx[:nz] = z0

        # Constraint bounds (+contact schedule zeroing)
        lba = self.lba_base.copy(); uba = self.uba_base.copy()
        if contact_schedule is not None:
            assert contact_schedule.shape == (N, self.nlegs)
            n_dyn  = N * nz
            n_cone = self._n_cone_rows()
            base_u = n_dyn + n_cone
            for k in range(N):
                for j in range(self.nlegs):
                    if contact_schedule[k, j] == 0:  # swing → u_leg = 0
                        row0 = base_u + k * nu + 3 * j
                        lba[row0:row0+3] = 0.0
                        uba[row0:row0+3] = 0.0

        z = self._solve_native_osqp(g, lba, uba, lbx, ubx)
        Z = z[:n_xblk].reshape(N + 1, nz)
        U = z[n_xblk:].reshape(N, nu)
        X = (self.C @ Z.T).T
        return U[0], U, X

    # --------------------------- lifting ---------------------------
    def _lift1(self, x: np.ndarray) -> np.ndarray:
        """
        Calls your lift_1d(x, p_max) robustly and returns a flat (nz,) vector.
        Accepts numpy arrays or CasADi DM.
        """
        x = np.asarray(x, dtype=float).reshape(1, -1)
        phi = self.lift_fn(x, p_max=self.p_max)  # <-- always pass p_max

        # If the lift returns (phi, *rest), keep the first
        if isinstance(phi, (tuple, list)):
            phi = phi[0]

        # CasADi DM → numpy (optional)
        try:
            import casadi as cs
            if isinstance(phi, cs.DM):
                phi = np.array(phi)
        except Exception:
            pass

        return np.asarray(phi, dtype=float).ravel()

    def _lift_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Batched lift for reference trajectory: (N+1, nx) -> (N+1, nz).
        Uses geom_observables if available; else loops over rows.
        """
        X = np.asarray(X, dtype=float)
        if self._batched_geom:
            # Your geom_observables handles batch; we just ravel row-wise if needed
            Z = _geom_obs(X, p_max=self.p_max)
            Z = np.asarray(Z, dtype=float)
            # Expect (N+1, nz) or (N+1, 1, nz). Normalize shapes:
            if Z.ndim == 3 and Z.shape[1] == 1:
                Z = Z[:, 0, :]
            return Z
        else:
            return np.vstack([self._lift1(xi) for xi in X])

    # ------------------------- internals -------------------------

    def _parse_grf_bounds(self, grf_min, grf_max):
        """
        Accepts scalar, (3,), or (nu,) for each of min/max.
        Returns broadcasted (nu,) lower/upper bounds for one stage.
        - Scalar: only constrains fz (fz ∈ [max(0, min), max]).
        - (3,): per-leg [fx, fy, fz] bounds, broadcast to all legs.
        - (nu,): fully specified bounds.
        """
        nu = self.nu
        nlegs = self.nlegs

        gmin = np.array(grf_min, dtype=float).ravel()
        gmax = np.array(grf_max, dtype=float).ravel()

        if gmin.size == 1 and gmax.size == 1:
            lb = -np.inf * np.ones(nu)
            ub =  np.inf * np.ones(nu)
            fz_idx = np.arange(2, nu, 3)
            lb[fz_idx] = max(0.0, float(gmin.item()))
            ub[fz_idx] = float(gmax.item())
            return lb, ub

        if gmin.size == 3 and gmax.size == 3:
            lb_leg = gmin; ub_leg = gmax
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
        """Number of friction-pyramid inequality rows across the whole horizon."""
        return (4 * self.nlegs * self.N) if getattr(self, "use_friction", False) else 0


    def _build_qp_backend_mats(self):
        """
        Construct constant Hessian H, constraint matrix A_all and base bounds
        (without z0 pin or contact zeroing).
        """
        N, nz, nu = self.N, self.nz, self.nu
        n_xblk = nz * (N + 1)
        n_ublk = nu * N
        self.n_tot = n_xblk + n_ublk

        # Hessian blocks: 0.5 * x^T H x + g^T x  (use 2*Q, 2*R so the math matches)
        Hx = sp.block_diag([2 * self.Q_lift] * (N + 1), format="csc")
        Hu = sp.block_diag([2 * self.R]     * N,       format="csc")
        self.H = sp.block_diag((Hx, Hu), format="csc")

        # Dynamics: z_{k+1} - A z_k - B u_k = 0
        A_dyn = sp.lil_matrix((nz * N, self.n_tot))
        for k in range(N):
            # z_{k+1}
            A_dyn[k * nz : (k + 1) * nz, (k + 1) * nz : (k + 2) * nz] = sp.eye(nz)
            # -A z_k
            A_dyn[k * nz : (k + 1) * nz, k * nz : (k + 1) * nz] = -self.A
            # -B u_k
            A_dyn[k * nz : (k + 1) * nz, n_xblk + k * nu : n_xblk + (k + 1) * nu] = -self.B
        lba_dyn = np.zeros(nz * N)
        uba_dyn = np.zeros(nz * N)

        # Friction pyramid (optional)
        if self.use_friction:
            rows_per_leg = 4  # [+fx - μfz; -fx - μfz; +fy - μfz; -fy - μfz] ≤ 0
            n_cone = rows_per_leg * self.nlegs * N
            A_cone = sp.lil_matrix((n_cone, self.n_tot))
            lba_cone = -np.inf * np.ones(n_cone)
            uba_cone = np.zeros(n_cone)
            r = 0
            for k in range(N):
                uoff = n_xblk + k * nu
                for j in range(self.nlegs):
                    c = uoff + 3 * j
                    # +fx - μ fz ≤ 0
                    A_cone[r, c + 0] =  1.0; A_cone[r, c + 2] = -self.mu; r += 1
                    # -fx - μ fz ≤ 0
                    A_cone[r, c + 0] = -1.0; A_cone[r, c + 2] = -self.mu; r += 1
                    # +fy - μ fz ≤ 0
                    A_cone[r, c + 1] =  1.0; A_cone[r, c + 2] = -self.mu; r += 1
                    # -fy - μ fz ≤ 0
                    A_cone[r, c + 1] = -1.0; A_cone[r, c + 2] = -self.mu; r += 1
            assert r == n_cone
        else:
            A_cone = sp.csc_matrix((0, self.n_tot))
            lba_cone = np.zeros(0)
            uba_cone = np.zeros(0)

        # GRF box bounds via identity on u
        A_u = sp.lil_matrix((nu * N, self.n_tot))
        for k in range(N):
            A_u[k * nu : (k + 1) * nu, n_xblk + k * nu : n_xblk + (k + 1) * nu] = sp.eye(nu)
        lba_u = np.tile(self.grf_min_1step, N)
        uba_u = np.tile(self.grf_max_1step, N)

        # Stack constraints
        self.A_all   = sp.vstack([A_dyn.tocsc(), A_cone.tocsc(), A_u.tocsc()], format="csc")
        self.lba_base = np.concatenate([lba_dyn, lba_cone, lba_u])
        self.uba_base = np.concatenate([uba_dyn, uba_cone, uba_u])

        # Variable-bounds templates (we’ll fold them into constraints in OSQP setup)
        self.lbx_template = -np.inf * np.ones(self.n_tot)
        self.ubx_template =  np.inf * np.ones(self.n_tot)


    def _init_native_osqp(self, osqp_opts: Optional[dict]):
        import osqp

        # Fold variable bounds into constraints once: [A_all; I] x ∈ [lba; lbx], [uba; ubx]
        I = sp.eye(self.n_tot, format="csc")
        self.A_osqp = sp.vstack([self.A_all, I], format="csc")

        # Initial placeholders (updated at solve time)
        l0 = np.concatenate([self.lba_base, self.lbx_template])
        u0 = np.concatenate([self.uba_base, self.ubx_template])

        # Settings
        settings = dict(
            eps_abs=1e-5,
            eps_rel=1e-5,
            max_iter=20000,
            verbose=False,
            warm_start=True,
            polish=True,
        )
        if osqp_opts and isinstance(osqp_opts, dict):
            settings.update(osqp_opts)

        # P must be symmetric csc
        P = (self.H + self.H.T) * 0.5

        self.osqp = osqp.OSQP()
        self.osqp.setup(P=P, q=np.zeros(self.n_tot), A=self.A_osqp, l=l0, u=u0, **settings)


    def _solve_native_osqp(self, g, lba, uba, lbx, ubx):
        # Update bounds and linear term
        l = np.concatenate([lba, lbx])
        u = np.concatenate([uba, ubx])

        # Warm-start if available
        if getattr(self, "_last_x", None) is not None and self._last_x.size == self.n_tot:
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


    def _compute_u_ref(self, contact_schedule: Optional[np.ndarray]) -> np.ndarray:
        """
        Baseline GRFs: distribute mg across stance legs at each stage.
        Returns Uref with shape (N, nu), legs in fixed order, each leg [fx, fy, fz].
        """
        if contact_schedule is None:
            cs = np.ones((self.N, self.nlegs), dtype=int)
        else:
            cs = np.asarray(contact_schedule, dtype=int)
            assert cs.shape == (self.N, self.nlegs)

        Uref = np.zeros((self.N, self.nu))
        mg = float(getattr(self, "mass", 0.0) * getattr(self, "gravity", 0.0))

        for k in range(self.N):
            nstance = int(cs[k].sum())
            if nstance <= 0:  # flight
                continue
            fz_each = mg / nstance
            for j in range(self.nlegs):
                if cs[k, j] == 1:
                    base = 3 * j
                    Uref[k, base + 0] = 0.0
                    Uref[k, base + 1] = 0.0
                    Uref[k, base + 2] = fz_each
        return Uref

