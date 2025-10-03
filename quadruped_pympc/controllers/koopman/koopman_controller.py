# quadruped_pympc/controllers/koopman/koopman_controller.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import numpy as np

from .koopman_mpc import Koopman_MPC


class KoopmanController:
    """
    Adapter around Koopman_MPC with the SAME compute_control(...) signature/returns
    as the nominal gradient controllers used by SRBDControllerInterface.
    """

    # Static handle for the most recently created instance (handy for diagnostics)
    _last_instance = None

    def __init__(self, cfg):
        self.cfg = cfg
        self.N  = int(cfg.koopman_mpc_params.get("horizon", 12))
        self.nx = 12  # x = [pos(3), eul_xyz(3), v_world(3), w_world(3)]

        # Debug throttling
        self._dbg_enabled = bool(cfg.koopman_mpc_params.get("debug", False))
        self._dbg_every   = int(cfg.koopman_mpc_params.get("debug_every", 200))
        self._dbg_k       = 0

        # --- Load EDMD Koopman operators (A, B, optional C) ---
        model_path = Path(cfg.koopman_mpc_params["model_path"])
        if not model_path.is_absolute():
            repo_root = Path(__file__).resolve().parents[3]
            model_path = repo_root / model_path
        data = np.load(model_path)
        A = data["A"]; B = data["B"]
        nz = A.shape[0]
        if "C" in data.files:
            C = data["C"]
        else:
            if nz < self.nx:
                raise ValueError(f"Lifted dim nz={nz} < nx={self.nx}; cannot build C=[I 0].")
            C = np.zeros((self.nx, nz))
            C[:self.nx, :self.nx] = np.eye(self.nx)

        # --- Weights ---
        Q = np.array(cfg.koopman_mpc_params.get("Q", np.eye(self.nx)))
        R = np.array(cfg.koopman_mpc_params.get("R", 0.1 * np.eye(B.shape[1])))

        # --- EDMD lift honoring p_max from config, with clean wrapper ---
        p_max_cfg = int(cfg.koopman_mpc_params.get("p_max", 5))

        # import lift_1d from simulation/edmd/edmd_runner.py
        import importlib.util, sys
        edmd_file = Path(__file__).resolve().parents[3] / "simulation" / "edmd" / "edmd_runner.py"
        edmd_dir  = edmd_file.parent
        if str(edmd_dir) not in sys.path:
            sys.path.insert(0, str(edmd_dir))
        spec = importlib.util.spec_from_file_location("edmd_runner", str(edmd_file))
        edmd = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(edmd)
        _lift_1d = getattr(edmd, "lift_1d")

        # Clean wrapper: accepts **kwargs so Koopman_MPC can pass p_max without errors.
        def lift_with_cfg(X, **kwargs):
            # Always use the controller's p_max_cfg; ignore any external p_max.
            return _lift_1d(X, p_max=p_max_cfg)

        # --- MPC instance ---
        # Note: Koopman_MPC may call lift_fn(..., p_max=...), which our wrapper accepts.
        self.mpc = Koopman_MPC(A=A, B=B, C=C, horizon=self.N, Q=Q, R=R, lift_fn=lift_with_cfg)

        # --- logging buffers ---
        self._log_enabled = bool(cfg.koopman_mpc_params.get("log_history", True))
        self._hist = {
            "x": [],        # (12,)
            "x_ref": [],    # (12,)
            "u0": [],       # (12,)
            "u_ref0": [],   # (12,)
            "stance": [],   # (4,) 0/1
        }

        # Publish last instance handle
        KoopmanController._last_instance = self
        self._last = None

    # ---------------------------------------------------------------------
    def compute_control(
        self,
        state_current: dict,
        ref_state: dict,
        contact_sequence: np.ndarray,
        inertia: np.ndarray,
        external_wrenches: np.ndarray = np.zeros((6,)),
        **kwargs,
    ):
        """
        Returns:
            nmpc_GRFs           : (12,) flat vector [FL,FR,RL,RR] x [fx,fy,fz]
            nmpc_footholds      : (4,3) footholds (world)
            nmpc_predicted_state: (N+1,12)
            status              : dict
        """
        # ---- current state unpack ---------------------------------------------------
        def _get(d: dict, names, default):
            for n in names:
                if n in d:
                    return np.asarray(d[n]).reshape(-1)
            return np.asarray(default).reshape(-1)

        p   = _get(state_current, ["base_pos", "base_position", "p"], np.zeros(3))
        eul = _get(state_current, ["base_ori_euler_xyz", "base_eul", "euler", "rpy"], np.zeros(3))
        v   = _get(state_current, ["base_lin_vel", "base_lin_vel_world", "v_world", "v"], np.zeros(3))
        w   = _get(state_current, ["base_ang_vel", "base_ang_vel_world", "w_world", "w"], np.zeros(3))
        x   = np.concatenate([p[:3], eul[:3], v[:3], w[:3]])

        # ---- build MPC reference: height only --------------------------------------
        ref_z_cfg = float(self.cfg.simulation_params.get("ref_z", getattr(self.cfg, "hip_height", 0.28)))
        z_from_ref = _get(ref_state, ["ref_base_height"], np.array([ref_z_cfg]))
        ref_z = float(z_from_ref[0])

        # All-zero ref except z-height
        x_ref = np.zeros_like(x)
        x_ref[2] = ref_z

        # ---- contact schedule (N,4) ------------------------------------------------
        contact_sched = self._schedule_from_sequence(contact_sequence, self.N)

        # ---- baseline u_ref: split mg across stance feet each step -----------------
        try:
            from quadruped_pympc import config as _cfg
            mg_total = _cfg.mass * _cfg.gravity_constant
        except Exception:
            mg_total = 0.0

        N = self.N; nu = 12
        Uref = np.zeros((N, nu), dtype=float)  # [FL,FR,RL,RR] x [fx,fy,fz]
        if mg_total > 0:
            for k in range(N):
                stance = contact_sched[k].astype(int)  # (4,)
                nstance = int(stance.sum())
                if nstance > 0:
                    fz_each = mg_total / nstance
                    for leg_id in range(4):
                        if stance[leg_id] == 1:
                            base = 3 * leg_id
                            Uref[k, base + 2] = fz_each  # fx=0, fy=0, fz=fz_each

        # ---- solve Koopman MPC with u_ref -----------------------------------------
        u0, U, X = self.mpc.solve(x0=x, x_ref=x_ref, contact_schedule=contact_sched, u_ref=Uref)

        # ---- (optional) debug ------------------------------------------------------
        if self._dbg_enabled:
            self._dbg_k += 1
            if (self._dbg_k % self._dbg_every) == 0:
                grfs = u0.reshape(4, 3)
                sum_fz = grfs[:, 2].sum()
                print(f"[KMPC] k={self._dbg_k} | z={X[0,2]:.3f}→{ref_z:.3f} | Σfz={sum_fz:.1f} | nstance={int(contact_sched[0].sum())}")

        # ---- pack footholds (defaults to zeros if missing) -------------------------
        feet_keys = ["ref_foot_FL", "ref_foot_FR", "ref_foot_RL", "ref_foot_RR"]
        nmpc_footholds = []
        for k in feet_keys:
            if k in ref_state:
                vref = np.asarray(ref_state[k])
                v0 = vref[0] if vref.ndim > 1 else vref
                nmpc_footholds.append(v0.reshape(3))
            else:
                nmpc_footholds.append(np.zeros(3))

        # ---- log one-step snapshot -------------------------------------------------
        if self._log_enabled:
            self._hist["x"].append(x.copy())
            self._hist["x_ref"].append(x_ref.copy())
            self._hist["u0"].append(u0.copy())
            self._hist["u_ref0"].append(Uref[0].copy())
            self._hist["stance"].append(contact_sched[0].astype(int).copy())

        status = {"ok": True}
        self._last = {"u0": u0.copy(), "U": U.copy(), "X": X.copy(), "Uref": Uref.copy()}
        return u0.reshape(-1), np.array(nmpc_footholds), X, status

    # ---------------------------------------------------------------------
    @staticmethod
    def _schedule_from_sequence(contact_sequence: np.ndarray, N: int) -> np.ndarray:
        """
        Accept common shapes and return (N,4) schedule with 1=stance, 0=swing.
        """
        cs = np.asarray(contact_sequence)
        if cs.ndim == 2:
            if cs.shape[0] == 4 and cs.shape[1] >= N:  # legs x time
                return cs[:, :N].T.astype(int)
            if cs.shape[1] == 4 and cs.shape[0] >= N:  # time x legs
                return cs[:N, :].astype(int)
        try:
            current = np.array([cs[0, 0], cs[1, 0], cs[2, 0], cs[3, 0]], dtype=int)
        except Exception:
            current = np.ones(4, dtype=int)
        return np.tile(current, (N, 1))

    # Convenience (optional)
    def compute_grfs(self, x: np.ndarray, x_ref: np.ndarray, contact_schedule: Optional[np.ndarray] = None):
        return self.mpc.solve(x0=x, x_ref=x_ref, contact_schedule=contact_schedule)

    def reset(self):
        self._last = None
        for k in self._hist:
            self._hist[k].clear()

    def get_history(self):
        """
        Return stacked numpy arrays for keys: x, x_ref, u0, u_ref0, stance.
        """
        out = {}
        for k, v in self._hist.items():
            try:
                out[k] = np.vstack(v) if v and np.asarray(v[0]).ndim > 0 else np.array(v)
            except Exception:
                out[k] = np.array(v)
        return out

    @staticmethod
    def last_instance():
        """
        Retrieve the most recently created KoopmanController (for diagnostics/plots).
        """
        return KoopmanController._last_instance
