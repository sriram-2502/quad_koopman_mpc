# quadruped_pympc/interfaces/srbd_controller_interface.py

import numpy as np
from gym_quadruped.utils.quadruped_utils import LegsAttr

from quadruped_pympc import config as cfg
from quadruped_pympc.controllers.koopman import KoopmanConvexMPC

from quadruped_pympc.controllers.convex.mit_convex_mpc import MITConvexCentroidalMPC

def _get_from_cfg():
    mass = getattr(cfg, "mass", 12.0)
    inertia = getattr(cfg, "inertia", np.diag([0.07, 0.26, 0.28]))
    g = getattr(cfg, "gravity_constant", 9.81)
    mp = getattr(cfg, "mpc_params", {}) or {}
    mu = mp.get("mu", 0.5)
    grf_max = mp.get("grf_max", mass * g)
    grf_min = mp.get("grf_min", 0.0)
    return float(mass), np.array(inertia, float), float(g), float(mu), float(grf_min), float(grf_max)


class SRBDControllerInterface:
    """Interface for controllers that optimize gait using an SRBD model."""

    def __init__(self):
        """Constructor for the SRBD controller interface."""
        self.type = cfg.mpc_params["type"]
        self.mpc_dt = cfg.mpc_params["dt"]
        self.horizon = cfg.mpc_params["horizon"]
        self.optimize_step_freq = cfg.mpc_params["optimize_step_freq"]
        self.step_freq_available = cfg.mpc_params["step_freq_available"]

        # Pull physical + limit params from config (single source of truth)
        self.mass, self.inertia_cfg, self.g, self.mu, self.fz_min, self.fz_max = _get_from_cfg()

        self.previous_contact_mpc = np.array([1, 1, 1, 1])

        # 'nominal'       optimizes directly the GRF (gradient-based)
        # 'input_rates'   optimizes delta GRF (gradient-based)
        # 'sampling'      GPU sampling-based MPC
        # 'collaborative' GRF + passive arm model (gradient-based)
        # 'lyapunov'      GRF + Lyapunov constraint (gradient-based)
        # 'kinodynamic'   SRBD with joints (experimental)
        # 'koopman'       learned linear Koopman model (this branch)

        if self.type == "nominal":
            from quadruped_pympc.controllers.gradient.nominal.centroidal_nmpc_nominal import (
                Acados_NMPC_Nominal,
            )
            self.controller = Acados_NMPC_Nominal()

            if self.optimize_step_freq:
                from quadruped_pympc.controllers.gradient.nominal.centroidal_nmpc_gait_adaptive import (
                    Acados_NMPC_GaitAdaptive,
                )
                self.batched_controller = Acados_NMPC_GaitAdaptive()

        elif self.type == "input_rates":
            from quadruped_pympc.controllers.gradient.input_rates.centroidal_nmpc_input_rates import (
                Acados_NMPC_InputRates,
            )
            self.controller = Acados_NMPC_InputRates()

            if self.optimize_step_freq:
                from quadruped_pympc.controllers.gradient.nominal.centroidal_nmpc_gait_adaptive import (
                    Acados_NMPC_GaitAdaptive,
                )
                self.batched_controller = Acados_NMPC_GaitAdaptive()

        elif self.type == "lyapunov":
            from quadruped_pympc.controllers.gradient.lyapunov.centroidal_nmpc_lyapunov import (
                Acados_NMPC_Lyapunov,
            )
            self.controller = Acados_NMPC_Lyapunov()

            if self.optimize_step_freq:
                from quadruped_pympc.controllers.gradient.nominal.centroidal_nmpc_gait_adaptive import (
                    Acados_NMPC_GaitAdaptive,
                )
                self.batched_controller = Acados_NMPC_GaitAdaptive()

        elif self.type == "kinodynamic":
            from quadruped_pympc.controllers.gradient.kinodynamic.kinodynamic_nmpc import (
                Acados_NMPC_KinoDynamic,
            )
            self.controller = Acados_NMPC_KinoDynamic()

            if self.optimize_step_freq:
                from quadruped_pympc.controllers.gradient.nominal.centroidal_nmpc_gait_adaptive import (
                    Acados_NMPC_GaitAdaptive,
                )
                self.batched_controller = Acados_NMPC_GaitAdaptive()

        elif self.type == "sampling":
            if self.optimize_step_freq:
                from quadruped_pympc.controllers.sampling.centroidal_nmpc_jax_gait_adaptive import (
                    Sampling_MPC,
                )
            else:
                from quadruped_pympc.controllers.sampling.centroidal_nmpc_jax import (
                    Sampling_MPC,
                )
            self.controller = Sampling_MPC()

        elif self.type == "koopman":
            mass, inertia_cfg, g, mu, fz_min, fz_max = _get_from_cfg()
            self.controller = KoopmanConvexMPC(
                mass=mass,
                inertia=inertia_cfg,
                N=self.horizon,
                dt=self.mpc_dt,
                g=g,
                mu=mu,
                model_path=None,
                lift_fn=None,
            )

        elif self.type == "mit_convex":
            mass, inertia_cfg, g, mu, fz_min, fz_max = _get_from_cfg()
            # Instantiate the centroidal convex MPC per the MIT paper
            self.controller = MITConvexCentroidalMPC(
                mass=mass,
                inertia=inertia_cfg,
                N=self.horizon,
                dt=self.mpc_dt,
                g=g,
                mu=mu,
                fz_min=fz_min,
                fz_max=fz_max,
                # (optional) tune weights:
                Qv=1.0,   # linear velocity tracking
                Qw=1.0,    # angular velocity tracking
                Rf=1e-6,   # force regularization
            )

        else:
            raise ValueError(f"Unknown MPC type: {self.type}")

    def compute_control(
        self,
        state_current: dict,
        ref_state: dict,
        contact_sequence: np.ndarray,
        inertia: np.ndarray,
        pgg_phase_signal: np.ndarray,
        pgg_step_freq: float,
        optimize_swing: int,
        external_wrenches: np.ndarray = np.zeros((6,)),
    ) -> [LegsAttr, LegsAttr, LegsAttr, LegsAttr, LegsAttr, float, np.ndarray]:
        """Compute control using the selected MPC.

        Args:
            state_current (dict): Current robot state (env-provided dict)
            ref_state (dict): Reference (env/gait generator) dict
            contact_sequence (np.ndarray): Contact sequence
            inertia (np.ndarray): Base inertia (3x3 flattened or similar)
            pgg_phase_signal (np.ndarray): Periodic gait generator phase (0..1 per leg)
            pgg_step_freq (float): Gait step frequency
            optimize_swing (int): Flag for swing optimization (sampling controllers)
            external_wrenches (np.ndarray): External wrench for compensation

        Returns:
            tuple:
                nmpc_GRFs          : LegsAttr of (3,) per leg (commanded GRFs)
                nmpc_footholds     : LegsAttr of (3,) per leg (world frame)
                nmpc_joints_pos    : LegsAttr or None
                nmpc_joints_vel    : LegsAttr or None
                nmpc_joints_acc    : LegsAttr or None
                best_sample_freq   : float (same as pgg_step_freq unless sampling)
                nmpc_predicted_state: np.ndarray (N+1, 12) predicted base states
        """

        current_contact = np.array(
            [contact_sequence[0][0], contact_sequence[1][0], contact_sequence[2][0], contact_sequence[3][0]]
        )

        # --------------------------- Sampling-based MPC ---------------------------
        if self.type == "sampling":
            # Convert data to jax and shift previous solution
            state_current_jax, reference_state_jax = self.controller.prepare_state_and_reference(
                state_current, ref_state, current_contact, self.previous_contact_mpc
            )
            self.previous_contact_mpc = current_contact

            for iter_sampling in range(self.controller.num_sampling_iterations):
                self.controller = self.controller.with_newkey()
                if self.controller.sampling_method == "cem_mppi":
                    if iter_sampling == 0:
                        self.controller = self.controller.with_newsigma(cfg.mpc_params["sigma_cem_mppi"])

                    (
                        nmpc_GRFs,
                        nmpc_footholds,
                        nmpc_predicted_state,
                        self.controller.best_control_parameters,
                        best_cost,
                        best_sample_freq,
                        costs,
                        sigma_cem_mppi,
                    ) = self.controller.jitted_compute_control(
                        state_current_jax,
                        reference_state_jax,
                        contact_sequence,
                        self.controller.best_control_parameters,
                        self.controller.master_key,
                        self.controller.sigma_cem_mppi,
                    )
                    self.controller = self.controller.with_newsigma(sigma_cem_mppi)
                else:
                    nominal_sample_freq = pgg_step_freq
                    (
                        nmpc_GRFs,
                        nmpc_footholds,
                        nmpc_predicted_state,
                        self.controller.best_control_parameters,
                        best_cost,
                        best_sample_freq,
                        costs,
                    ) = self.controller.jitted_compute_control(
                        state_current_jax,
                        reference_state_jax,
                        contact_sequence,
                        self.controller.best_control_parameters,
                        self.controller.master_key,
                        pgg_phase_signal,
                        nominal_sample_freq,
                        optimize_swing,
                    )

            # Format footholds as LegsAttr
            nmpc_footholds = LegsAttr(
                FL=ref_state["ref_foot_FL"][0],
                FR=ref_state["ref_foot_FR"][0],
                RL=ref_state["ref_foot_RL"][0],
                RR=ref_state["ref_foot_RR"][0],
            )
            nmpc_GRFs = np.array(nmpc_GRFs)  # flat or stacked depending on sampling impl

            nmpc_joints_pos = None
            nmpc_joints_vel = None
            nmpc_joints_acc = None

        # ------------------------ Gradient-based / Koopman ------------------------
        else:
            if self.type == "kinodynamic":
                (
                    nmpc_GRFs,
                    nmpc_footholds,
                    nmpc_joints_pos,
                    nmpc_joints_vel,
                    nmpc_joints_acc,
                    nmpc_predicted_state,
                    status,
                ) = self.controller.compute_control(
                    state_current, ref_state, contact_sequence, inertia=inertia, external_wrenches=external_wrenches
                )

                # Convert joint trajectories to LegsAttr
                nmpc_joints_pos = LegsAttr(
                    FL=nmpc_joints_pos[0:3], FR=nmpc_joints_pos[3:6], RL=nmpc_joints_pos[6:9], RR=nmpc_joints_pos[9:12]
                )
                nmpc_joints_vel = LegsAttr(
                    FL=nmpc_joints_vel[0:3], FR=nmpc_joints_vel[3:6], RL=nmpc_joints_vel[6:9], RR=nmpc_joints_vel[9:12]
                )
                nmpc_joints_acc = LegsAttr(
                    FL=nmpc_joints_acc[0:3], FR=nmpc_joints_acc[3:6], RL=nmpc_joints_acc[6:9], RR=nmpc_joints_acc[9:12]
                )

            elif self.type == "mit_convex":
                # MIT convex MPC with acados
                nmpc_GRFs, nmpc_footholds, nmpc_predicted_state = \
                    self.controller.compute_control(
                        state_current, ref_state, contact_sequence, 
                    )
                nmpc_joints_pos = None
                nmpc_joints_vel = None
                nmpc_joints_acc = None

            else:
                # Includes 'nominal', 'input_rates', 'lyapunov', and 'koopman'
                nmpc_GRFs, nmpc_footholds, nmpc_predicted_state, _ = self.controller.compute_control(
                    state_current, ref_state, contact_sequence, inertia=inertia, external_wrenches=external_wrenches
                )

                nmpc_joints_pos = None
                nmpc_joints_vel = None
                nmpc_joints_acc = None

            # Format footholds as LegsAttr (expects list/array of 4 items)
            nmpc_footholds = LegsAttr(
                FL=nmpc_footholds[0], FR=nmpc_footholds[1], RL=nmpc_footholds[2], RR=nmpc_footholds[3]
            )

            best_sample_freq = pgg_step_freq

        # ---------------------------- Post-formatting ----------------------------
        # Mask GRFs by current contacts (swing -> zeros)
        # NOTE: returns per-leg (3,) arrays here; downstream WBC maps to torques.
        nmpc_GRFs = LegsAttr(
            FL=nmpc_GRFs[0:3] * current_contact[0],
            FR=nmpc_GRFs[3:6] * current_contact[1],
            RL=nmpc_GRFs[6:9] * current_contact[2],
            RR=nmpc_GRFs[9:12] * current_contact[3],
        )

        return (
            nmpc_GRFs,
            nmpc_footholds,
            nmpc_joints_pos,
            nmpc_joints_vel,
            nmpc_joints_acc,
            best_sample_freq,
            nmpc_predicted_state,
        )

    # ------------------------------ RTI utility ------------------------------
    def compute_RTI(self):
        # Only meaningful for Acados-based gradient controllers
        self.controller.acados_ocp_solver.options_set("rti_phase", 1)
        self.controller.acados_ocp_solver.solve()
        # print("preparation phase time: ", controller.acados_ocp_solver.get_stats('time_tot'))
