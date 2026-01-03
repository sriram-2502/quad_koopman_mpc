# SRBD State and Frame Conventions

This document defines the **state representation and coordinate frames** used for
Single Rigid Body Dynamics (SRBD) Model Predictive Control (MPC) and
EDMD / Koopman-based modeling on the Unitree Go1 platform.

The conventions follow **MIT-style convex MPC**, common legged-robot estimators
(e.g., `legged_estimation`), and hardware best practices.

---

## State Representation Overview

The SRBD MPC state uses a **mixed-frame representation**:

- **Translational quantities** are expressed in the **world frame**
- **Rotational quantities** are expressed in the **body frame**
- Orientation is represented using **small-angle error coordinates**, not full rotation matrices

This choice preserves linear kinematics, keeps the MPC convex, and matches IMU measurements.

---

## State Variables and Frames

| Quantity | Symbol | Frame | Source | Used in SRBD MPC | Used in EDMD | Notes |
|--------|--------|-------|--------|------------------|--------------|-------|
| CoM / base position | \(p\) | World (W) | State estimator | Yes | Optional | Linear kinematics; often excluded from EDMD |
| Linear velocity | \(v\) | World (W) | State estimator | Yes | Yes | Keeps position update linear |
| Angular velocity | \(\omega\) | Body (B) | IMU gyro | Yes | Yes | Natural IMU frame; constant inertia |
| Orientation | \(R_{WB}\) | World ← Body | State estimator | No (directly) | Optional | Used to compute orientation error |
| Orientation error | \(\theta\) | Local error | Computed from reference | Yes | Yes | Small-angle approximation |
| Contact forces | \(f_i\) | World (W) | MPC decision variable | Yes | Yes | Subject to friction & contact constraints |
| Gravity | \(g\) | World (W) | Known constant | Yes | Yes (via bias) | Handled via affine term in EDMD |
| Constant observable | \(1\) | — | — | No | Required | Enables affine Koopman dynamics |

---

## Why This Representation Is Used

### Linear Position Kinematics
Using world-frame linear velocity yields:
\[
p_{k+1} = p_k + \Delta t\, v_{W,k}
\]
which is **exactly linear** and preserves convexity in MPC.

### Physically Consistent Rotational Dynamics
Angular velocity is kept in the body frame:
- Directly measured by the IMU
- Rigid-body inertia matrix is constant
- Avoids unnecessary frame rotations and noise amplification

### Orientation as an Error State
MPC operates near a nominal orientation.
Using small-angle error coordinates enables:
\[
\theta_{k+1} = \theta_k + \Delta t\, \omega_{B,k}
\]
which is linear and valid over short horizons.

---

## Notes on Estimation

Although IMU measurements are body-frame,
the state estimator provides world-frame position and velocity by
fusing IMU data with leg kinematics and contact constraints.

This allows:
- Measurement in body frame
- Estimation in world frame
- Optimization in world frame

without breaking SRBD MPC structure.

---

## Summary

**Best-practice SRBD state for legged MPC:**
\[
x = [\,p_W,\ v_W,\ \theta,\ \omega_B\,]
\]

This representation is:
- Hardware-proven
- Convex-MPC compatible
- Estimator-friendly
- EDMD / Koopman compatible

---

## SRBD Model Dynamics (MPC Form)

We use a standard SRBD / centroidal dynamics model with contact forces as inputs.
Let there be \(n_c\) contact points (for a quadruped, \(n_c \le 4\)).

### Continuous-time dynamics

**Kinematics**
\[
\dot p_W = v_W
\]

**Translational dynamics**
\[
m\,\dot v_W = \sum_{i=1}^{n_c} f_{i,W} + m\,g_W
\]

**Orientation error kinematics (small-angle approximation)**
\[
\dot\theta \approx \omega_B
\]

**Rotational dynamics (body frame)**
\[
I_B\,\dot\omega_B + \omega_B \times (I_B\,\omega_B)
= \sum_{i=1}^{n_c} \tau_{i,B}
\]
where the contact torque contribution from foot \(i\) is
\[
\tau_{i,B} = r_{i,B} \times f_{i,B},\qquad f_{i,B} = R_{WB}^\top f_{i,W}
\]
and \(r_{i,B}\) is the vector from CoM to contact point expressed in body frame.

> In convex MPC implementations, the nonlinear term
> \(\omega_B \times (I_B\omega_B)\) is often dropped or linearized about the
> current operating point to keep the dynamics affine in the decision variables.

---

### Discrete-time dynamics (Euler discretization)

For MPC with timestep \(\Delta t\),
\[
p_{k+1} = p_k + \Delta t\, v_{W,k}
\]
\[
v_{W,k+1} = v_{W,k} + \Delta t\left(\frac{1}{m}\sum_{i=1}^{n_c} f_{i,W,k} + g_W\right)
\]
\[
\theta_{k+1} = \theta_k + \Delta t\, \omega_{B,k}
\]
\[
\omega_{B,k+1} \approx \omega_{B,k} + \Delta t\, I_B^{-1}\left(\sum_{i=1}^{n_c} \left(r_{i,B,k}\times f_{i,B,k}\right)\right)
\]
(optionally with additional linearized terms if using a more accurate rotational model).

---

### Input definition

Stack the contact forces into a single input vector:
\[
u_k = \begin{bmatrix}
f_{1,W,k} \\
\vdots \\
f_{n_c,W,k}
\end{bmatrix} \in \mathbb{R}^{3n_c}
\]

These inputs are constrained by:
- contact schedule (swing legs: \(f_i = 0\))
- unilateral normal force (\(f_{iz}\ge 0\))
- friction cone / pyramid constraints

---

