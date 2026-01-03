# Basic EDMD Model with Δp Position Handling and Gravity Bias

This document describes a **basic EDMD / Koopman model** for SRBD-level dynamics
where **position is handled via a delta-position (Δp) update**, while the learned
model focuses on the **force-driven dynamics** (velocity and angular motion).

A **constant observable** is explicitly included in the lifting to handle
**gravity and other affine bias terms**.

This formulation is simple, robust, and well-suited for use inside MPC.

---

## 1. Motivation

Directly learning absolute position dynamics with EDMD often leads to:
- integrator drift,
- poor long-horizon predictions,
- sensitivity to estimator noise.

However, SRBD kinematics already give an exact and linear relationship:
\[
p_{k+1} = p_k + \Delta t\, v_{W,k}
\]

Therefore, the recommended approach is:
- **do not learn absolute position** with EDMD,
- handle position using a **Δp (incremental) update**,
- use EDMD to learn how **velocities evolve under contact forces and gravity**.

---

## 2. State and Input Definitions

### Physical state (used in MPC / rollout)

We consider the following SRBD-level quantities:

- Position (world frame):  
  \[
  p_k \in \mathbb{R}^3
  \]

- Linear velocity (world frame):  
  \[
  v_{W,k} \in \mathbb{R}^3
  \]

- Angular velocity (body frame, optional):  
  \[
  \omega_{B,k} \in \mathbb{R}^3
  \]

Position is **not** included in the EDMD state.

---

### Control input

The control input is a **net wrench** applied at the CoM:
\[
u_k =
\begin{bmatrix}
F_k \\
\tau_k
\end{bmatrix}
\in \mathbb{R}^6
\]

where:
- \(F_k\) is the net force,
- \(\tau_k\) is the net torque.

(If MPC uses GRFs, the wrench is obtained via a linear mapping \(u = H f\).)

---

## 3. Δp Position Update (Exact Kinematics)

Position is propagated using exact discrete-time kinematics:
\[
\Delta p_k := p_{k+1} - p_k = \Delta t\, v_{W,k}
\]
\[
p_{k+1} = p_k + \Delta p_k
\]

This update is:
- linear,
- exact under zero-order hold,
- independent of the learned model.

EDMD does **not** attempt to learn this integrator.

---

## 4. EDMD State for Learning

The EDMD model focuses on the **force-driven state**:
\[
x_k =
\begin{bmatrix}
v_{W,k} \\
\omega_{B,k}
\end{bmatrix}
\in \mathbb{R}^{n_x}
\]

(Angular velocity can be omitted for purely translational models.)

---

## 5. Lifted State with Constant Observable

To capture **affine effects** such as gravity, we define the lifted state as:
\[
z_k = \psi(x_k) =
\begin{bmatrix}
1 \\
x_k \\
\phi(x_k)
\end{bmatrix}
\in \mathbb{R}^{n_z}
\]

where:
- the leading constant `1` enables representation of constant/bias terms,
- \(\phi(x)\) contains nonlinear features (e.g., polynomials).

### Example lifting
A simple and stable choice:
\[
\phi(x) = x \odot x
\]
(elementwise squares).

---

## 6. EDMD with Control (EDMDc)

Using snapshot data \((x_k, u_k, x_{k+1})\), EDMDc learns:
\[
z_{k+1} = A z_k + B u_k
\]

Because the constant observable is included, this implicitly represents:
\[
x_{k+1} \approx A_x x_k + B_x u_k + c_x
\]
where \(c_x\) captures gravity and other constant effects.

---

## 7. Training Data Construction

From logged data:
- \(v_{W,k}\), \(\omega_{B,k}\) (estimator),
- \(u_k\) (net wrench or mapped GRFs),
- timestep \(\Delta t\),

construct:
\[
x_k = [v_{W,k};\ \omega_{B,k}]
\]
\[
x_{k+1} = [v_{W,k+1};\ \omega_{B,k+1}]
\]

and fit EDMDc on:
\[
(z_k,\ u_k,\ z_{k+1})
\]

Position does **not** enter the regression.

---

## 8. Rollout with Δp Handling

At runtime or inside MPC:

1. Predict lifted dynamics:
   \[
   z_{k+1} = A z_k + B u_k
   \]

2. Read out velocity:
   \[
   \hat v_{W,k} = C_v z_k
   \]

3. Update position using kinematics:
   \[
   p_{k+1} = p_k + \Delta t\, \hat v_{W,k}
   \]

This separates:
- **learning** (velocity dynamics),
- **integration** (position update).

---

## 9. Advantages of the Δp EDMD Formulation

- Avoids learning an integrator (reduces drift)
- Robust to estimator noise in position
- Naturally compatible with MPC
- Constant observable cleanly captures gravity
- Matches standard SRBD kinematics

This formulation is especially effective for:
- velocity tracking MPC,
- joystick teleoperation,
- short-horizon predictive control.

---

## 10. Summary

**Key design choices:**

- EDMD state: velocities (and angular velocities)
- Position: updated via exact Δp kinematics
- Lifting: includes constant `1` for affine terms
- Gravity: handled implicitly through the bias term
- MPC: uses learned dynamics + exact kinematics

This provides a **simple, stable, and hardware-ready EDMD model** for SRBD-based control.

---
