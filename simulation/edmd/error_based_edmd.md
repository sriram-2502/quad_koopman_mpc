# Error-Based EDMD + SRBD MPC with GRFs (Joystick Velocity Tracking)

This document describes a **clean, hardware-oriented setup** for using an
**error-based EDMD (Koopman) model** inside an **SRBD-style MPC** where the
**decision variables are ground reaction forces (GRFs)** and the objective is
**joystick velocity tracking**.

The design is intentionally minimal, convex, and consistent with
MIT-style convex MPC and common Go1 estimation stacks.

---

## 1. Control Objective

The robot is controlled via a joystick that specifies **velocity commands**, not positions.

### Joystick reference
At each control step:
\[
v_W^{ref} = [v_x^{ref}, v_y^{ref}, 0], \qquad
\omega_{B,z}^{ref}
\]

When the joystick is released:
\[
v_W^{ref} = 0, \qquad \omega_{B,z}^{ref} = 0
\]

The controller must:
- track the commanded velocity and yaw rate,
- naturally return to zero velocity when the reference goes to zero,
- remain stable and feasible on hardware.

---

## 2. Tracking State and Error Definition

### Tracked SRBD-level state (minimal)
We define the **tracking state** as:
\[
x_d =
\begin{bmatrix}
v_W \\
\omega_{B,z}
\end{bmatrix}
\in \mathbb{R}^4
\]

where:
- \(v_W \in \mathbb{R}^3\) is world-frame linear velocity,
- \(\omega_{B,z}\) is body-frame yaw rate.

### Tracking error
Given a reference \(x_d^{ref}\), define the error:
\[
e_k = x_{d,k} - x_{d,k}^{ref}
\]

When the joystick reference is zero, \(e_k = x_{d,k}\), so regulating the error
to zero makes the robot stop.

---

## 3. Control Inputs: GRFs and Net Wrench Mapping

### MPC decision variables
The MPC optimizes **ground reaction forces**:
\[
f_k =
\begin{bmatrix}
f_{1,k} \\
f_{2,k} \\
f_{3,k} \\
f_{4,k}
\end{bmatrix}
\in \mathbb{R}^{12}, \qquad f_{i,k}\in\mathbb{R}^3
\]

These are expressed in the **world frame**.

### Net wrench about the CoM
The net wrench is:
\[
u_k =
\begin{bmatrix}
F_k \\
\tau_k
\end{bmatrix}
\in \mathbb{R}^6
\]

with:
\[
F_k = \sum_{i=1}^4 f_{i,k}, \qquad
\tau_k = \sum_{i=1}^4 r_{i,k} \times f_{i,k}
\]

This can be written compactly as:
\[
u_k = H_k f_k
\]

where:
\[
H_k =
\begin{bmatrix}
I_3 & I_3 & I_3 & I_3 \\
[r_{1,k}]_\times & [r_{2,k}]_\times & [r_{3,k}]_\times & [r_{4,k}]_\times
\end{bmatrix}
\in \mathbb{R}^{6\times 12}
\]

The vectors \(r_{i,k}\) are CoM-to-foot vectors in the world frame.
In practice, \(H_k\) is often frozen at the current step across the horizon.

---

## 4. Error-Based EDMD Model (Training)

### Lifted error state
We define a lifted state:
\[
z_k = \psi(e_k) =
\begin{bmatrix}
1 \\
e_k \\
\phi(e_k)
\end{bmatrix}
\]

Typical choices for \(\phi(\cdot)\):
- elementwise squares \(e \odot e\),
- small cross terms if needed.

The constant `1` enables affine dynamics.

### Nominal wrench and deviation
Assume access to a nominal wrench \(u_k^{ref}\)
(e.g., gravity compensation or gait feedforward).

Define:
\[
\delta u_k = u_k - u_k^{ref}
\]

### Learned dynamics
EDMD with control (EDMDc) learns:
\[
z_{k+1} = A z_k + B \,\delta u_k
\]

This model is trained on data triples:
\[
(e_k,\ \delta u_k,\ e_{k+1})
\]

---

## 5. Substituting GRFs into the EDMD Dynamics

Since:
\[
\delta u_k = H_k f_k - u_k^{ref}
\]

the lifted dynamics become:
\[
z_{k+1}
= A z_k + B (H_k f_k - u_k^{ref})
\]

or equivalently:
\[
z_{k+1} = A z_k + (B H_k) f_k + c_k,
\qquad c_k = -B u_k^{ref}
\]

This is **affine in the GRFs**, so it fits directly into a QP.

---

## 6. MPC Formulation

### Decision variables
\[
\{z_0,\dots,z_N,\ f_0,\dots,f_{N-1}\}
\]

### Initial condition
From the estimator and joystick:
\[
e_0 = x_{d,0} - x_{d,0}^{ref}, \qquad
z_0 = \psi(e_0)
\]

### Dynamics constraints
For \(k = 0,\dots,N-1\):
\[
z_{k+1} - A z_k - B H_k f_k = -B u_k^{ref}
\]

### Contact and friction constraints (per foot)
For each foot \(i\) at step \(k\):

- **Swing foot**:
\[
f_{i,k} = 0
\]

- **Stance foot (friction pyramid)**:
\[
\begin{aligned}
-\mu f_{z} &\le f_x \le \mu f_{z} \\
-\mu f_{z} &\le f_y \le \mu f_{z} \\
0 &\le f_z \le f_{z,\max}
\end{aligned}
\]

These are linear inequalities → QP-compatible.

---

## 7. MPC Cost Function (Velocity Tracking)

### Error readout
Because the error is included linearly in the lift:
\[
\hat e_k = C_e z_k
\]

### Cost
\[
J =
\sum_{k=0}^{N-1}
\left(
\hat e_k^\top Q \hat e_k
+ f_k^\top R_f f_k
\right)
+ \hat e_N^\top Q_f \hat e_N
\]

Optional (recommended for hardware):
\[
\sum_{k=0}^{N-2} \|f_{k+1}-f_k\|_{R_\Delta}^2
\]

This ensures smooth force profiles.

---

## 8. Runtime Control Loop

At each control cycle:

1. Read estimator:
   \(v_W,\ \omega_{B,z}\)
2. Read joystick:
   \(v_W^{ref},\ \omega_{B,z}^{ref}\)
3. Compute error \(e_0\) and lift \(z_0\)
4. Build and solve the QP
5. Apply first-step forces \(f_0\)
6. Map GRFs → joint torques via WBC
7. Repeat at next cycle

When the joystick reference goes to zero, the MPC naturally drives
\(e \rightarrow 0\), causing the robot to stop.

---

## 9. Key Takeaways

- Joystick control is **velocity tracking**, not position tracking.
- EDMD is trained on **tracking error dynamics**, not raw states.
- MPC optimizes **GRFs**, preserving friction and contact constraints.
- Net wrench appears only as a **linear mapping** inside the dynamics.
- No position state is required for joystick teleoperation.
- The formulation is a **convex QP**, suitable for real-time hardware use.

---
