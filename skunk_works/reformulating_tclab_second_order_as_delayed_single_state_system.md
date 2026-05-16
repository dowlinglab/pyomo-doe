# Reformulating the TCLab Second-Order Model as a Delayed Single-State System

## 1. Original TCLab Second-Order Model Formulation

The standard TCLab model used in the CBE and Pyomo.DoE notebooks is a two-state lumped thermal model. The model distinguishes between:

* the heater temperature $T_H$
* the sensor/plate temperature $T_S$

with ambient temperature $T_{amb}$.

The governing equations are:

$$
C_p^H \frac{dT_H}{dt}
=
U_a(T_{amb} - T_H)
+ U_b(T_S - T_H)
+ \alpha P u(t)
$$

$$
C_p^S \frac{dT_S}{dt}
=
U_b(T_H - T_S)
$$

where:

| Parameter | Meaning |
| --- | --- |
| $C_p^H$ | effective heater heat capacity |
| $C_p^S$ | effective sensor/plate heat capacity |
| $U_a$ | heat transfer coefficient to ambient |
| $U_b$ | heat transfer coefficient between heater and sensor |
| $\alpha$ | heater gain constant |
| $P$ | maximum heater power |
| $u(t)$ | heater input signal ($Q_1$, typically 0-100%) |

The measured output is:

$$
y(t) = T_S(t)
$$

while $T_H(t)$ is unmeasured.

The model therefore contains:

* 2 states
* 4 fitted physical parameters
* only 1 measured output

---

## 2. Estimability Challenges

The linked Pyomo.DoE notebooks demonstrate substantial practical estimability challenges when estimating:

$$
U_a, \quad U_b, \quad C_p^H, \quad C_p^S
$$

from TCLab data.

The primary issues are:

### 2.1 Only One Temperature is Measured

The model contains two thermal states:

$$
T_H, \quad T_S
$$

but only:

$$
T_S
$$

is observed experimentally.

As a result, the estimation algorithm attempts to infer hidden heater dynamics entirely from sensor data.

---

### 2.2 Parameters Enter in Coupled Combinations

The dynamics depend strongly on parameter ratios and products rather than on the individual parameters themselves.

For example:

$$
\frac{U_a}{C_p^H}, \qquad \frac{U_b}{C_p^H}, \qquad \frac{U_b}{C_p^S}
$$

appear repeatedly in the state-space model.

This creates strong parameter correlations.

---

### 2.3 Similar Dynamic Effects

Several parameters influence the model in qualitatively similar ways:

* increasing $C_p^H$ slows heater dynamics
* decreasing $U_b$ also slows sensor response
* increasing $C_p^S$ slows sensor response

The measured data therefore cannot easily distinguish among these effects.

---

### 2.4 Flat Profile Likelihoods

The profile likelihood analyses in the linked notebooks show:

* broad confidence regions
* nearly flat likelihood profiles
* substantial parameter tradeoffs

This indicates weak practical identifiability.

---

### 2.5 Regularization is Needed

The regularization notebook introduces prior information because the experimental data alone do not sufficiently constrain all four parameters.

This is a strong indication that the parameterization is over-complex relative to the available measurements.

---

## 3. Transfer Function Perspective on the Estimability Problem

The estimability challenge becomes clearer from the transfer function.

Define deviation variables:

$$
T_H' = T_H - T_{amb}
$$

$$
T_S' = T_S - T_{amb}
$$

The transfer function from heater input $u(t)$ to measured sensor temperature $T_S'(t)$ is:

$$
\frac{T_S'(s)}{u(s)}
=
\frac{\alpha P U_b}
{
C_p^H C_p^S s^2
+ \left[
C_p^H U_b
+ C_p^S(U_a + U_b)
\right]s
+ U_a U_b
}
$$

The denominator coefficients are:

$$
a_2 = C_p^H C_p^S
$$

$$
a_1 = C_p^H U_b + C_p^S(U_a + U_b)
$$

$$
a_0 = U_a U_b
$$

Thus the input-output behavior identifies combinations of parameters rather than the individual physical parameters themselves.

The data therefore naturally identify:

* dominant time scales
* damping characteristics
* steady-state gain

rather than:

$$
U_a, \quad U_b, \quad C_p^H, \quad C_p^S
$$

independently.

This explains why multiple parameter sets can generate nearly identical temperature trajectories.

---

## 4. Delayed Single-State Reformulation

### 4.1 Motivation

A simpler physics-based interpretation is:

The TCLab approximately behaves like a single thermal mass, but the measured sensor temperature responds to the heater input with a lag because the heater and sensor are physically separated.

Instead of explicitly modeling an unmeasured heater temperature $T_H$, we model:

* one thermal state
* plus a delayed heater effect

---

### 4.2 Single-State Energy Balance

The simplified model is:

$$
C_p \frac{dT_S}{dt}
=
U_a(T_{amb} - T_S)
+ \alpha P u_d(t)
$$

where:

$$
u_d(t) \approx u(t - \theta)
$$

is a delayed/smoothed heater signal.

Using deviation variables:

$$
T_S' = T_S - T_{amb}
$$

gives:

$$
C_p \frac{dT_S'}{dt}
=
-U_a T_S'
+ \alpha P u_d
$$

or:

$$
\frac{dT_S'}{dt}
=
-\frac{U_a}{C_p}T_S'
+ \frac{\alpha P}{C_p}u_d
$$

---

### 4.3 Equivalent Gain-Time-Constant Form

Define:

$$
\tau = \frac{C_p}{U_a}
$$

$$
K = \frac{\alpha P}{U_a}
$$

Then:

$$
\frac{dT_S'}{dt}
=
-\frac{1}{\tau}T_S'
+ \frac{K}{\tau}u_d
$$

Thus the model can be parameterized using either:

Input-output parameters

$$
K, \quad \tau, \quad \theta
$$

or

Physics-based parameters

$$
U_a, \quad C_p, \quad \theta
$$

---

## 5. Continuous-Time Delay Approximation

A pure delay is difficult to represent directly in state-space form.

Instead, approximate the delay using an $n$-stage lag chain.

Define delay states:

$$
z_1, \ z_2, \ \dots, \ z_n
$$

with:

$$
u_d = z_n
$$

and dynamics:

$$
\dot z_1 = \frac{n}{\theta}(u - z_1)
$$

$$
\dot z_i = \frac{n}{\theta}(z_{i-1} - z_i), \qquad i = 2, \dots, n
$$

This approximates:

$$
u(t - \theta)
$$

with a smooth distributed delay.

---

### 5.1 Second-Order Delay Approximation (n = 2)

Delay states

$$
\dot z_1 = \frac{2}{\theta}(u - z_1)
$$

$$
\dot z_2 = \frac{2}{\theta}(z_1 - z_2)
$$

$$
u_d = z_2
$$

#### Gain-Time-Constant Form

$$
\dot T_S' = -\frac{1}{\tau}T_S' + \frac{K}{\tau}z_2
$$

#### Physics-Based Form

$$
C_p \dot T_S' = -U_a T_S' + \alpha P z_2
$$

---

### 5.2 Third-Order Delay Approximation (n = 3)

Delay states

$$
\dot z_1 = \frac{3}{\theta}(u - z_1)
$$

$$
\dot z_2 = \frac{3}{\theta}(z_1 - z_2)
$$

$$
\dot z_3 = \frac{3}{\theta}(z_2 - z_3)
$$

$$
u_d = z_3
$$

#### Gain-Time-Constant Form

$$
\dot T_S' = -\frac{1}{\tau}T_S' + \frac{K}{\tau}z_3
$$

#### Physics-Based Form

$$
C_p \dot T_S' = -U_a T_S' + \alpha P z_3
$$

---

### 5.3 Fourth-Order Delay Approximation (n = 4)

Delay states

$$
\dot z_1 = \frac{4}{\theta}(u - z_1)
$$

$$
\dot z_2 = \frac{4}{\theta}(z_1 - z_2)
$$

$$
\dot z_3 = \frac{4}{\theta}(z_2 - z_3)
$$

$$
\dot z_4 = \frac{4}{\theta}(z_3 - z_4)
$$

$$
u_d = z_4
$$

#### Gain-Time-Constant Form

$$
\dot T_S' = -\frac{1}{\tau}T_S' + \frac{K}{\tau}z_4
$$

#### Physics-Based Form

$$
C_p \dot T_S' = -U_a T_S' + \alpha P z_4
$$

---

## 6. Interpretation of the Reformulated Model

The delayed single-state model preserves the key physical interpretation:

* $C_p$ represents thermal inertia
* $U_a$ represents heat loss to ambient
* $\theta$ represents transport/sensor lag

while avoiding explicit modeling of the unmeasured heater temperature.

Pedagogically, the interpretation is:

We assume the TCLab behaves approximately as one thermal body with one dominant temperature. The lag term compensates for the fact that the heater and sensor are physically separated and therefore do not respond instantaneously together.

Compared to the original second-order model, the reformulated model:

* reduces parameter dimension
* avoids hidden thermal states
* improves practical estimability
* preserves continuous-time physics-based structure
* supports arbitrary heater input signals $u(t)$
* remains compatible with state-space methods and optimal experimental design frameworks
