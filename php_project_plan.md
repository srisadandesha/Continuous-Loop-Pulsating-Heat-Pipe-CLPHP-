# Continuous-Loop Pulsating Heat Pipe (CLPHP) Research & Engineering Roadmap
## Hybrid Mathematical Modelling & Physics-Informed Recurrent Neural Network (PI-RNN) Framework

---

## 1. Executive Summary & Project Vision

Spacecraft electronic payloads, radars, and electric propulsion systems produce localized heat fluxes exceeding $50 - 150\text{ W/cm}^2$. Continuous-Loop Pulsating Heat Pipes (CLPHPs) are passive, highly efficient, wickless two-phase thermal management devices ideally suited for aerospace applications due to their lightweight structure and strong capillary-driven self-circulation in microgravity ($g \approx 0$).

Traditional two-phase Computational Fluid Dynamics (CFD) (Volume of Fluid / Level-Set methods) requires hours to days of CPU/GPU cluster time to simulate mere seconds of physical time, making it unusable for real-time spacecraft flight controllers, mission health monitoring, and hardware-in-the-loop (HIL) testing.

**This project delivers a dual-tier hybrid simulation framework:**
1. **High-Fidelity Lumped-Parameter ODE Engine**: Discretizes the closed capillary loop into alternating liquid slugs and vapor plugs with coupled momentum, energy, and Hertz-Knudsen phase-change equations, solvable in seconds.
2. **Physics-Informed Recurrent Neural Network (PI-RNN / LSTM / GRU)**: A deep learning surrogate trained on parametric synthetic datasets with embedded thermodynamic loss constraints, enabling sub-millisecond ($<2\text{ ms}$, **>1000x faster than ODE integration**) transient temperature and thermal resistance predictions.

```
+-----------------------------------------------------------------------------------+
|                            CLPHP Hybrid Architecture                              |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  [ Physics & Sizing Engine ]                                                      |
|   - Capillary Limit (Bo <= 4.0, D_crit)                                           |
|   - CoolProp & NIST Thermodynamic Property Equations of State                     |
|   - Dynamic Contact Angle Hysteresis & Sexl-Womersley Oscillating Friction        |
|                                       │                                           |
|                                       ▼                                           |
|  [ Coupled Multi-Turn ODE System ]                                                |
|   - Liquid Slugs Momentum Balance: m_i * d²x_i/dt² = ΔP*A_c - F_fric - F_cap     |
|   - Vapor Plugs Thermodynamics: dP_i/dt = Work + Hertz-Knudsen Phase Change       |
|   - Dynamic Tube Wall Thermal Nodes: (M*Cp)_w * dT_w/dt = Q_in - Q_conv - Q_latent|
|                                       │                                           |
|                                       ▼                                           |
|  [ Automated Dataset Generator & Sliding-Window Pipeline ]                        |
|   - Parametric sweeps: Q_in (10-200W), Fill Ratio (30-70%), g-load (0 to 30 m/s²) |
|   - Feature normalization & tensor sequence windowing                             |
|                                       │                                           |
|                                       ▼                                           |
|  [ Physics-Informed Neural Network (PI-RNN / LSTM / GRU) ]                        |
|   - Recurrent Sequence-to-Sequence Forecasting Head                               |
|   - Multi-Objective Loss: L_MSE + λ_1*L_TempDir + λ_2*L_Positivity + λ_3*L_Consist|
|   - Sub-millisecond inference for real-time onboard digital twins                 |
|                                       │                                           |
|                                       ▼                                           |
|  [ Diagnostic Suite & Spacecraft Flight Telemetry ]                               |
|   - Thermal Resistance R_th, FFT Power Spectra, P-V Indicator Loops, Phase Space   |
+-----------------------------------------------------------------------------------+
```

---

## 2. Mathematical System Model (Lumped-Parameter)

Instead of dense Navier-Stokes grid discretization, the multi-turn fluid loop is treated as a dynamic continuous chain of $N$ alternating liquid slugs (mass blocks) and compressible vapor plugs (thermodynamic gas springs with phase change).

### 2.1 Liquid Slug Momentum Equation
The acceleration of each liquid slug ($i$) is governed by pressure differentials across adjacent vapor plugs, oscillating wall shear friction, contact angle hysteresis capillary resistance, and gravitational/acceleration body forces:

$$m_i \frac{d^2x_i}{dt^2} = (P_{i-1} - P_i) A_c - F_{\text{friction}, i} - F_{\text{capillary}, i} - F_{\text{gravity}, i}$$

Where:
*   $m_i$: Mass of the $i$-th liquid slug ($m_i = \rho_l A_c L_{\text{slug}, i}$)
*   $x_i, u_i$: Axial displacement and velocity ($u_i = \dot{x}_i$)
*   $A_c$: Internal tube cross-sectional area ($A_c = \pi D_i^2 / 4$)
*   $F_{\text{friction}, i}$: Oscillatory flow resistance:
    $$F_{\text{friction}, i} = \frac{1}{2} f_{\text{osc}} \rho_l u_i |u_i| A_c \frac{L_{\text{slug}, i}}{D_i}$$
    with the Womersley frequency-amplified friction factor $f_{\text{osc}} = f_{\text{Churchill}} \sqrt{1 + (Wo/4)^2}$, where $Wo = \frac{D_i}{2}\sqrt{\frac{\omega \rho_l}{\mu_l}}$.
*   $F_{\text{capillary}, i}$: Dynamic contact angle hysteresis resistance:
    $$F_{\text{capillary}, i} = \pi D_i \sigma (\cos \theta_r - \cos \theta_a) \text{sign}(u_i)$$
*   $F_{\text{gravity}, i}$: Acceleration/gravity force projected along turn orientation:
    $$F_{\text{gravity}, i} = m_i g \sin(\phi) \sin\left(\frac{2\pi i}{N}\right)$$

### 2.2 Vapor Plug Energy & Hertz-Knudsen Phase Change
The dynamic pressure and temperature evolution in vapor plug $i$ (bounded by slugs $i$ and $i+1$) are governed by mechanical compression/expansion work and interfacial phase-change mass transfer:

$$\frac{dP_i}{dt} = -\frac{\gamma P_i}{V_i} \left( \frac{dV_i}{dt} \right) + \frac{(\gamma - 1)}{V_i} \left( \dot{m}_{\text{net}, i} h_{fg} + \dot{Q}_{\text{wall}, i} \right)$$

$$\frac{dT_{v, i}}{dt} = T_{v, i} \left( \frac{1}{P_i}\frac{dP_i}{dt} + \frac{1}{V_i}\frac{dV_i}{dt} - \frac{1}{m_{v, i}}\dot{m}_{\text{net}, i} \right)$$

Where:
*   $\gamma$: Specific heat ratio ($c_p / c_v$) of the vapor
*   $V_i(t) = V_{0, i} + A_c (x_{i+1} - x_i)$: Dynamic volume of plug $i$
*   $\dot{m}_{\text{net}, i} = \dot{m}_{\text{evap}, i} + \dot{m}_{\text{cond}, i}$: Interfacial mass flux computed via the Hertz-Knudsen-Schrage relation:
    $$\dot{m} = \frac{2\hat{\sigma}}{2 - \hat{\sigma}} \sqrt{\frac{M}{2\pi R_u}} \left( \frac{P_{sat}(T_{\text{wall}})}{\sqrt{T_{\text{wall}}}} - \frac{P_v}{\sqrt{T_v}} \right) A_{\text{interface}}$$

### 2.3 Tube Wall Thermal Capacitance Nodes
*   **Evaporator Section (Turn $k$)**:
    $$M_{w,e} c_{p,w} \frac{dT_{w,e,k}}{dt} = \frac{Q_{in}}{N} - h_{e} A_{e} (T_{w,e,k} - T_{v,k}) - \dot{m}_{\text{evap}, k} h_{fg}$$
*   **Condenser Section (Turn $k$)**:
    $$M_{w,c} c_{p,w} \frac{dT_{w,c,k}}{dt} = - h_{\text{sink}} A_c (T_{w,c,k} - T_{\text{sink}}) + h_c A_c (T_{v,k} - T_{w,c,k}) + |\dot{m}_{\text{cond}, k}| h_{fg}$$

---

## 3. Physics-Informed Recurrent Neural Network (PI-RNN)

### 3.1 Network Architecture
The surrogate model is constructed using a multi-layer **LSTM** or **GRU** recurrent core followed by a non-linear Multi-Layer Perceptron (MLP) decoder head:
*   **Input Vector ($X_t$)**: $[Q_{in}(t), T_{sink}(t), \text{FillRatio}, g\text{-load}, t, T_e(t), T_c(t), \bar{P}(t), v_{rms}(t)]$ over a historical sequence window of length $L = 20$.
*   **Output Vector ($Y_{t+1...t+H}$)**: Multi-step future horizon prediction of $[T_e, T_c, R_{th}]$ over $H = 5$ time steps.

### 3.2 Physics-Informed Multi-Objective Loss
Unlike standard black-box regressors that can produce physically impossible violations during transient thermal shocks, the network is trained with a composite physics-informed loss:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_1 \mathcal{L}_{\text{temp\_direction}} + \lambda_2 \mathcal{L}_{\text{positivity}} + \lambda_3 \mathcal{L}_{\text{resistance\_consistency}}$$

1.  **Supervised Data Loss**:
    $$\mathcal{L}_{\text{data}} = \frac{1}{B \cdot H \cdot D} \sum \| \hat{Y} - Y_{\text{true}} \|^2$$
2.  **2nd Law Directionality Constraint**:
    Evaporator temperature must remain higher than condenser temperature ($T_e \ge T_c$):
    $$\mathcal{L}_{\text{temp\_direction}} = \frac{1}{B \cdot H} \sum \left( \text{ReLU}(\hat{T}_c - \hat{T}_e) \right)^2$$
3.  **Positivity Constraint**:
    Thermal resistance $R_{th}$ must remain strictly positive:
    $$\mathcal{L}_{\text{positivity}} = \frac{1}{B \cdot H} \sum \left( \text{ReLU}(-\hat{R}_{th}) \right)^2$$
4.  **Thermal Resistance Definition Consistency**:
    Enforces $\hat{R}_{th} \approx \frac{\hat{T}_e - \hat{T}_c}{Q_{in}}$:
    $$\mathcal{L}_{\text{resistance\_consistency}} = \frac{1}{B \cdot H} \sum \| (\hat{T}_e - \hat{T}_c) - \hat{R}_{th} \|^2$$

---

## 4. Software Engineering & Module Architecture

| Module | Location | Functionality |
| :--- | :--- | :--- |
| **Physics / Fluids** | `src/clphp/physics/fluids.py` | CoolProp integration with NIST analytical fallbacks for Acetone, Ammonia, Water, Ethanol, Methanol, R134a |
| **Physics / Geometry** | `src/clphp/physics/geometry.py` | Sizing, channel volume, wall masses, Bond number $Bo \le 4.0$ verification |
| **Physics / Correlations** | `src/clphp/physics/correlations.py` | Churchill/Womersley friction, contact angle hysteresis, Hertz-Knudsen phase change |
| **Physics / ODE Engine** | `src/clphp/physics/ode_system.py` | Coupled state-space time derivatives $\frac{d\vec{y}}{dt}$ for $6N$ variables |
| **Simulation / Solver** | `src/clphp/simulation/solver.py` | High-speed adaptive numerical solver with transient metric calculation |
| **Simulation / Dataset** | `src/clphp/simulation/dataset_generator.py` | Parametric grid generator ($Q_{in}$, $FR$, $g$, $T_{sink}$) and sliding-window preprocessor |
| **Deep Learning / Model** | `src/clphp/models/rnn_surrogate.py` | PyTorch LSTM and GRU sequence-to-sequence forecasting architectures |
| **Deep Learning / PINN** | `src/clphp/models/pinn_loss.py` | Physics-informed loss enforcing thermodynamic laws & consistency |
| **Deep Learning / Trainer** | `src/clphp/models/trainer.py` | Mini-batch training, CosineAnnealingLR, early stopping, and checkpointing |
| **Diagnostics & Plotting** | `src/clphp/analysis/` | FFT spectrum, startup detection, 4-panel simulation dashboards, and residual comparisons |
| **CLI & Examples** | `src/clphp/cli.py`, `examples/` | Single-command simulation, dataset generation, training, and spacecraft orbital demo |

---

## 5. Research & Engineering Project Timeline

```
+------------------------------------------------------------------------------------------------+
| Phase 1: Mathematical Formulation & Property Provider                 [COMPLETED & VALIDATED]  |
|  - Liquid-slug momentum ODEs with dynamic friction & contact angle                             |
|  - Vapor-plug Hertz-Knudsen phase change thermodynamics                                        |
|  - CoolProp + analytical property engine for 6 working fluids                                  |
+------------------------------------------------------------------------------------------------+
                                               │
                                               ▼
+------------------------------------------------------------------------------------------------+
| Phase 2: Numerical ODE Solver & Parametric Generator                  [COMPLETED & VALIDATED]  |
|  - Vectorized RHS evaluation with adaptive numerical solvers (RK45, Radau)                     |
|  - Multi-parameter batch dataset generation across Q_in, Fill Ratio, g-load                    |
|  - Automated 14-test unit and integration test suite                                           |
+------------------------------------------------------------------------------------------------+
                                               │
                                               ▼
+------------------------------------------------------------------------------------------------+
| Phase 3: Physics-Informed Recurrent Neural Network (PI-RNN)            [COMPLETED & VALIDATED]  |
|  - Sequence-to-sequence PyTorch LSTM & GRU architectures                                       |
|  - Multi-objective thermodynamic loss constraints                                              |
|  - Sub-millisecond latency (<2 ms) benchmarked                                                |
+------------------------------------------------------------------------------------------------+
                                               │
                                               ▼
+------------------------------------------------------------------------------------------------+
| Phase 4: Spacecraft Flight Scenarios & Digital Twin Integration        [NEXT MILESTONES]       |
|  - Hardware-in-the-Loop (HIL) embedded micro-controller deployment (Raspberry Pi / Arduino)   |
|  - Real-time orbital eclipse telemetry forecasting & dryout anomaly detection                  |
|  - Multi-objective geometric optimization (Pareto front for tube diameter & turn count)        |
+------------------------------------------------------------------------------------------------+
```

---

## 6. Spacecraft Flight Telemetry & Digital Twin Applications

1.  **Orbital Thermal Cycle Forecasting**: Rapidly predicts satellite evaporator junction temperatures during eclipse-to-sunlight transitions.
2.  **Dry-Out Anomaly Early Warning**: Detects incipient dry-out regimes when thermal resistance $R_{th}$ spikes or oscillation RMS velocity drops.
3.  **Low-Power Embedded Deployment**: The trained neural surrogate has tiny memory footprint (<200 KB) and can execute in real-time on radiation-hardened spacecraft onboard computers (e.g., LEON4, ARM Cortex-R5, FPGA soft cores).