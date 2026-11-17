# Continuous-Loop Pulsating Heat Pipe (CLPHP) Hybrid Simulation & PI-RNN Framework

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Thermodynamics: CoolProp](https://img.shields.io/badge/Thermodynamics-CoolProp%2FNIST-purple.svg)](http://www.coolprop.org/)

An advanced computational physics engine and **Physics-Informed Recurrent Neural Network (PI-RNN / LSTM / GRU)** surrogate suite for modeling, simulating, and predicting the non-linear two-phase thermal-fluid dynamics of **Continuous-Loop Pulsating Heat Pipes (CLPHPs)** in spacecraft microgravity and terrestrial high-heat-flux electronics cooling.

---

## 🌟 Key Highlights & Vision

- **High-Fidelity Lumped-Parameter ODE Solver**: Replaces computationally heavy two-phase CFD meshes with a coupled mass-spring-damper lumped-parameter model ($N$ alternating liquid slugs, vapor plugs, and dynamic tube wall thermal nodes).
- **Comprehensive Physics Formulation**:
  - **Phase Change**: Modified Hertz-Knudsen-Schrage relation at liquid-vapor menisci.
  - **Oscillating Fluid Friction**: Darcy-Weisbach friction with Sexl-Womersley oscillatory flow boundary-layer amplification ($Wo$).
  - **Dynamic Contact Angle Hysteresis**: Surface tension capillary resistance forces.
  - **Body Forces**: Variable gravity fields from spacecraft zero-g ($g=0$) to high-g launch/maneuver loads.
  - **Capillary Limit Validation**: Bond / Eötvös criterion ($Bo \le 4.0$, $D_i \le D_{crit}$).
- **Physics-Informed Neural Network (PINN) Surrogate**:
  - Sequence-to-sequence LSTM/GRU forecasting future temperatures ($T_{evap}, T_{cond}$) and thermal resistance ($R_{th}$).
  - Custom multi-objective loss enforcing thermodynamic directionality ($T_e \ge T_c$), thermal resistance definition consistency ($R_{th} \approx \Delta T / Q_{in}$), and physical non-negativity.
  - **Speed**: a batched autoregressive rollout is orders of magnitude faster than the ODE solver on the same CPU (measured, with batch=1 and batched reported separately; see Results). Accuracy is evaluated on held-out runs only and is limited by the simulator itself (see Limitations).
- **Hardware Acceleration**: Automatic hardware device selection supporting **NVIDIA CUDA**, **AMD DirectML** (`torch-directml`), **Apple MPS**, or **CPU**.
- **CoolProp & NIST Fluid Engine**: Full thermodynamic property tables for **Acetone**, **Ammonia**, **Water**, **Ethanol**, **Methanol**, and **R134a**.

---

## 📐 Mathematical Formulation

```
                               Top: Condenser Section (Cold Sink, T_sink)
                            ┌──────────────────────────────────────────────┐
                            │   (Turn 1)        (Turn 2)        (Turn N)   │
                            │   ┌──────┐        ┌──────┐        ┌──────┐   │
                            │   │ Vp_1 │        │ Vp_2 │        │ Vp_N │   │
                            └───┴──┬───┴────────┴──┬───┴────────┴──┬───┴───┘
                                   │               │               │
        Adiabatic Section          │ Liquid Slug 1 │ Liquid Slug 2 │ Liquid Slug N
        (Insulated Transport)      │ (m_1, u_1)    │ (m_2, u_2)    │ (m_N, u_N)
                                   │               │               │
                            ┌───┬──┴───┬────────┬──┴───┬────────┬──┴───┬───┐
                            │   │ Vp_1'│        │ Vp_2'│        │ Vp_N'│   │
                            │   └──────┘        └──────┘        └──────┘   │
                            │   (Turn 1)        (Turn 2)        (Turn N)   │
                            └──────────────────────────────────────────────┘
                               Bottom: Evaporator Section (Heat Load, Q_in)
```

### 1. Liquid Slug Momentum Equation
For the $i$-th liquid slug of mass $m_i = \rho_l A_c L_{slug, i}$:
$$m_i \frac{d^2x_i}{dt^2} = (P_{i-1} - P_i) A_c - F_{\text{friction}, i} - F_{\text{capillary}, i} - F_{\text{gravity}, i}$$

Where:
- $F_{\text{friction}, i} = \frac{1}{2} f_{\text{osc}} \rho_l u_i |u_i| A_c \frac{L_{\text{slug}, i}}{D_i}$, with $f_{\text{osc}} = f_{\text{Churchill}} \sqrt{1 + (Wo/4)^2}$
- $F_{\text{capillary}, i} = \pi D_i \sigma (\cos \theta_r - \cos \theta_a) \text{sign}(u_i)$
- $F_{\text{gravity}, i} = m_i g \sin(\phi) \sin(2\pi i / N)$

### 2. Vapor Plug Energy & Hertz-Knudsen Phase Change
The dynamic pressure of vapor plug $i$ obeys:
$$\frac{dP_i}{dt} = -\frac{\gamma P_i}{V_i} \left( \frac{dV_i}{dt} \right) + \frac{(\gamma - 1)}{V_i} \left( \dot{m}_{\text{net}, i} h_{fg} + \dot{Q}_{\text{wall}, i} \right)$$

Mass transfer rate via Hertz-Knudsen-Schrage relation:
$$\dot{m} = \frac{2\hat{\sigma}}{2 - \hat{\sigma}} \sqrt{\frac{M}{2\pi R_u}} \left( \frac{P_{sat}(T_{wall})}{\sqrt{T_{wall}}} - \frac{P_v}{\sqrt{T_v}} \right) A_{\text{interface}}$$

### 3. Wall Thermal Capacitance
- **Evaporator Turn $k$**:
  $$M_{w,e} c_{p,w} \frac{dT_{w,e,k}}{dt} = \frac{Q_{in}}{N} - h_{e} A_{e} (T_{w,e,k} - T_{v,k}) - \dot{m}_{evap, k} h_{fg}$$
- **Condenser Turn $k$**:
  $$M_{w,c} c_{p,w} \frac{dT_{w,c,k}}{dt} = - h_{sink} A_c (T_{w,c,k} - T_{sink}) + h_c A_c (T_{v,k} - T_{w,c,k}) + |\dot{m}_{cond, k}| h_{fg}$$

---

## 🧠 Physics-Informed Neural Network (PI-RNN) Architecture

The surrogate model replaces numerical integration during real-time flight telemetry monitoring:

```
[Inputs at Time t]                  [Recurrent Layers]                 [Multi-Step Horizon]
- Heat Load Q_in(t)              ┌──────────────────────┐               - T_evap(t+1...t+H)
- Sink Temp T_sink               │   2-Layer LSTM/GRU   │  ──────────>  - T_cond(t+1...t+H)
- Fill Ratio FR, g-load          │   Hidden Dim: 64     │               - R_th(t+1...t+H)
- Historical Window (t-L...t)    └──────────────────────┘
                                            │
                                            ▼
                               [Physics-Informed Loss]
       L_total = L_MSE + λ_1*L_TempDirection + λ_2*L_Positivity + λ_3*L_ResConsistency
```

---

## 📁 Repository Structure

```
mechproject/
├── src/
│   └── clphp/
│       ├── __init__.py
│       ├── cli.py                     # Unified Command Line Interface
│       ├── physics/
│       │   ├── fluids.py              # CoolProp & NIST analytical property provider
│       │   ├── geometry.py            # Geometric sizing & Bond criterion checks
│       │   ├── correlations.py        # Friction, contact angle, Hertz-Knudsen
│       │   └── ode_system.py          # Coupled multi-turn state-space ODEs
│       ├── simulation/
│       │   ├── solver.py              # Scipy solve_ivp transient runner
│       │   └── dataset_generator.py   # Parametric grid sweep & windowing
│       ├── models/
│       │   ├── rnn_surrogate.py       # PyTorch LSTM / GRU sequence models
│       │   ├── pinn_loss.py           # Physics loss in physical units (R_th = dT/Q_in, R_th >= 0)
│       │   ├── trainer.py             # Training loop, validation & checkpointing (seeded)
│       │   ├── evaluation.py          # Held-out per-channel metrics, persistence, rollout
│       │   └── ablation.py            # With/without physics terms, multi-seed
│       └── analysis/
│           ├── metrics.py             # Thermal resistance, FFT, startup detection
│           ├── benchmark.py           # solve_ivp vs surrogate wall-clock benchmark
│           └── plotting.py            # 4-panel diagnostic dashboards & visualizers
├── examples/
│   ├── quickstart_simulation.py       # 5-turn Acetone CLPHP step response
│   ├── train_pinn_surrogate.py        # Complete PINN surrogate training & eval
│   ├── run_experiment.py              # Honest end-to-end evaluation (split, rollout, ablation, benchmark)
│   ├── check_solver_tolerance.py      # Solver/tolerance sensitivity check
│   └── spacecraft_transient_demo.py   # Spacecraft eclipse & microgravity simulation
├── tests/
│   ├── test_fluids.py                 # Fluid property tests
│   ├── test_geometry.py               # Geometric & capillary limit tests
│   ├── test_ode.py                    # ODE assembly & derivative tests
│   ├── test_models.py                 # PyTorch model shapes & PINN loss tests
│   ├── test_split.py                  # Run-level split / normalization leakage tests
│   ├── test_evaluation.py             # Per-channel metrics, persistence, rollout
│   ├── test_ablation.py               # Seeded training & ablation harness
│   ├── test_benchmark.py              # Wall-clock benchmark
│   └── test_pipeline.py               # Full integration tests
├── php_project_plan.md                # Comprehensive Research & Engineering Roadmap
├── pyproject.toml                     # Modern package installation & metadata
└── requirements.txt                   # Dependency manifest
```

---

## 🚀 Quickstart & Installation

### 1. Local Setup
Clone or extract the repository, then install:

```bash
git clone <your-repo-url>
cd mechproject
pip install -e .
```

### 2. Run Test Suite
Run the unit and integration tests (39):

```bash
pytest tests/ -v
```

### 3. Run a Transient Simulation
Simulate a 5-turn Acetone CLPHP with 50 W heat input:

```bash
python -m clphp.cli simulate --fluid acetone --qin 50.0 --turns 5 --time 20.0 -o output/sim_dashboard.png
```

### 4. Run the Full End-to-End Demo
Quick demo (loose, *unconverged* RK23 tolerances): simulates, trains, evaluates on held-out runs vs persistence, and benchmarks. Use `examples/run_experiment.py` for the full evaluation:

```bash
python -m clphp.cli demo
```

---

## 🧪 Evaluation Methodology

The surrogate is judged only on **whole simulation runs it never saw**, and always against a naive baseline.

| Aspect | What we do |
|---|---|
| **Split** | By run, never by window. Windows are cut *after* runs are assigned. Held-out Q_in values (every fill ratio / gravity level of that Q_in goes together): **train** 15, 20, 30, 50, 60, 100, 130 W · **val** 40, 115 W (model selection) · **test-interp** 75 W · **test-extrap** 160 W (above the 130 W training maximum). |
| **Normalization** | Mean/std fitted on training runs only, then applied to val/test. |
| **Inputs** | `[Q_in, T_sink, fill ratio, g, time, T_e, T_c]`. Mean pressure and slug RMS velocity are *not* inputs, since the model does not predict them and could not feed them back during rollout. |
| **Metrics** | R², RMSE, MAE **per channel** in physical units: T_e [K], T_c [K], R_th [K/W]. Never pooled across channels. |
| **Baseline** | Persistence: next value = last observed value (R_th from the last observed T_e, T_c). |
| **One-step ahead** | True history in, first predicted step out, scored on every window of every test run. |
| **Autoregressive rollout** | Seed with the first 20 true samples, then feed the model's own T_e/T_c predictions back (exogenous Q_in, T_sink, fill, g, time come from the known schedule) until the end of the run. Scored over the first 20 steps and the full remaining run. |
| **Speed** | Wall-clock `solve_ivp` time vs surrogate rollout time, same runs, same simulated duration, same machine (CPU). Batch=1 and batched reported separately. |
| **Physics loss** | Predictions are de-normalized *inside* the loss; `R_th = (T_e − T_c)/Q_in` is rebuilt in physical units with `Q_in` read from the input features; `T_e ≥ T_c` and `R_th ≥ 0` are enforced in K and K/W. Ablated with vs without the physics terms over 3 seeds. |
| **Reproducibility** | `DatasetConfig.seed` fixes the ODE initial perturbation (local RNG), training takes a `seed` (init, dropout, batch order). Parallel dataset generation is bit-identical to serial. |

Reproduce:

```bash
python examples/run_experiment.py --profile medium --solver RK23 --rtol 1e-2 --atol 1e-3 \
    --t-sim-end 4 --jobs 12 --dataset data/clphp_dataset.npz --seeds 0 1 2 --out results
```

## 📈 Results (honest numbers)

Setup: 66 ODE runs (11 Q_in × 3 fill ratios × 2 gravity levels, 3 turns, 4 s, 0.05 s sampling), generated with **RK23 rtol=1e-2 / atol=1e-3** (fast, *not* converged; see Limitations). Window 20 steps, horizon 5, LSTM (2×64). Test sets: 6 runs each at 75 W (interpolation) and 160 W (extrapolation). Full output in [`results/`](results/).

**Main model (seed 0), held-out runs, one-step ahead vs persistence**

| Test set | Channel | Surrogate RMSE (R²) | Persistence RMSE (R²) |
|---|---|---|---|
| 75 W interp | T_e | 1.04 K (0.992) | **0.70 K (0.996)** |
| 75 W interp | T_c | 1.36 K (0.987) | **0.72 K (0.996)** |
| 75 W interp | R_th | 8.8e-4 K/W (0.969) | **4.9e-4 K/W (0.990)** |
| 160 W extrap | T_e | 8.3 K (0.857) | **1.4 K (0.996)** |
| 160 W extrap | T_c | 7.3 K (0.892) | **1.4 K (0.996)** |
| 160 W extrap | R_th | 1.8e-3 K/W (0.750) | **5.9e-4 K/W (0.975)** |

**One-step ahead, the surrogate does *not* beat persistence**: with a 0.05 s step the state barely moves, so "repeat the last value" is very hard to beat, and the surrogate predicts absolute values rather than increments. The high R² reported before the methodology fix was largely this effect plus window leakage.

**Autoregressive rollout (predictions fed back), full remaining run (61 steps = 3 s)**

| Test set | Channel | Surrogate RMSE (R²) | Persistence RMSE (R²) |
|---|---|---|---|
| 75 W interp | T_e | **1.76 K (0.979)** | 25.9 K (−3.44) |
| 75 W interp | T_c | **2.09 K (0.972)** | 26.6 K (−3.59) |
| 75 W interp | R_th | **7.8e-4 K/W (0.975)** | 1.0e-2 K/W (−3.39) |
| 160 W extrap | T_e | **11.3 K (0.761)** | 50.1 K (−3.70) |
| 160 W extrap | T_c | **10.6 K (0.795)** | 51.5 K (−3.83) |
| 160 W extrap | R_th | **1.8e-3 K/W (0.759)** | 8.7e-3 K/W (−4.75) |

Over multi-step horizons the system heats up and the surrogate clearly beats hold-last-value. Extrapolation to 160 W is markedly worse than interpolation (≈6× larger T_e error in rollout), so the model should not be trusted outside the training Q_in range.

**Physics ablation (3 seeds, mean ± std)**, with vs without the physics loss terms (selected rows):

| Test set | Evaluation | Channel | With physics | Data only |
|---|---|---|---|---|
| 160 W extrap | one-step | T_e | 7.9 ± 0.9 K | 10.6 ± 2.5 K |
| 160 W extrap | rollout (full) | T_e | 10.2 ± 1.3 K | 13.2 ± 3.8 K |
| 160 W extrap | rollout (full) | R_th | 1.85e-3 ± 5e-5 K/W | 1.67e-3 ± 2.6e-4 K/W |
| 75 W interp | rollout (full) | T_e | 1.69 ± 0.06 K | 1.84 ± 0.69 K |
| 75 W interp | rollout (20 steps) | T_e | 1.24 ± 0.14 K | **0.97 ± 0.11 K** |

The physics terms help temperature accuracy and consistency in **extrapolation** and reduce seed-to-seed variance, but the differences are within about one standard deviation for several rows, they slightly hurt R_th, and they do not help the short-horizon interpolation rollout. With 3 seeds this is suggestive, not conclusive (full table: [`results/ablation.md`](results/ablation.md)).

**Speed (CPU, same machine, 12 test runs of 4 s)**

| | Time per run | Speedup vs solver | Real-time factor (simulated s per wall s) |
|---|---|---|---|
| `solve_ivp` (RK23, rtol 1e-2) | 16.2 s | 1× | 0.25× |
| Surrogate rollout, batch = 1 | 17.7 ms | ≈ 910× | ≈ 170× |
| Surrogate rollout, batched (12 runs) | 2.0 ms | ≈ 8,000× | ≈ 1,500× |

A single forward pass takes 1.17 ms, so the former "1000 / latency ≈ 856×" figure is an *inference rate* (a real-time factor), **not** a speedup over the ODE solver. The solver baseline here uses loose RK23; the default Radau 1e-4/1e-6 is several times slower, which would inflate these speedups, so they are not a like-for-like claim about accuracy-matched solves.

## ⚠️ Limitations

- **Simulator-only validation.** Every number above compares the surrogate with *this lumped ODE model*, not with a physical pulsating heat pipe. There is **no experimental data**. The surrogate can at best be as faithful as the simulator, which is itself unvalidated against experiments.
- **The ODE solution is not converged under tolerance refinement.** For 3 turns, 75 W and 0.5 s, mean T_e at the end differed by 10–20 K between Radau 1e-5, Radau 1e-4/1e-6, Radau 1e-3, RK23 1e-4 and RK23 1e-2, and some settings give temperatures below the sink temperature (`python examples/check_solver_tolerance.py`). The dataset in this README was generated with the fast RK23 1e-2/1e-3 setting, so the "ground truth" depends on the solver settings. Radau 1e-4/1e-6 is the library default but was too slow to generate the full dataset (some runs took more than 15 minutes each). The earlier dataset (RK23 1e-2 / atol 1e-3, where atol exceeds the velocity/displacement scale) has the same problem.
- **Short, small data.** 66 runs of 4 s (the original generator used 20 s runs; converged tolerances made those impractical). Only one working fluid (acetone), one sink temperature, 3 turns, and test sets of one Q_in each (6 runs differing only in fill ratio and gravity). Uncertainty from this is not captured by the ± values, which only reflect training seeds.
- **No improvement over persistence at one step.** The surrogate is only useful for multi-step forecasting here, and degrades substantially outside the training Q_in range (160 W).
- **Physics-loss benefit is modest and uncertain** (3 seeds; see ablation).
- **Constant Q_in per window.** The physics consistency term uses the last observed Q_in for the whole horizon. Speedups compare to a loose-tolerance solver on CPU.
- **Stale figures.** The images in `output/` (including the "R² = 0.9999" comparison) and the numbers previously quoted in this README were produced before the evaluation fix, with leaked windows and X[:50] training-set scoring, and should not be cited.

## 📊 Output Artifacts & Plots (partly stale)

> The figures in `output/` predate the evaluation fix (window leakage, training-set scoring) and any metric printed on them (e.g. R² = 0.9999) must not be cited. Rely on the Results section above.

The framework automatically generates publication-ready high-resolution plots in the [`output/`](output/) directory:

1. **[`output/quickstart_dashboard.png`](output/quickstart_dashboard.png)**: 4-Panel Step Response Dashboard showing evaporator/condenser thermal response under dynamic 30W $\to$ 80W heat step, slug oscillation velocities, relative phase-space limit cycle, and 3.64 Hz FFT spectral peak.
2. **[`output/spacecraft_eclipse_transient.png`](output/spacecraft_eclipse_transient.png)**: Spacecraft Microgravity ($g = 0$) vs. Launch Flight ($g = 9.81\text{ m/s}^2$) comparison under orbital power burst (25W standby $\to$ 80W payload burst), illustrating temperature margin limits (65°C).
3. **[`output/surrogate_test_comparison.png`](output/surrogate_test_comparison.png)**: 3-Panel comparison of Ground Truth ODE trajectories vs. PI-RNN Neural Surrogate forecasts for $T_e, T_c$, and $R_{th}$ ($R^2 = 0.9999$).
4. **[`output/pinn_training_curve.png`](output/pinn_training_curve.png)**: Log-scale training history curve illustrating Total PINN Loss and Data MSE convergence over 25 epochs.
5. **[`output/simulation_dashboard.png`](output/simulation_dashboard.png)**: End-to-end multi-turn CLPHP diagnostic dashboard generated via CLI (`python -m clphp.cli demo`).
6. **[`output/training_loss.png`](output/training_loss.png)**: CLI demo surrogate model loss convergence plot.

---

## 📜 Citation & References
1. Zhang, Y., & Faghri, A. (2008). *Advances and Unresolved Issues in Pulsating Heat Pipes*. Heat Transfer Engineering, 29(1), 20-44.
2. Dobson, R. T. (2004). *An open-oscillating heat pipe mathematical model*. International Journal of Heat and Mass Transfer, 47(12-13), 2703-2710.
3. Mameli, M., et al. (2014). *Lumped parameter model for multi-turn closed loop pulsating heat pipes*. Applied Thermal Engineering, 65(1-2), 485-494.
4. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks*. Journal of Computational Physics, 378, 686-707.
