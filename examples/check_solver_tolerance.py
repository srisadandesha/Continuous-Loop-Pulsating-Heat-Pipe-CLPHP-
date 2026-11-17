"""
Tolerance / solver sensitivity check for the CLPHP ODE simulator.

Runs the same case with several solvers and tolerances and prints how far the
mean evaporator/condenser temperature trajectories move relative to a
reference run. Use it before trusting any dataset generated with a given
setting:   python examples/check_solver_tolerance.py --time 0.5 --turns 3

Observed on the shipped model (3 turns, Q_in=75 W, 0.5 s): the trajectories
change by 10-20 K between tolerance settings, i.e. the ODE solution is NOT
converged under tolerance refinement at these settings (see README).
"""

import argparse
import time

import numpy as np

from clphp.physics.geometry import PHPGeometry
from clphp.simulation.solver import simulate_clphp

CASES = [
    ("Radau", 1e-5, 1e-7),   # reference
    ("Radau", 1e-4, 1e-6),   # default
    ("Radau", 1e-3, 1e-5),
    ("RK23", 1e-4, 1e-6),
    ("RK23", 1e-2, 1e-3),    # legacy dataset setting
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", type=float, default=0.5)
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--qin", type=float, default=75.0)
    args = ap.parse_args()

    geo = PHPGeometry(n_turns=args.turns, d_inner=0.002, fill_ratio=0.5)
    ref = None
    for method, rtol, atol in CASES:
        t0 = time.perf_counter()
        r = simulate_clphp(geometry=geo, q_in=args.qin, t_span=(0.0, args.time),
                           t_eval_step=args.time / 10, solver_method=method, rtol=rtol, atol=atol)
        wall = time.perf_counter() - t0
        if ref is None:
            ref = r
        d_te = np.max(np.abs(r.t_evap_mean - ref.t_evap_mean))
        d_tc = np.max(np.abs(r.t_cond_mean - ref.t_cond_mean))
        print(f"{method:6s} rtol={rtol:g} atol={atol:g}  wall={wall:6.1f}s  T_e(end)={r.t_evap_mean[-1]:.2f} K  "
              f"max|dT_e|={d_te:.2f} K  max|dT_c|={d_tc:.2f} K  (vs first row)")


if __name__ == "__main__":
    main()
