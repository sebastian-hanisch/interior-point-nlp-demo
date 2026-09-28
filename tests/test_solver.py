import numpy as np

import ip_reference as ref
import ip_solver as solver


def test_interior_point_converges_to_reference_k1_inactive():
    reference = ref.reference_solution(V0=10.0, h_max=5.0)
    out = solver.interior_point_solve(10.0, 1, 5.0, r_min=0.01, ar_max=100.0)
    assert abs(out.r - reference["r"]) < 1e-6
    assert abs(out.h - reference["h"]) < 1e-6


def test_interior_point_converges_to_reference_k1_active():
    reference = ref.reference_solution(V0=10.0, h_max=1.5)
    out = solver.interior_point_solve(10.0, 1, 1.5, r_min=0.01, ar_max=100.0)
    assert abs(out.r - reference["r"]) < 1e-6
    assert abs(out.h - reference["h"]) < 1e-6


def test_slacks_and_multipliers_stay_positive():
    out = solver.interior_point_solve(10.0, 3, 5.0, r_min=1.3, ar_max=1.8)
    assert np.all(out.s > 0)
    assert np.all(out.mu > 0)


def test_complementarity_gap_shrinks_over_outer_stages():
    out = solver.interior_point_solve(10.0, 3, 5.0, r_min=1.3, ar_max=1.8)
    gaps = [row["comp_gap"] for row in out.history]
    assert gaps[-1] < gaps[0]
    assert gaps[-1] < 1e-6
