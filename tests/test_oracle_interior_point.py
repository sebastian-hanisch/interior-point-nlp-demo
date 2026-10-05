"""Unabhängiges Orakel für Innere-Punkte-Verfahren und Enumeration: das Problem lässt sich über
h = V0/(pi r^2) auf EINE Variable r reduzieren; dort ist f(r) = 2 pi r^2 + 2 V0/r konvex, und alle
drei Ungleichungen werden zu Untergrenzen für r. Damit gilt r* = max(r_frei, größte Untergrenze).
Multiplikatoren folgen aus der Stationarität per kleinster Quadrate (nur numpy)."""
import numpy as np
import pytest

import ip_active_set as active_set
import ip_functions as fn
import ip_solver as solver

PI = np.pi

# (V0, K, h_max, r_min, ar_max): App-Bereich, Randwerte und Gleichstand zweier Grenzen
CASES = [
    (10.0, 1, 5.0, 1.3, 1.8), (10.0, 3, 5.0, 1.3, 1.8), (10.0, 2, 5.0, 1.3, 1.8),
    (10.0, 3, 1.5, 1.3, 1.8), (1.0, 3, 0.5, 1.3, 1.8), (100.0, 3, 6.0, 1.3, 1.8),
    (100.0, 1, 0.5, 1.3, 1.8), (37.0, 3, 3.1, 1.3, 1.8), (62.0, 2, 4.4, 1.3, 1.8),
    (45.0, 3, 6.0, 1.3, 1.8),
]


def oracle(V0, K, h_max, r_min, ar_max):
    r_free = (V0 / (2 * PI)) ** (1 / 3)
    lows = [np.sqrt(V0 / (PI * h_max)), r_min, (V0 / (PI * ar_max)) ** (1 / 3)][:K]
    r = max(r_free, max(lows))
    h = V0 / (PI * r * r)
    g = [h - h_max, r_min - r, h - ar_max * r][:K]
    return r, h, g


def oracle_multipliers(V0, K, r, h, g, ar_max):
    x = np.array([r, h])
    grads = [fn.grad_height_limit(), fn.grad_min_radius(), fn.grad_max_aspect_ratio(ar_max)]
    active = [j for j in range(K) if abs(g[j]) < 1e-9]
    A = np.column_stack([fn.grad_volume_constraint(x)] + [grads[j] for j in active])
    sol, *_ = np.linalg.lstsq(A, -fn.grad_surface_area(x), rcond=None)
    mu = np.zeros(K)
    for idx, j in enumerate(active):
        mu[j] = sol[1 + idx]
    return float(sol[0]), mu, active


def test_oracle_itself_beats_grid_on_the_curve():
    # Handprüfung des Orakels: kein zulässiger Punkt auf der Volumenkurve ist besser
    V0, K, h_max, r_min, ar_max = 10.0, 3, 5.0, 1.3, 1.8
    r, h, g = oracle(V0, K, h_max, r_min, ar_max)
    assert (r, round(h, 4)) == (1.3, 1.8835)
    f_star = fn.surface_area(np.array([r, h]))
    for rr in np.linspace(0.3, 6, 400):
        hh = V0 / (PI * rr * rr)
        if hh <= h_max and rr >= r_min and hh <= ar_max * rr:
            assert fn.surface_area(np.array([rr, hh])) >= f_star - 1e-9


@pytest.mark.parametrize("case", CASES)
def test_interior_point_matches_reduced_problem_and_multipliers(case):
    V0, K, h_max, r_min, ar_max = case
    r, h, g = oracle(*case)
    sol = solver.interior_point_solve(V0, K, h_max, r_min, ar_max)
    assert abs(sol.r - r) < 1e-6 and abs(sol.h - h) < 1e-6
    lam, mu, active = oracle_multipliers(V0, K, r, h, g, ar_max)
    if len(active) <= 1:  # bei zwei gleichzeitig aktiven Grenzen sind die Multiplikatoren nicht eindeutig
        assert abs(sol.lam - lam) < 1e-5 * max(1.0, abs(lam))
        assert np.max(np.abs(sol.mu - mu)) < 1e-5 * max(1.0, float(np.max(mu)))


@pytest.mark.parametrize("case", CASES)
def test_enumeration_finds_the_oracle_solution_and_active_set(case):
    V0, K, h_max, r_min, ar_max = case
    r, h, g = oracle(*case)
    out = active_set.enumerate_active_sets(V0, K, h_max, r_min, ar_max)
    assert out["n_cases"] == 2 ** K
    assert out["found"] is not None
    assert abs(out["found"]["r"] - r) < 1e-6 and abs(out["found"]["h"] - h) < 1e-6
    active = [j for j in range(K) if abs(g[j]) < 1e-9]
    if len(active) <= 1:
        assert list(out["found"]["A"]) == active
