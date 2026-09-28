import numpy as np

import ip_functions as fn


def test_surface_area_matches_formula():
    x = np.array([2.0, 3.0])
    assert fn.surface_area(x) == 2 * np.pi * 4 + 2 * np.pi * 2 * 3


def test_gradient_matches_finite_differences():
    eps = 1e-6
    x = np.array([1.7, 2.3])
    analytic = fn.grad_surface_area(x)
    for i in range(2):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        numeric = (fn.surface_area(xp) - fn.surface_area(xm)) / (2 * eps)
        assert abs(analytic[i] - numeric) < 1e-4


def test_make_inequalities_returns_correct_count():
    assert len(fn.make_inequalities(1, 5.0, 1.3, 1.8)) == 1
    assert len(fn.make_inequalities(2, 5.0, 1.3, 1.8)) == 2
    assert len(fn.make_inequalities(3, 5.0, 1.3, 1.8)) == 3


def test_height_limit_sign():
    assert fn.height_limit(np.array([1.0, 2.0]), 3.0) < 0
    assert fn.height_limit(np.array([1.0, 4.0]), 3.0) > 0


def test_min_radius_sign():
    assert fn.min_radius(np.array([2.0, 1.0]), 1.3) < 0
    assert fn.min_radius(np.array([1.0, 1.0]), 1.3) > 0


def test_max_aspect_ratio_sign():
    assert fn.max_aspect_ratio(np.array([2.0, 2.0]), 1.8) < 0
    assert fn.max_aspect_ratio(np.array([1.0, 3.0]), 1.8) > 0
