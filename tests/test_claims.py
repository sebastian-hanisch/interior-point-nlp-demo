"""Jede Zahl aus README.md und App wird hier aus den echten Auswertungsfunktionen neu berechnet."""
import ip_evaluation as ev


def test_claim_reference_convergence_for_k1():
    rows = ev.reference_convergence_check()
    for row in rows:
        assert row["err"] < 1e-6


def test_claim_case_count_is_exactly_2_to_the_k():
    rows = ev.combinatorics_scaling_check()
    assert [r["n_cases"] for r in rows] == [2, 4, 8]


def test_claim_enumeration_is_not_more_expensive_than_interior_point_at_k_leq_3():
    """Ehrlicher Befund (Plan-Korrektur): bei der realistischen Portfolio-Groesse ist die
    Enumeration NICHT teurer als das Innere-Punkte-Verfahren."""
    rows = ev.combinatorics_scaling_check()
    for row in rows:
        assert row["enum_total_iter"] <= row["ip_total_iter"]


def test_claim_crossover_exists_in_synthetic_extension():
    rows = ev.synthetic_scaling_check(K_values=(1, 2, 3, 4, 5, 6, 7, 8))
    below = [r for r in rows if r["enum_total_iter"] <= r["ip_total_iter"]]
    above = [r for r in rows if r["enum_total_iter"] > r["ip_total_iter"]]
    assert len(below) > 0
    assert len(above) > 0
    assert max(r["K"] for r in below) < min(r["K"] for r in above)


def test_claim_complementarity_converges():
    out = ev.complementarity_check()
    assert out["final_gap"] < 1e-6
    assert out["monotone_decreasing"]


def test_claim_gradient_check_below_1e_minus_6():
    out = ev.gradient_check()
    assert out["f_max_rel_err"] < 1e-6
    assert out["g1_max_rel_err"] < 1e-6
