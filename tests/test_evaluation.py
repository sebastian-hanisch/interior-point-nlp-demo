import ip_evaluation as ev


def test_analyse_returns_all_expected_keys():
    out = ev.analyse(ev.Settings(V0=10.0, h_max=5.0, K=2))
    assert set(out.keys()) == {"solution", "enum", "f_val"}


def test_reference_convergence_check_both_presets():
    rows = ev.reference_convergence_check()
    for row in rows:
        assert row["err"] < 1e-6


def test_combinatorics_scaling_check_case_counts():
    rows = ev.combinatorics_scaling_check()
    assert [r["n_cases"] for r in rows] == [2, 4, 8]
    for row in rows:
        assert row["agreement_err"] < 1e-4


def test_synthetic_scaling_check_shows_exponential_case_growth():
    rows = ev.synthetic_scaling_check(K_values=(1, 2, 3, 4, 5))
    n_cases = [r["n_cases"] for r in rows]
    assert n_cases == [2, 4, 8, 16, 32]


def test_synthetic_scaling_check_finds_a_crossover():
    rows = ev.synthetic_scaling_check(K_values=(1, 2, 3, 4, 5, 6, 7, 8))
    enum_final = rows[-1]["enum_total_iter"]
    ip_final = rows[-1]["ip_total_iter"]
    assert enum_final > ip_final


def test_complementarity_check_converges():
    out = ev.complementarity_check()
    assert out["monotone_decreasing"]
    assert out["final_gap"] < 1e-6


def test_gradient_check_below_threshold():
    out = ev.gradient_check()
    assert out["f_max_rel_err"] < 1e-6
    assert out["g1_max_rel_err"] < 1e-6
