import ip_active_set as active_set


def test_enumerate_active_sets_case_count_is_2_to_the_k():
    for K in (1, 2, 3):
        out = active_set.enumerate_active_sets(V0=10.0, K=K, h_max=5.0, r_min=1.3, ar_max=1.8)
        assert out["n_cases"] == 2 ** K
        assert out["expected_n_cases"] == 2 ** K


def test_enumerate_active_sets_finds_a_feasible_case():
    for K in (1, 2, 3):
        out = active_set.enumerate_active_sets(V0=10.0, K=K, h_max=5.0, r_min=1.3, ar_max=1.8)
        assert out["found"] is not None


def test_enumerate_active_sets_case_count_grows_with_k():
    counts = []
    for K in (1, 2, 3):
        out = active_set.enumerate_active_sets(V0=10.0, K=K, h_max=5.0, r_min=1.3, ar_max=1.8)
        counts.append(out["n_cases"])
    assert counts == [2, 4, 8]
