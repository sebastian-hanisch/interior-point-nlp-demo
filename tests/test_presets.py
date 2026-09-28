import ip_constants as C
import ip_evaluation as ev


def test_all_presets_have_valid_settings():
    for key, preset in C.PRESETS.items():
        assert C.V0_MIN <= preset["V0"] <= C.V0_MAX
        assert C.H_MAX_MIN <= preset["h_max"] <= C.H_MAX_MAX


def test_preset_hoehenlimit_nicht_bindend_switches_active_constraint_with_k():
    p = C.PRESETS["hoehenlimit_nicht_bindend"]
    out_k1 = ev.analyse(ev.Settings(V0=p["V0"], h_max=p["h_max"], K=1))
    out_k2 = ev.analyse(ev.Settings(V0=p["V0"], h_max=p["h_max"], K=2))
    assert out_k1["solution"].r < 1.3
    assert abs(out_k2["solution"].r - 1.3) < 1e-4


def test_preset_hoehenlimit_bindet_is_unaffected_by_k():
    p = C.PRESETS["hoehenlimit_bindet"]
    out_k1 = ev.analyse(ev.Settings(V0=p["V0"], h_max=p["h_max"], K=1))
    out_k3 = ev.analyse(ev.Settings(V0=p["V0"], h_max=p["h_max"], K=3))
    assert abs(out_k1["solution"].r - out_k3["solution"].r) < 1e-4
    assert abs(out_k1["solution"].h - out_k3["solution"].h) < 1e-4
