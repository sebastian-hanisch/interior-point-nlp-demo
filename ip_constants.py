"""Regler-Grenzen, feste Annahmen, gemessene Werte und Presets."""

V0_MIN, V0_MAX, V0_DEFAULT = 1.0, 100.0, 10.0
H_MAX_MIN, H_MAX_MAX, H_MAX_DEFAULT = 0.5, 6.0, 5.0
K_MIN, K_MAX, K_DEFAULT = 1, 3, 3
R_MIN_DEFAULT = 1.3
AR_MAX_DEFAULT = 1.8

PRESETS = {
    "hoehenlimit_nicht_bindend": dict(
        label="Höhenlimit nicht bindend",
        V0=10.0, h_max=5.0,
        help="Bei K=1 identisch zur klassischen Lösung; ab K=2 bindet stattdessen der "
             "Mindestradius (r*=1,1675 < r_min=1,3).",
    ),
    "hoehenlimit_bindet": dict(
        label="Höhenlimit bindet",
        V0=10.0, h_max=1.5,
        help="Das Höhenlimit bindet bei jedem K — Mindestradius und Aspect-Ratio-Limit bleiben "
             "hier stets inaktiv (Kontrollfall: K ändert die Lösung nicht).",
    ),
}
