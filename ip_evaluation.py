"""Kennzahlen: Settings-Dataclass, analyse()-Einstiegspunkt, Referenz-Konvergenz (K=1),
Kombinatorik-Skalierung an der echten, physikalisch motivierten Vehikel-B-Erweiterung (K=1,2,3),
eine EXPLIZIT synthetische Skalierungs-Messreihe (K=1..8, ueber Vehikel B hinaus, nur um den in
der Literatur bekannten Trend selbst sichtbar zu machen), Komplementaritaets-Check,
Gradienten-Check."""
from dataclasses import dataclass

import numpy as np

import ip_active_set as active_set
import ip_functions as fn
import ip_reference as ref
import ip_solver as solver


@dataclass(frozen=True)
class Settings:
    V0: float
    h_max: float
    K: int
    r_min: float = 1.3
    ar_max: float = 1.8


def analyse(settings: Settings) -> dict:
    solution = solver.interior_point_solve(settings.V0, settings.K, settings.h_max,
                                           settings.r_min, settings.ar_max)
    enum = active_set.enumerate_active_sets(settings.V0, settings.K, settings.h_max,
                                            settings.r_min, settings.ar_max)
    f_val = fn.surface_area(np.array([solution.r, solution.h]))
    return {"solution": solution, "enum": enum, "f_val": f_val}


def reference_convergence_check(V0: float = 10.0, h_max_values=(5.0, 1.5)) -> list:
    """Fuer K=1 (nur das Hoehenlimit als Ungleichung, identisch zu Stueck 4-6) muss das Innere-
    Punkte-Verfahren zur selben Referenzloesung konvergieren."""
    rows = []
    for h_max in h_max_values:
        reference = ref.reference_solution(V0, h_max)
        solution = solver.interior_point_solve(V0, 1, h_max, r_min=0.01, ar_max=100.0)
        err = float(np.hypot(solution.r - reference["r"], solution.h - reference["h"]))
        rows.append({"h_max": h_max, "case": reference["case"], "err": err,
                    "total_inner_iter": solution.total_inner_iter})
    return rows


def combinatorics_scaling_check(V0: float = 10.0, h_max: float = 5.0, r_min: float = 1.3,
                                ar_max: float = 1.8, K_values=(1, 2, 3)) -> list:
    """Zentrale Messung an der ECHTEN, physikalisch motivierten Vehikel-B-Erweiterung: waechst
    die Fallzahl der aktiven-Menge-Enumeration tatsaechlich wie 2^K? Und ist das Innere-Punkte-
    Verfahren bei dieser (kleinen) Portfolio-Groesse ueberhaupt schon guenstiger - oder eine
    ehrliche Plan-Korrektur noetig?"""
    rows = []
    for K in K_values:
        enum = active_set.enumerate_active_sets(V0, K, h_max, r_min, ar_max)
        ip_solution = solver.interior_point_solve(V0, K, h_max, r_min, ar_max)
        err = 0.0
        if enum["found"] is not None:
            err = float(np.hypot(ip_solution.r - enum["found"]["r"],
                                 ip_solution.h - enum["found"]["h"]))
        rows.append({"K": K, "n_cases": enum["n_cases"], "expected_n_cases": enum["expected_n_cases"],
                    "enum_total_iter": enum["total_iter"], "ip_total_iter": ip_solution.total_inner_iter,
                    "ip_cheaper": ip_solution.total_inner_iter < enum["total_iter"],
                    "agreement_err": err})
    return rows


def _synthetic_inequalities(K: int, h_max: float) -> list:
    """NUR fuer die Skalierungs-Messreihe (NICHT Teil von Vehikel B): K Hoehenlimit-artige
    Ungleichungen mit wachsend lockereren Grenzen (h_max, h_max+0.5, h_max+1.0, ...) - nur die
    straffste bindet je, die wahre Loesung aendert sich dadurch NICHT. Macht die 2^K-Kombinatorik
    selbst sichtbar, unabhaengig vom physikalischen Modell."""
    specs = []
    for i in range(K):
        limit = h_max + i * 0.5
        specs.append((lambda x, limit=limit: x[1] - limit, lambda x: np.array([0.0, 1.0])))
    return specs


def synthetic_scaling_check(V0: float = 10.0, h_max: float = 5.0,
                            K_values=(1, 2, 3, 4, 5, 6, 7, 8)) -> list:
    """Explizit ueber Vehikel B hinaus (kuenstliche, redundante Hoehenlimit-Kopien statt
    physikalisch verschiedener Nebenbedingungen) - zeigt den in der Literatur behaupteten
    Skalierungsvorteil des Innere-Punkte-Verfahrens SELBST dann, wenn er bei der realistischen
    Portfolio-Groesse (K<=3) noch nicht sichtbar wird: wo liegt der Kreuzungspunkt?"""
    rows = []
    for K in K_values:
        ineqs = _synthetic_inequalities(K, h_max)
        cube = V0 ** (1 / 3)
        x0 = np.array([cube, cube])
        lam0 = -2 / cube

        total_iter = 0
        n_cases = 0
        import itertools
        for size in range(K + 1):
            for A in itertools.combinations(range(K), size):
                n_cases += 1
                res = active_set.solve_case(list(A), V0, ineqs, x0, lam0)
                total_iter += res["n_iter"]

        ip_solution = solver.interior_point_solve(V0, K, h_max, r_min=0.01, ar_max=100.0,
                                                   ineqs=ineqs)
        rows.append({"K": K, "n_cases": n_cases, "expected_n_cases": 2 ** K,
                    "enum_total_iter": total_iter,
                    "ip_total_iter": ip_solution.total_inner_iter})
    return rows


def complementarity_check(V0: float = 10.0, h_max: float = 5.0, r_min: float = 1.3,
                          ar_max: float = 1.8, K: int = 3) -> dict:
    """Die gestoerte Komplementaritaet s_j*mu_j soll mit mu_bar_k gegen 0 gehen - gemessen, nicht
    nur behauptet."""
    solution = solver.interior_point_solve(V0, K, h_max, r_min, ar_max)
    gaps = [row["comp_gap"] for row in solution.history]
    mu_bars = [row["mu_bar"] for row in solution.history]
    monotone = all(b <= a + 1e-9 for a, b in zip(gaps, gaps[1:]))
    return {"history": solution.history, "final_gap": gaps[-1], "final_mu_bar": mu_bars[-1],
            "monotone_decreasing": monotone}


def gradient_check(eps: float = 1e-6) -> dict:
    x = np.array([1.3, 0.7])
    V0 = 10.0
    results = {}
    for name, func, gradf in (
        ("f", fn.surface_area, fn.grad_surface_area),
        ("g1", lambda xx: fn.volume_constraint(xx, V0), fn.grad_volume_constraint),
    ):
        analytic = gradf(x)
        numeric = np.zeros(2)
        for i in range(2):
            xp, xm = x.copy(), x.copy()
            xp[i] += eps
            xm[i] -= eps
            numeric[i] = (func(xp) - func(xm)) / (2 * eps)
        results[f"{name}_max_rel_err"] = float(
            np.max(np.abs(analytic - numeric) / np.maximum(np.abs(analytic), 1e-8)))
    return results
