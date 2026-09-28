"""Generalisierte aktive-Menge-Enumeration fuer K Ungleichungen (Verallgemeinerung der
Stueck-4/6-Idee auf beliebig viele Ungleichungen statt nur einer): ALLE 2^K Teilmengen "aktiv"
werden durchprobiert, je ein reduziertes KKT-System per gedaempftem Newton geloest (numerische
Jacobi-Matrix wie in Stueck 4/6), dann geprueft, welche Teilmenge primal UND dual zulaessig ist.
Dient als Vergleichsverfahren fuer das primal-duale Innere-Punkte-Verfahren in ip_solver.py."""
import itertools

import numpy as np

import ip_functions as fn


def _numeric_jacobian(F, x, eps=1e-6):
    n = len(x)
    J = np.zeros((n, n))
    base = F(x)
    for j in range(n):
        xp = x.copy()
        xp[j] += eps
        J[:, j] = (F(xp) - base) / eps
    return J


def _newton_damped(F, x0, max_iter=100, tol=1e-9):
    x = np.asarray(x0, dtype=float)
    for k in range(1, max_iter + 1):
        Fx = F(x)
        norm = float(np.linalg.norm(Fx))
        if norm < tol:
            return x, k - 1, True
        J = _numeric_jacobian(F, x)
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            return x, k - 1, False
        step = 1.0
        for _ in range(30):
            if float(np.linalg.norm(F(x + step * dx))) < norm:
                break
            step *= 0.5
        x = x + step * dx
    return x, max_iter, False


def solve_case(A: list, V0: float, ineqs: list, x0, lam0: float, max_iter: int = 100,
              tol: float = 1e-9) -> dict:
    """Loest das reduzierte KKT-System fuer eine feste aktive Teilmenge A (Indizes in ineqs)."""
    nA = len(A)

    def F(v):
        r, h, lam = v[0], v[1], v[2]
        mus = v[3:3 + nA]
        x = np.array([r, h])
        stat = fn.grad_surface_area(x) + lam * fn.grad_volume_constraint(x)
        for idx, j in enumerate(A):
            stat = stat + mus[idx] * ineqs[j][1](x)
        eqs = [fn.volume_constraint(x, V0)]
        for j in A:
            eqs.append(ineqs[j][0](x))
        return np.concatenate([stat, eqs])

    v0 = np.concatenate([x0, [lam0], np.zeros(nA)])
    sol, n_iter, converged = _newton_damped(F, v0, max_iter=max_iter, tol=tol)
    r, h, lam = float(sol[0]), float(sol[1]), float(sol[2])
    mus = {j: float(sol[3 + idx]) for idx, j in enumerate(A)}
    return {"r": r, "h": h, "lam": lam, "mus": mus, "n_iter": n_iter, "converged": converged}


def enumerate_active_sets(V0: float, K: int, h_max: float, r_min: float, ar_max: float,
                          x0=None) -> dict:
    """Probiert alle 2^K Teilmengen durch, gibt die gefundene KKT-Loesung + Kombinatorik-
    Kennzahlen (Fallzahl, Gesamt-Newton-Iterationen ueber alle Faelle) zurueck."""
    ineqs = fn.make_inequalities(K, h_max, r_min, ar_max)
    cube = V0 ** (1 / 3)
    if x0 is None:
        x0 = np.array([cube, cube])
    lam0 = -2 / cube
    total_iter = 0
    n_cases = 0
    found = None
    for size in range(K + 1):
        for A in itertools.combinations(range(K), size):
            n_cases += 1
            res = solve_case(list(A), V0, ineqs, x0, lam0)
            total_iter += res["n_iter"]
            x = np.array([res["r"], res["h"]])
            primal_ok = all(ineqs[j][0](x) <= 1e-6 for j in range(K) if j not in A)
            dual_ok = all(res["mus"][j] >= -1e-6 for j in A)
            feasible = res["converged"] and primal_ok and dual_ok
            if feasible and found is None:
                found = {"A": A, "r": res["r"], "h": res["h"], "n_iter": res["n_iter"]}
    return {"n_cases": n_cases, "total_iter": total_iter, "found": found,
            "expected_n_cases": 2 ** K}
