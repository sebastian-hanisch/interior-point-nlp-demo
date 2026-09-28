"""Primal-duales Innere-Punkte-Verfahren (Waechter & Biegler 2006, IPOPT): Schlupfvariablen s>0
fuer jede Ungleichung, EIN Newton-System fuer Stationaritaet + Gleichung + g_j+s_j=0 + gestoerte
Komplementaritaet s_j*mu_j=mu_bar (ALLE j gemeinsam, keine Fallunterscheidung wie in
ip_active_set.py), mit Fraction-to-Boundary-Regel (s,mu bleiben strikt positiv) und aeusserer
Folge mu_bar_k -> 0 (analog zur Barriere-Folge aus Stueck 5, aber primal-dual: das Newton-System
waechst nur LINEAR mit K in der Groesse, nicht mit 2^K in der Fallzahl wie bei aktiver-Menge-
Enumeration)."""
from dataclasses import dataclass

import numpy as np

import ip_functions as fn


def _numeric_jacobian(F, v, eps=1e-6):
    n = len(v)
    J = np.zeros((n, n))
    base = F(v)
    for j in range(n):
        vp = v.copy()
        vp[j] += eps
        J[:, j] = (F(vp) - base) / eps
    return J


@dataclass
class InteriorPointSolution:
    r: float
    h: float
    lam: float
    s: np.ndarray
    mu: np.ndarray
    total_inner_iter: int
    n_outer: int
    history: list


def interior_point_solve(V0: float, K: int, h_max: float, r_min: float, ar_max: float, x0=None,
                         mu_bar0: float = 1.0, gamma: float = 5.0, n_outer: int = 15,
                         inner_tol: float = 1e-8, inner_max_iter: int = 50,
                         tau: float = 0.995, ineqs: list = None) -> InteriorPointSolution:
    if ineqs is None:
        ineqs = fn.make_inequalities(K, h_max, r_min, ar_max)
    cube = V0 ** (1 / 3)
    x = np.array(x0, dtype=float) if x0 is not None else np.array([cube, cube])
    s = np.ones(K)
    mu = np.ones(K)
    lam = -2 / cube
    mu_bar = mu_bar0
    total_inner_iter = 0
    history = []

    def residual(x, s, lam, mu):
        stat = fn.grad_surface_area(x) + lam * fn.grad_volume_constraint(x)
        for j in range(K):
            stat = stat + mu[j] * ineqs[j][1](x)
        res_eq = np.array([fn.volume_constraint(x, V0)])
        res_ineq = np.array([ineqs[j][0](x) + s[j] for j in range(K)])
        res_comp = np.array([s[j] * mu[j] - mu_bar for j in range(K)])
        return np.concatenate([stat, res_eq, res_ineq, res_comp])

    def full_F(v):
        xx, ss, ll, mm = v[0:2], v[2:2 + K], v[2 + K], v[3 + K:3 + 2 * K]
        return residual(xx, ss, ll, mm)

    for k in range(n_outer):
        for _inner in range(inner_max_iter):
            F = residual(x, s, lam, mu)
            norm = float(np.linalg.norm(F))
            total_inner_iter += 1
            if norm < inner_tol:
                break
            v = np.concatenate([x, s, [lam], mu])
            J = _numeric_jacobian(full_F, v)
            try:
                dv = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                break
            dx, ds, dlam, dmu = dv[0:2], dv[2:2 + K], dv[2 + K], dv[3 + K:3 + 2 * K]

            alpha = 1.0
            for j in range(K):
                if ds[j] < 0:
                    alpha = min(alpha, -tau * s[j] / ds[j])
                if dmu[j] < 0:
                    alpha = min(alpha, -tau * mu[j] / dmu[j])

            x = x + alpha * dx
            s = s + alpha * ds
            lam = lam + alpha * dlam
            mu = mu + alpha * dmu
        history.append({"k": k, "mu_bar": mu_bar, "r": float(x[0]), "h": float(x[1]),
                        "comp_gap": float(np.max(s * mu)) if K > 0 else 0.0})
        mu_bar /= gamma

    return InteriorPointSolution(r=float(x[0]), h=float(x[1]), lam=lam, s=s, mu=mu,
                                 total_inner_iter=total_inner_iter, n_outer=n_outer,
                                 history=history)
