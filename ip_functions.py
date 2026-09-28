"""Vehikel B, ERWEITERT (eigene Kopie aus lagrange-kkt-demo/straf-barriere-demo/sqp-demo, kein
Import): zylindrischer Transportbehaelter. Zielfunktion + Gleichungsnebenbedingung wie bisher,
PLUS bis zu drei LINEARE Ungleichungen (Hoehenlimit, Mindestradius, max. Aspect Ratio), als Liste
ansprechbar fuer variables K. x = (r, h)."""
import numpy as np


def surface_area(x: np.ndarray) -> float:
    r, h = x
    return 2 * np.pi * r ** 2 + 2 * np.pi * r * h


def grad_surface_area(x: np.ndarray) -> np.ndarray:
    r, h = x
    return np.array([4 * np.pi * r + 2 * np.pi * h, 2 * np.pi * r])


def hess_surface_area() -> np.ndarray:
    return np.array([[4 * np.pi, 2 * np.pi], [2 * np.pi, 0.0]])


def volume_constraint(x: np.ndarray, V0: float) -> float:
    """g1(x) = 0 bei festem Volumen V0."""
    r, h = x
    return np.pi * r ** 2 * h - V0


def grad_volume_constraint(x: np.ndarray) -> np.ndarray:
    r, h = x
    return np.array([2 * np.pi * r * h, np.pi * r ** 2])


def hess_volume_constraint(x: np.ndarray) -> np.ndarray:
    r, h = x
    return np.array([[2 * np.pi * h, 2 * np.pi * r], [2 * np.pi * r, 0.0]])


def height_limit(x: np.ndarray, h_max: float) -> float:
    """g2(x) <= 0: Hoehenlimit."""
    return x[1] - h_max


def grad_height_limit() -> np.ndarray:
    return np.array([0.0, 1.0])


def min_radius(x: np.ndarray, r_min: float) -> float:
    """g3(x) <= 0: Mindestradius (Handhabung/Struktur)."""
    return r_min - x[0]


def grad_min_radius() -> np.ndarray:
    return np.array([-1.0, 0.0])


def max_aspect_ratio(x: np.ndarray, ar_max: float) -> float:
    """g4(x) <= 0: maximales Hoehen-zu-Radius-Verhaeltnis (Kippstabilitaet)."""
    r, h = x
    return h - ar_max * r


def grad_max_aspect_ratio(ar_max: float) -> np.ndarray:
    return np.array([-ar_max, 1.0])


def make_inequalities(K: int, h_max: float, r_min: float, ar_max: float) -> list:
    """Liste von (g(x), grad_g(x)) Funktionspaaren, je nach K in {1,2,3}. Alle linear (konstanter
    Gradient, Hesse-Matrix null), damit die Fallzahl-Kombinatorik isoliert von zusaetzlicher
    Nichtlinearitaet bleibt."""
    specs = [
        (lambda x: height_limit(x, h_max), lambda x: grad_height_limit()),
        (lambda x: min_radius(x, r_min), lambda x: grad_min_radius()),
        (lambda x: max_aspect_ratio(x, ar_max), lambda x: grad_max_aspect_ratio(ar_max)),
    ]
    return specs[:K]
