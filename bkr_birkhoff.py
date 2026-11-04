"""Relaxation der Menge der Anordnungen auf das Birkhoff-Polytop (doppeltstochastische Matrizen): Frank–Wolfe mit linearer Zuordnung, konvexe Schranke und indefinite Näherung (FAQ-Art).

Kosten wie in Stück 1 bis 5: cost(p) = ½ tr(F X D Xᵀ) mit der Permutationsmatrix X (X[i, p_i] = 1; F und D symmetrisch, Diagonale 0). Relaxiert wird X auf die Menge der Matrizen mit nichtnegativen Einträgen und Zeilen- und Spaltensummen 1;
ihre Ecken sind genau die Permutationsmatrizen. Zielfunktion g(X) = ½ tr(F X D Xᵀ), Gradient F X D.

Frank–Wolfe: aus X die Richtung Y = die Permutation, die ⟨F X D, Y⟩ minimiert (lineare Zuordnung); g(X + α(Y − X)) = g(X) + b α + a α² mit b = ⟨F X D, Y − X⟩ ≤ 0 und a = ½ ⟨F (Y − X) D, Y − X⟩;
Schrittweite exakt: α = min(1, −b / 2a) bei a > 0, sonst α = 1. Die Frank–Wolfe-Lücke −b ist null genau an stationären Punkten.

Konvex: Für jede Permutationsmatrix P gilt tr(P D Pᵀ) = tr(D) = 0 und tr(F P Pᵀ) = tr(F) = 0. Mit F' = F + s_F I, D' = D + s_D I, s_F = −λ_min(F) > 0, s_D = −λ_min(D) > 0 (beide positiv semidefinit)
ist g'(X) = ½ tr(F' X D' Xᵀ) konvex, und g'(P) = cost(P) + ½ s_F s_D n. Frank–Wolfe liefert in jedem Schritt die gültige untere Schranke g'(X) + b ≤ min g' und damit CQ = g'(X) + b − ½ s_F s_D n ≤ Optimum.
Indefinit (FAQ-Art): dasselbe Frank–Wolfe auf g selbst; Start Schwerpunkt J/n oder zufällig im Inneren; Rundung: lineare Zuordnung auf −X. Das ist eine eigene einfache Fassung, nicht das Programm der Literatur."""

from dataclasses import dataclass, field

import numpy as np

import bkr_constants as C
import bkr_cost as K
from bkr_assignment import hungarian
from bkr_rng import SplitMix64


def barycenter(n: int) -> np.ndarray:
    return np.full((n, n), 1.0 / n)


def random_start(n: int, seed: int, lam: float = C.LAMBDA) -> np.ndarray:
    """Zufälliger Start im Inneren: λ · (zufällige Permutationsmatrix) + (1 − λ) · J/n (SplitMix64)."""
    rng = SplitMix64(seed)
    p = list(range(n))
    rng.shuffle(p)
    P = np.zeros((n, n))
    P[np.arange(n), p] = 1.0
    return lam * P + (1 - lam) * barycenter(n)


def lap_min(M) -> tuple:
    """(Permutationsmatrix, Spalte je Zeile) der kleinsten linearen Zuordnung über M."""
    M = np.asarray(M, dtype=float)
    cols, _ = hungarian(M)
    Y = np.zeros(M.shape)
    Y[np.arange(len(M)), cols] = 1.0
    return Y, cols


def round_to_perm(X) -> list:
    """Rundung: die Permutation, die ⟨X, P⟩ maximiert (lineare Zuordnung auf −X)."""
    return [int(c) for c in lap_min(-np.asarray(X, dtype=float))[1]]


def is_doubly_stochastic(X, tol: float = 1e-9) -> bool:
    X = np.asarray(X, dtype=float)
    return bool(X.min() >= -tol and np.allclose(X.sum(axis=0), 1.0, atol=1e-8) and np.allclose(X.sum(axis=1), 1.0, atol=1e-8))


def objective(Fm, Dm, X) -> float:
    """g(X) = ½ tr(F X D Xᵀ) = ½ Σ (F X D) ∘ X."""
    return 0.5 * float(((Fm @ X @ Dm) * X).sum())


@dataclass(frozen=True, eq=False)
class FW:
    X: np.ndarray               # Endpunkt
    gap: float                  # Frank–Wolfe-Lücke im letzten Schritt (null: stationär)
    steps: int                  # ausgeführte Schritte mit Bewegung
    lower: float                # beste untere Schranke g(X_k) + b_k über alle Schritte (gültig nur bei konvexem g)
    values: tuple = field(default_factory=tuple)      # g(X_k) vor jedem Schritt (Länge steps + 1)
    lowers: tuple = field(default_factory=tuple)      # laufendes Maximum von g(X_k) + b_k
    gaps: tuple = field(default_factory=tuple)        # Frank–Wolfe-Lücke vor jedem Schritt


def frank_wolfe(Fm, Dm, X0, iters: int = C.FW_ITERS, tol: float = 1e-9) -> FW:
    """Frank–Wolfe auf dem Birkhoff-Polytop für g(X) = ½ tr(Fm X Dm Xᵀ); hält an, wenn die Lücke unter `tol` fällt."""
    Fm, Dm = np.asarray(Fm, dtype=float), np.asarray(Dm, dtype=float)
    X = np.array(X0, dtype=float)
    values, lowers, gaps = [], [], []
    lb, gap, steps = -float("inf"), float("inf"), 0
    for _ in range(iters):
        grad = Fm @ X @ Dm
        Y, _ = lap_min(grad)
        Dl = Y - X
        b = float((grad * Dl).sum())
        gX = objective(Fm, Dm, X)
        lb = max(lb, gX + b)
        gap = max(0.0, -b)
        values.append(gX)
        lowers.append(lb)
        gaps.append(gap)
        if gap < tol:
            break
        a = 0.5 * float(((Fm @ Dl @ Dm) * Dl).sum())
        alpha = 1.0 if a <= 1e-12 else min(1.0, -b / (2 * a))
        X = X + alpha * Dl
        steps += 1
    if steps == len(values):                  # nach dem letzten Schritt den Endwert nachtragen
        values.append(objective(Fm, Dm, X))
    return FW(X, gap, steps, lb, tuple(values), tuple(lowers), tuple(gaps))


# ----------------------------------------------------------------------------- konvexe Relaxation

def shifts(F, D) -> tuple:
    """(s_F, s_D) = (−λ_min(F), −λ_min(D)); beide positiv, weil F und D die Spur 0 haben (nicht beide null)."""
    sf = -float(np.linalg.eigvalsh(np.asarray(F, dtype=float)).min())
    sd = -float(np.linalg.eigvalsh(np.asarray(D, dtype=float)).min())
    return sf, sd


@dataclass(frozen=True, eq=False)
class Convex:
    cq: float                   # gültige untere Schranke des Optimums
    X: np.ndarray               # Frank–Wolfe-Endpunkt der konvexen Relaxation
    perm: list                  # gerundete Anordnung
    rounded_cost: int           # ihre echten Kosten
    s_f: float
    s_d: float
    constant: float             # ½ s_F s_D n
    interior: float             # größter Abstand eines Eintrags von 0 oder 1
    lowers: tuple               # untere Schranke CQ nach jedem Schritt (laufendes Maximum)
    rounded: tuple              # Kosten der gerundeten Anordnung nach jedem Schritt
    steps: int


def convex_bound(F, D, iters: int = C.CONVEX_ITERS) -> Convex:
    """Konvexe Relaxation (Spektralverschiebung): Frank–Wolfe auf g' mit zertifizierter unterer Schranke und gerundeter Anordnung."""
    F, D = np.asarray(F), np.asarray(D)
    n = len(F)
    Fm, Dm = F.astype(float), D.astype(float)
    sf, sd = shifts(Fm, Dm)
    Fp, Dp = Fm + sf * np.eye(n), Dm + sd * np.eye(n)
    const = 0.5 * sf * sd * n
    # laufend verfolgen: untere Schranke und gerundete Kosten je Schritt
    X = barycenter(n)
    lb = -float("inf")
    lowers, rounded = [], []
    gap, steps = float("inf"), 0
    for _ in range(iters):
        grad = Fp @ X @ Dp
        Y, _ = lap_min(grad)
        Dl = Y - X
        b = float((grad * Dl).sum())
        lb = max(lb, objective(Fp, Dp, X) + b)
        lowers.append(lb - const)
        rounded.append(K.cost(round_to_perm(X), F, D))
        gap = max(0.0, -b)
        if gap < 1e-9:
            break
        a = 0.5 * float(((Fp @ Dl @ Dp) * Dl).sum())
        X = X + (1.0 if a <= 1e-12 else min(1.0, -b / (2 * a))) * Dl
        steps += 1
    perm = round_to_perm(X)
    return Convex(lb - const, X, perm, K.cost(perm, F, D), sf, sd, const, float(np.abs(X - np.round(X)).max()), tuple(lowers), tuple(rounded), steps)


def convex_value_at_permutation(F, D, p) -> float:
    """g'(P) für die Permutationsmatrix zu p: gleich cost(p) + ½ s_F s_D n (zur Probe der Verschiebung)."""
    F, D = np.asarray(F, dtype=float), np.asarray(D, dtype=float)
    n = len(F)
    sf, sd = shifts(F, D)
    X = np.zeros((n, n))
    X[np.arange(n), np.asarray(p, dtype=np.intp)] = 1.0
    return objective(F + sf * np.eye(n), D + sd * np.eye(n), X)


# ----------------------------------------------------------------------------- indefinite Relaxation (FAQ-Art)

@dataclass(frozen=True, eq=False)
class Run:
    perm: list                  # gerundete Anordnung
    cost: int                   # ihre Kosten
    steps: int
    gap0: float                 # Frank–Wolfe-Lücke am Start (null: der Start ist stationär)
    X: np.ndarray
    values: tuple               # g(X_k) je Schritt


def faq_run(F, D, X0, iters: int = C.FW_ITERS) -> Run:
    """Ein Lauf der FAQ-Art: Frank–Wolfe auf g, dann Rundung per Zuordnung."""
    res = frank_wolfe(F, D, X0, iters)
    perm = round_to_perm(res.X)
    return Run(perm, K.cost(perm, F, D), res.steps, res.gaps[0] if res.gaps else 0.0, res.X, res.values)


def random_runs(F, D, instance_seed: int, count: int, iters: int = C.FW_ITERS) -> list:
    """`count` Läufe von zufälligen Starts (Start k aus SplitMix64 mit der Saat instance_seed · 1000 + k)."""
    n = len(F)
    return [faq_run(F, D, random_start(n, instance_seed * 1000 + k), iters) for k in range(count)]
