"""Die Gilmore–Lawler-Schranke (GLB) des quadratischen Zuordnungsproblems in der Koopmans–Beckmann-Form und die Gegenstücke.

Kosten wie in Stück 1 und 2: cost(p) = Σ_{i<k} F[i,k] · D[p_i, p_k] (jedes Paar einmal; F und D symmetrisch, Diagonale 0; p[i] = Platz des Bauteils i).

Schranke: Sitzt Bauteil i auf Platz k, kosten die übrigen Bauteile mindestens L[i,k] := das kleinste Skalarprodukt der Flusszeile von i (ohne i) mit der Abstandszeile von k (ohne k), wenn jedes andere
Bauteil auf einen anderen Platz kommt, aber die Forderung weggelassen wird, dass die Plätze der **verschiedenen** Bauteile zusammenpassen: Fluss aufsteigend, Abstand absteigend sortiert (Umordnungsungleichung).
Jede Anordnung kostet daher mindestens ½ · Σ_i L[i, p_i] (½, weil jedes Paar in der Summe zweimal vorkommt); die kleinste solche Summe über alle Permutationen löst die lineare Zuordnung (Ungarische Methode).
Ihr halber Wert ist die GLB. Die **Zeilenminimum-Schranke** ½ · Σ_i min_k L[i,k] lässt die Zuordnung weg. Die Zuordnung selbst ist eine Anordnung ("Runden"); ihre echten Kosten sind eine obere Schranke."""

from dataclasses import dataclass

import numpy as np

import bkr_cost as K
from bkr_assignment import hungarian
from bkr_rng import SplitMix64


def flow_rows(F) -> np.ndarray:
    """Je Bauteil die Flüsse zu den anderen, aufsteigend sortiert (n × (n−1))."""
    F = np.asarray(F)
    n = len(F)
    return np.array([np.sort(np.delete(F[i], i)) for i in range(n)], dtype=np.int64).reshape(n, max(n - 1, 0))


def distance_rows(D) -> np.ndarray:
    """Je Platz die Abstände zu den anderen, absteigend sortiert (n × (n−1))."""
    D = np.asarray(D)
    n = len(D)
    return np.array([np.sort(np.delete(D[k], k))[::-1] for k in range(n)], dtype=np.int64).reshape(n, max(n - 1, 0))


def lower_matrix(F, D) -> np.ndarray:
    """L[i,k] (ganzzahlig): kleinstes Skalarprodukt der Flusszeile i mit der Abstandszeile k."""
    return flow_rows(F) @ distance_rows(D).T


@dataclass(frozen=True, eq=False)
class Bound:
    L: np.ndarray            # (n, n)
    assign: np.ndarray       # Platz je Bauteil (Zuordnung der linearen Aufgabe, eine kleinste)
    lap: int                 # Wert der linearen Zuordnung über L (ganzzahlig)
    rowmin2: int             # Σ_i min_k L[i,k] (ganzzahlig)
    rounded_cost: int        # echte Kosten der Anordnung `assign`

    @property
    def glb(self) -> float:
        """Die Gilmore–Lawler-Schranke (halbzahlig)."""
        return self.lap / 2

    @property
    def rowmin(self) -> float:
        """Die Schranke ohne Zuordnung."""
        return self.rowmin2 / 2


def gilmore_lawler(F, D) -> Bound:
    F, D = np.asarray(F), np.asarray(D)
    L = lower_matrix(F, D)
    assign, lap = hungarian(L)
    return Bound(L, assign, int(lap), int(L.min(axis=1).sum()), K.cost(assign.tolist(), F, D))


def costs_of_many(Q, F, D) -> np.ndarray:
    """Kosten aller Anordnungen in Q (Zeilen)."""
    F32, D32 = np.asarray(F).astype(np.int32), np.asarray(D).astype(np.int32)
    return ((F32[None] * D32[Q[:, :, None], Q[:, None, :]]).sum(axis=(1, 2)) // 2).astype(np.int64)


def random_starts(n: int, count: int, seed: int) -> np.ndarray:
    rng = SplitMix64(seed * 7919 + n * 13 + 5)
    out = np.empty((count, n), dtype=np.intp)
    for r in range(count):
        p = list(range(n))
        rng.shuffle(p)
        out[r] = p
    return out


def descents(F, D, starts) -> tuple:
    """Steilster Abstieg (kleinstes Δ, bei Gleichstand das erste Paar, solange Δ < 0) von jeder Startanordnung, alle Starts zugleich. Rückgabe: (Endanordnungen, Endkosten)."""
    F, D = np.asarray(F), np.asarray(D)
    cur = np.array(starts, dtype=np.intp).copy()
    n = cur.shape[1]
    cost = costs_of_many(cur, F, D)
    pairs = K.pair_list(n)
    pa = np.array([a for a, _ in pairs], dtype=np.intp)
    pb = np.array([b for _, b in pairs], dtype=np.intp)
    active = np.arange(len(cur))
    while len(active) and len(pairs):
        dl = K.all_deltas(cur[active], F, D)
        j = dl.argmin(axis=1)
        dm = dl[np.arange(len(active)), j]
        go = dm < 0
        idx = active[go]
        if not len(idx):
            break
        r = np.arange(len(idx))
        ta = cur[idx, pa[j[go]]].copy()
        cur[idx, pa[j[go]]] = cur[idx, pb[j[go]]]
        cur[idx, pb[j[go]]] = ta
        cost[idx] += dm[go]
        active = idx
    return cur, cost


def best_found(F, D, seed: int, count: int = 200) -> tuple:
    """(beste Kosten, Anordnung) der besten von `count` steilsten Abstiegen aus Zufallsstarts: eine **obere** Schranke des Optimums."""
    ends, cost = descents(F, D, random_starts(len(F), count, seed))
    i = int(cost.argmin())
    return int(cost[i]), ends[i]
