"""Kennzahlen, Vergleiche über Instanzen und Verdikte. Jedes Verdikt ist eine reine Funktion (Art, Text); die App ruft st.<Art>(Text).

Alle Aussagen gelten für die gerechneten Instanzen, die sechs erfundenen Familien (siehe README) und die **einfachen** Fassungen dieser Demo (Spektralverschiebung, FAQ-Art), nicht für das Programm der Literatur; Zeiten stehen nie in Verdikten.
Lücke bei aufgezähltem Optimum: (Optimum − Schranke) / Optimum; bei nicht aufgezählten Instanzen ersetzt die beste gefundene Lösung das Optimum und die Lücke ist eine **obere** Grenze der wahren Lücke."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import permutations

import numpy as np

import bkr_birkhoff as Bk
import bkr_bound as B
import bkr_constants as C
import bkr_cost as K
import bkr_landscape as L
import bkr_spectral as S
from bkr_instance import hand_instance, make_instance

TOL = 1e-7


@dataclass(frozen=True)
class Settings:
    family: str = C.DEFAULT_FAMILY
    n: int = C.DEFAULT_N
    seed: int = C.DEFAULT_SEED


# ----------------------------------------------------------------------------- Formatierung

def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def pct(x: float, digits: int = 0) -> str:
    return de(100.0 * x, digits) + " %"


def fam(name: str) -> str:
    return C.FAMILY_NAMES.get(name, name)


def num(x: float) -> str:
    """Ganze Zahl ohne Nachkommastelle, sonst eine Stelle: 27 / 27,5 / −12,1."""
    return str(int(round(x))) if abs(x - round(x)) < 1e-9 else de(x, 1)


def rel(a: float, b: float) -> float:
    """(a − b) / a; NaN, wenn a = 0; Rundungsreste unter 1e-12 werden zu 0."""
    if not a:
        return float("nan")
    r = (a - b) / a
    return 0.0 if abs(r) < 1e-12 else r


def median_of(rows: list, key: str) -> float:
    return float(np.median([r[key] for r in rows]))


def starts_for(n: int) -> int:
    """Zahl der Zufallsstarts der gewählten Instanz: bis n = 9 fünfzig, darüber zwanzig."""
    return C.STARTS_LIVE if n <= 9 else C.STARTS_LARGE


# ----------------------------------------------------------------------------- Vergleichswert

def landscape_of(s: Settings) -> L.Landscape:
    return L.landscape(make_instance(s.family, s.n, s.seed))


def reference(inst, seed: int) -> tuple:
    """(Vergleichswert, Anordnung, aufgezählt?): das aufgezählte Optimum bis n = 9, sonst die beste von 200 steilsten Abstiegen."""
    if inst.n in C.EXACT_SIZES:
        ls = L.landscape(inst)
        return ls.opt, [int(x) for x in ls.optima[0]], True
    ub, p = B.best_found(inst.F, inst.D, seed)
    return ub, [int(x) for x in p], False


# ----------------------------------------------------------------------------- Zeilen für die Vergleiche

def row_of(family: str, n: int, seed: int, starts: int = C.STARTS_SWEEP) -> dict:
    """Eine Instanz mit aufgezähltem Optimum: Schranken, konvexe Relaxation samt Rundung, Läufe der FAQ-Art und der exakte Erfolg eines Abstiegs."""
    inst = make_instance(family, n, seed)
    ls = L.landscape(inst)
    opt = ls.opt
    cv = Bk.convex_bound(inst.F, inst.D)
    glb = B.gilmore_lawler(inst.F, inst.D).glb
    pb = S.spectral_bounds(inst.F, inst.D).pb
    bary = Bk.faq_run(inst.F, inst.D, Bk.barycenter(n))
    runs = Bk.random_runs(inst.F, inst.D, seed, starts)
    costs = np.array([r.cost for r in runs])
    return {"family": family, "n": n, "seed": seed, "ref": opt, "exact": True, "glb": glb, "pb": pb, "cq": cv.cq, "gap_glb": rel(opt, glb), "gap_pb": rel(opt, pb), "gap_cq": rel(opt, cv.cq),
            "cq_negative": cv.cq < 0, "cq_valid": cv.cq <= opt + TOL, "round_cost": cv.rounded_cost, "round_over": (cv.rounded_cost - opt) / opt if opt else float("nan"), "round_hit": cv.rounded_cost == opt,
            "interior": cv.interior, "constant": cv.constant, "bary_cost": bary.cost, "bary_hit": bary.cost == opt, "bary_stationary": bary.gap0 < 1e-9, "start_success": float((costs == opt).mean()),
            "best_cost": int(costs.min()), "best_hit": int(costs.min()) == opt, "swap_success": ls.success, "steps": float(np.mean([r.steps for r in runs])), "starts": starts}


def large_row(family: str, n: int, seed: int, starts: int = C.STARTS_LARGE) -> dict:
    """Eine große Instanz: beste von `starts` Läufen der FAQ-Art gegen die beste von 200 steilsten Abstiegen (Vergleichswert)."""
    inst = make_instance(family, n, seed)
    ub, _ = B.best_found(inst.F, inst.D, seed)
    runs = Bk.random_runs(inst.F, inst.D, seed, starts)
    bary = Bk.faq_run(inst.F, inst.D, Bk.barycenter(n))
    costs = np.array([r.cost for r in runs])
    return {"family": family, "n": n, "seed": seed, "ref": ub, "exact": False, "best_cost": int(costs.min()), "median_cost": float(np.median(costs)), "bary_cost": bary.cost, "gap_best": (int(costs.min()) - ub) / ub if ub else float("nan"),
            "gap_median": (float(np.median(costs)) - ub) / ub if ub else float("nan"), "gap_bary": (bary.cost - ub) / ub if ub else float("nan"), "faq_better": int(costs.min()) < ub, "faq_equal": int(costs.min()) == ub,
            "faq_worse": int(costs.min()) > ub, "starts": starts}


def sweep(family: str, n: int, seeds=None) -> list:
    """Die festen Instanzen einer Familie: n = 6, 8 mit Aufzählung und 50 Starts, n ≥ 12 mit der besten gefundenen Lösung und 20 Starts."""
    if n in C.SWEEP_SIZES:
        return [row_of(family, n, s) for s in (seeds if seeds is not None else C.SWEEP_SEEDS_N[n])]
    return [large_row(family, n, s) for s in (seeds if seeds is not None else C.LARGE_SEEDS)]


def pooled(by_family: dict) -> list:
    return [r for f in C.FAMILIES if f in by_family for r in by_family[f]]


def quartiles(x) -> tuple:
    x = np.asarray(x, dtype=float)
    return float(np.percentile(x, 25)), float(np.median(x)), float(np.percentile(x, 75))


def family_summary(rows: list) -> dict:
    """Zusammenfassung der Zeilen einer Familie mit aufgezähltem Optimum."""
    ss = [r["start_success"] for r in rows]
    return {"count": len(rows), "gap_glb": median_of(rows, "gap_glb"), "gap_pb": median_of(rows, "gap_pb"), "gap_cq": median_of(rows, "gap_cq"), "cq_negative": sum(1 for r in rows if r["cq_negative"]),
            "cq_valid": all(r["cq_valid"] for r in rows), "round_over": median_of(rows, "round_over"), "round_hit": sum(1 for r in rows if r["round_hit"]), "interior": median_of(rows, "interior"),
            "constant": median_of(rows, "constant"), "ref": median_of(rows, "ref"), "start_success": float(np.median(ss)), "start_min": min(ss), "start_max": max(ss), "swap_success": median_of(rows, "swap_success"),
            "best_hit": sum(1 for r in rows if r["best_hit"]), "bary_hit": sum(1 for r in rows if r["bary_hit"]), "bary_stationary": sum(1 for r in rows if r["bary_stationary"]), "steps": median_of(rows, "steps")}


def large_summary(rows: list) -> dict:
    return {"count": len(rows), "gap_best": median_of(rows, "gap_best"), "gap_median": median_of(rows, "gap_median"), "gap_bary": median_of(rows, "gap_bary"), "better": sum(1 for r in rows if r["faq_better"]),
            "equal": sum(1 for r in rows if r["faq_equal"]), "worse": sum(1 for r in rows if r["faq_worse"])}


# ----------------------------------------------------------------------------- Experiment 1 bis 4: eine Instanz

def best_so_far(costs) -> list:
    """Laufendes Minimum der Kosten über die Starts: bestes von 1, 2, 3, … Starts."""
    out, cur = [], None
    for c in costs:
        cur = c if cur is None else min(cur, c)
        out.append(cur)
    return out


def instance_data(s: Settings, ls: L.Landscape | None = None) -> dict:
    """Vergleichswert, Schranken, konvexe Relaxation und Läufe der FAQ-Art einer Instanz. `ls`: aufgezählte Landschaft (n ≤ 9)."""
    inst = make_instance(s.family, s.n, s.seed)
    if s.n in C.EXACT_SIZES:
        ls = ls if ls is not None else L.landscape(inst)
        ref, best, exact, perms, swap = ls.opt, [int(x) for x in ls.optima[0]], True, ls.perms, ls.success
    else:
        ref, best, exact = reference(inst, s.seed)
        perms, swap = math.factorial(s.n), None
    glb = B.gilmore_lawler(inst.F, inst.D).glb
    pb = S.spectral_bounds(inst.F, inst.D).pb
    cv = Bk.convex_bound(inst.F, inst.D)
    bary = Bk.faq_run(inst.F, inst.D, Bk.barycenter(inst.n))
    runs = Bk.random_runs(inst.F, inst.D, s.seed, starts_for(s.n))
    costs = [r.cost for r in runs]
    return {"inst": inst, "ref": ref, "best": best, "exact": exact, "perms": perms, "glb": glb, "pb": pb, "convex": cv, "bary": bary, "runs": runs, "costs": costs, "curve": best_so_far(costs), "swap_success": swap,
            "start_success": float(np.mean([c == ref for c in costs])), "best_cost": min(costs), "gap_glb": rel(ref, glb), "gap_pb": rel(ref, pb), "gap_cq": rel(ref, cv.cq), "starts": len(costs)}


def hand_data() -> list:
    """Die drei Handbeispiele der README: Schranken, konvexe Relaxation, Lauf vom Schwerpunkt und Optimum."""
    out = []
    for name, flows, (rows, cols) in (("Drei Bauteile auf einer Zeile", C.HAND_FLOWS, C.HAND_GRID), ("Vier Bauteile auf dem 2×2-Raster", C.HAND_GRID_FLOWS, C.HAND_GRID_GRID),
                                      ("Vier Bauteile auf einer Zeile (mit Lücke)", C.HAND_GAP_FLOWS, C.HAND_GAP_GRID)):
        inst = hand_instance(flows, rows, cols)
        out.append({"name": name, "inst": inst, "glb": B.gilmore_lawler(inst.F, inst.D).glb, "pb": S.spectral_bounds(inst.F, inst.D).pb, "convex": Bk.convex_bound(inst.F, inst.D), "bary": Bk.faq_run(inst.F, inst.D, Bk.barycenter(inst.n)),
                    "opt": min(K.cost(p, inst.F, inst.D) for p in permutations(range(inst.n)))})
    return out


def first_hit(curve, ref) -> int | None:
    """Die kleinste Zahl K von Starts, bei der das beste von K Starts den Vergleichswert erreicht (None: nie)."""
    for k, c in enumerate(curve, start=1):
        if c <= ref:
            return k
    return None


# ----------------------------------------------------------------------------- Verdikte

def _head(s: Settings, v: dict) -> str:
    inst = v["inst"]
    return f"Familie **{fam(s.family)}**, {s.n} Bauteile, Raster {inst.rows}×{inst.cols}, Seed {s.seed}: "


def _refname(v: dict) -> str:
    return "das aufgezählte Optimum" if v["exact"] else f"die beste gefundene Lösung (beste von {C.STARTS} Abstiegen)"


def verdict_convex(s: Settings, v: dict):
    cv = v["convex"]
    neg = cv.cq < 0
    tail = (" Sie liegt unter 0 und ist damit schwächer als die triviale Schranke 0: die Konstante ½ s_F s_D n (" + num(cv.constant) + ") ist ein Vielfaches des Optimums, das die Verschiebung der Spektren wieder abziehen muss." if neg else "")
    return ("warning" if neg else "info"), (_head(s, v) + f"Die konvexe Relaxation (Verschiebung s_F = {de(cv.s_f, 2)}, s_D = {de(cv.s_d, 2)}) liefert nach {len(cv.lowers)} Frank–Wolfe-Schritten die gültige untere Schranke **{num(cv.cq)}** (Lücke {'' if v['exact'] else 'höchstens '}{pct(v['gap_cq'], 1)}), "
                                              f"gegen Gilmore–Lawler {num(v['glb'])} (Lücke {pct(v['gap_glb'], 1)}) und die projizierte Schranke {num(v['pb'])}; {_refname(v)} ist {v['ref']}.{tail}")


def verdict_solution(s: Settings, v: dict):
    cv = v["convex"]
    hit = cv.rounded_cost == v["ref"]
    over = "das ist der Vergleichswert" if hit else "das sind " + pct((cv.rounded_cost - v["ref"]) / v["ref"], 1) + " über dem Vergleichswert " + str(v["ref"])
    return ("success" if hit else "warning"), (_head(s, v) + f"Die konvexe Lösung liegt im Inneren des Polytops (größter Abstand eines Eintrags von 0 oder 1: {de(cv.interior, 2)}). Gerundet per Zuordnung ergibt sie eine Anordnung mit Kosten **{cv.rounded_cost}**, {over}.")


def verdict_faq(s: Settings, v: dict):
    bary, runs = v["bary"], v["runs"]
    hits = sum(1 for c in v["costs"] if c == v["ref"])
    stationary = bary.gap0 < 1e-9
    hit_b = "; das ist der Vergleichswert" if bary.cost == v["ref"] else ""
    if stationary:
        start = "Der Schwerpunkt ist hier ein **stationärer Punkt** (Frank–Wolfe-Lücke 0): der Lauf bewegt sich nicht, die Rundung ist eine beliebige Zuordnung mit Kosten " + str(bary.cost) + hit_b
    else:
        start = f"Vom Schwerpunkt aus läuft Frank–Wolfe {bary.steps} Schritte und rundet zu Kosten {bary.cost}" + hit_b
    return ("success" if hits else "warning"), (_head(s, v) + f"{start}. Von {len(runs)} Zufallsstarts enden **{hits}** bei einer Anordnung mit dem Vergleichswert {v['ref']} "
                                                f"({pct(hits / len(runs))}); die Kosten der Ergebnisse reichen von {min(v['costs'])} bis {max(v['costs'])}.")


def verdict_starts(s: Settings, v: dict):
    k = first_hit(v["curve"], v["ref"])
    swap = v["swap_success"]
    swaptxt = f" Zum Vergleich erreicht ein steilster Abstieg von einer Zufallsanordnung in {pct(swap, 1)} der Fälle ein Optimum (exakt gezählt)." if swap is not None else ""
    if k is None:
        return "warning", _head(s, v) + f"Das beste von {v['starts']} Starts kostet {v['best_cost']} und erreicht den Vergleichswert {v['ref']} nicht; {pct(v['start_success'], 1)} der Starts waren erfolgreich.{swaptxt}"
    return "success", _head(s, v) + f"Der Vergleichswert {v['ref']} wird erstmals beim **{k}. Start** erreicht; {pct(v['start_success'], 1)} der {v['starts']} Starts sind erfolgreich.{swaptxt}"


def verdict_sweep(n: int, by_family: dict):
    summ = {f: family_summary(r) for f, r in by_family.items() if r}
    rows = pooled(by_family)
    total = len(rows)
    valid = all(x["cq_valid"] for x in summ.values())
    neg = sum(x["cq_negative"] for x in summ.values())
    hit = sum(x["round_hit"] for x in summ.values())
    parts = "; ".join(f"{fam(f)} {pct(x['gap_cq'])}" for f, x in summ.items())
    return ("success" if valid else "error"), (f"Konvexe Relaxation bei n = {n}, Median über {total} Instanzen: Lücke der Schranke {pct(median_of(rows, 'gap_cq'))} (Gilmore–Lawler {pct(median_of(rows, 'gap_glb'), 1)}); je Familie: {parts}. Die Schranke liegt in {neg} von {total} Instanzen unter 0. "
                                               f"Die Rundung der konvexen Lösung kostet im Median {pct(median_of(rows, 'round_over'))} mehr als das Optimum und ist in {hit} von {total} Instanzen optimal. "
                                               f"{'Die Schranke lag in jeder Instanz höchstens beim Optimum (gültig).' if valid else 'ACHTUNG: in mindestens einer Instanz lag die Schranke über dem Optimum.'}")


def verdict_starts_sweep(n: int, by_family: dict):
    summ = {f: family_summary(r) for f, r in by_family.items() if r}
    rows = pooled(by_family)
    total = len(rows)
    parts = "; ".join(f"{fam(f)} {de(x['start_success'], 2)} (Tauschsuche {de(x['swap_success'], 2)})" for f, x in summ.items())
    best = sum(x["best_hit"] for x in summ.values())
    bary = sum(x["bary_hit"] for x in summ.values())
    worse = [f for f, x in summ.items() if x["start_success"] < x["swap_success"]]
    return "info", (f"Anteil der Starts mit Optimum (Median über die Instanzen, in Klammern der exakte Erfolg eines Abstiegs aus Stück 1) bei n = {n}: {parts}. Ein einzelner Start der FAQ-Art ist in {len(worse)} von {len(summ)} Familien seltener erfolgreich als ein Abstieg. "
                    f"Das beste von {rows[0]['starts']} Starts findet das Optimum in {best} von {total} Instanzen, der Schwerpunkt als einziger Start in {bary}. Verglichen werden ein Start und ein Abstieg, nicht gleiche Rechenzeit.")


def verdict_large(by_n: dict):
    """by_n: {n: {Familie: Zeilen}} der großen Instanzen."""
    sizes = sorted(by_n)
    parts = []
    for n in sizes:
        rows = pooled(by_n[n])
        b, e, w = (sum(1 for r in rows if r[k]) for k in ("faq_better", "faq_equal", "faq_worse"))
        parts.append(f"n = {n}: FAQ-Art (bestes von {rows[0]['starts']} Starts) besser in {b}, gleich in {e}, schlechter in {w} von {len(rows)} Instanzen; Median über alle Instanzen {pct(median_of(rows, 'gap_best'), 1)} gegen die beste von {C.STARTS} Abstiegen")
    return "info", ("Läufe der FAQ-Art gegen die beste von 200 steilsten Abstiegen (Vergleichswert; positive Abweichung = FAQ-Art schlechter): " + "; ".join(parts) + ". Ein Lauf der FAQ-Art kostet viele Zuordnungen, ein Abstieg Millisekunden.")


def verdict_core(s: Settings, v: dict):
    cv = v["convex"]
    return "info", (f"An einer Instanz sieht man: konvexe Schranke {num(cv.cq)}, Rundung der konvexen Lösung {cv.rounded_cost}, bestes von {v['starts']} Starts der FAQ-Art {v['best_cost']}, bei {'Optimum' if v['exact'] else 'bester gefundener Lösung'} {v['ref']}. "
                    f"Was die Relaxation der Menge bringt und wie viel die Zahl der Starts entscheidet, zeigen Experiment 2 bis 5.")
