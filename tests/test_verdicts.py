"""Verdikte (reine Funktionen in bkr_evaluation.py) an handgesetzten Werten: Art, Schwellen und jede Zahl im Text (DEMO-PLAYBOOK §1b)."""

import pytest

import bkr_evaluation as ev


def _row(ss, swap, cq=0.3, neg=False, hit=False, best=True, bary=False, stat=False, over=0.2, starts=50, glb=0.1, valid=True):
    return {"gap_glb": glb, "gap_pb": 0.2, "gap_cq": cq, "cq_negative": neg, "cq_valid": valid, "round_over": over, "round_hit": hit, "interior": 0.3, "constant": 100.0, "ref": 50.0, "start_success": ss,
            "swap_success": swap, "best_hit": best, "bary_hit": bary, "bary_stationary": stat, "steps": 100.0, "starts": starts}


def _big(better, equal, worse, gap=0.0, starts=20):
    return {"gap_best": gap, "gap_median": 0.05, "gap_bary": 0.1, "faq_better": better, "faq_equal": equal, "faq_worse": worse, "starts": starts}


def test_convex_and_solution_verdicts_quote_the_computed_values(land6):
    s = ev.Settings("netlist", 6, 35)
    v = ev.instance_data(s, land6[1])
    cv = v["convex"]
    kind, text = ev.verdict_convex(s, v)
    assert "Familie **Netzliste**, 6 Bauteile, Raster 2×3, Seed 35" in text and f"**{ev.num(cv.cq)}**" in text and f"s_F = {ev.de(cv.s_f, 2)}" in text and "nach 300 Frank–Wolfe-Schritten" in text and f"{v['ref']}" in text
    assert kind == ("warning" if cv.cq < 0 else "info")
    kind2, text2 = ev.verdict_solution(s, v)
    assert f"**{cv.rounded_cost}**" in text2 and f"{ev.de(cv.interior, 2)}" in text2 and kind2 == ("success" if cv.rounded_cost == v["ref"] else "warning") and ("das ist der Vergleichswert" in text2) == (cv.rounded_cost == v["ref"])


def test_convex_verdict_for_a_negative_bound_and_solution_verdict_for_a_hit():
    s = ev.Settings("sparse", 6, 100000)
    v = ev.instance_data(s)
    cv = v["convex"]
    if cv.cq < 0:
        assert "Sie liegt unter 0" in ev.verdict_convex(s, v)[1] and ev.verdict_convex(s, v)[0] == "warning"
    h = ev.hand_data()[0]
    vh = {"inst": h["inst"], "exact": True, "ref": h["opt"], "convex": h["convex"], "glb": h["glb"], "pb": h["pb"], "gap_cq": 0.05, "gap_glb": 0.0}
    k, t = ev.verdict_solution(ev.Settings("netlist", 6, 1), vh)
    assert k == "success" and "das ist der Vergleichswert" in t
    neg = dict(vh, convex=type("C", (), {"cq": -3.0, "s_f": 2.0, "s_d": 3.0, "constant": 9.0, "lowers": (1, 2), "rounded_cost": 8, "interior": 0.3})(), gap_cq=1.5)
    kk, tt = ev.verdict_convex(ev.Settings("netlist", 6, 1), neg)
    assert kk == "warning" and "Sie liegt unter 0" in tt and "ein Vielfaches des Optimums" in tt


def test_faq_verdict_with_a_stationary_and_a_moving_barycenter():
    h = ev.hand_data()
    s = ev.Settings("netlist", 6, 1)
    runs = [type("R", (), {"cost": c})() for c in (6, 8, 6, 7)]
    base = {"inst": h[2]["inst"], "exact": True, "ref": 6, "runs": runs, "costs": [r.cost for r in runs]}
    kind, text = ev.verdict_faq(s, dict(base, bary=h[2]["bary"]))
    assert kind == "success" and "**stationärer Punkt**" in text and "Kosten 8" in text and "Von 4 Zufallsstarts enden **2**" in text and "(50 %)" in text and "von 6 bis 8" in text
    moving = type("B", (), {"gap0": 0.7, "steps": 100, "cost": 6})()
    kind2, text2 = ev.verdict_faq(s, dict(base, bary=moving))
    assert "läuft Frank–Wolfe 100 Schritte und rundet zu Kosten 6; das ist der Vergleichswert" in text2 and "stationärer" not in text2
    none = dict(base, runs=runs[:1], costs=[7], bary=moving)
    assert ev.verdict_faq(s, none)[0] == "warning" and "enden **0**" in ev.verdict_faq(s, none)[1]


def test_starts_verdict_with_and_without_a_hit():
    h = ev.hand_data()[2]
    s = ev.Settings("netlist", 6, 1)
    v = {"inst": h["inst"], "exact": True, "ref": 5, "curve": [9, 7, 7, 5, 5], "best_cost": 5, "start_success": 0.2, "swap_success": 0.633, "starts": 5}
    kind, text = ev.verdict_starts(s, v)
    assert kind == "success" and "erstmals beim **4. Start**" in text and "20,0 % der 5 Starts" in text and "63,3 %" in text
    miss = dict(v, curve=[9, 7], best_cost=7, start_success=0.0, starts=2, swap_success=None)
    kind2, text2 = ev.verdict_starts(s, miss)
    assert kind2 == "warning" and "erreicht den Vergleichswert 5 nicht" in text2 and "Zum Vergleich" not in text2


def test_sweep_verdict_counts_and_checks_validity():
    by = {"netlist": [_row(0.1, 0.5, cq=0.4, hit=False, over=0.2), _row(0.2, 0.5, cq=0.6, neg=False, hit=True, over=0.1)], "sparse": [_row(0.0, 0.5, cq=1.2, neg=True, over=0.4), _row(0.1, 0.5, cq=1.4, neg=True, over=0.5)]}
    kind, text = ev.verdict_sweep(8, by)
    assert kind == "success" and "bei n = 8" in text and "über 4 Instanzen" in text and "Lücke der Schranke 90 %" in text and "(Gilmore–Lawler 10,0 %)" in text and "Netzliste 50 %" in text and "Dünn besetzt 130 %" in text
    assert "in 2 von 4 Instanzen unter 0" in text and "im Median 30 % mehr als das Optimum und ist in 1 von 4 Instanzen optimal" in text and "(gültig)" in text
    by["netlist"][0] = _row(0.1, 0.5, valid=False)
    assert ev.verdict_sweep(8, by)[0] == "error" and "ACHTUNG" in ev.verdict_sweep(8, by)[1]


def test_starts_sweep_verdict_compares_with_the_descent():
    by = {"netlist": [_row(0.1, 0.8, best=True, bary=True), _row(0.3, 0.8, best=True)], "uniform": [_row(0.9, 0.3, best=False), _row(0.7, 0.3, best=True, bary=True)]}
    kind, text = ev.verdict_starts_sweep(6, by)
    assert kind == "info" and "bei n = 6: Netzliste 0,20 (Tauschsuche 0,80); Gleichverteilt 0,80 (Tauschsuche 0,30)" in text and "in 1 von 2 Familien seltener erfolgreich als ein Abstieg" in text
    assert "Das beste von 50 Starts findet das Optimum in 3 von 4 Instanzen, der Schwerpunkt als einziger Start in 2" in text and "nicht gleiche Rechenzeit" in text


def test_large_verdict_counts_and_medians():
    by = {12: {"netlist": [_big(False, True, False)] * 2, "cluster": [_big(False, False, True, gap=0.02)] * 2}, 16: {"netlist": [_big(True, False, False, gap=-0.01)]}}
    kind, text = ev.verdict_large(by)
    assert kind == "info" and "n = 12: FAQ-Art (bestes von 20 Starts) besser in 0, gleich in 2, schlechter in 2 von 4 Instanzen; Median über alle Instanzen 1,0 %" in text
    assert "n = 16: FAQ-Art (bestes von 20 Starts) besser in 1, gleich in 0, schlechter in 0 von 1 Instanzen; Median über alle Instanzen −1,0 %" in text and "Millisekunden" in text


def test_core_verdict_quotes_the_numbers(land6):
    s = ev.Settings("netlist", 6, 35)
    v = ev.instance_data(s, land6[1])
    text = ev.verdict_core(s, v)[1]
    assert f"konvexe Schranke {ev.num(v['convex'].cq)}" in text and f"Rundung der konvexen Lösung {v['convex'].rounded_cost}" in text and "bestes von 50 Starts" in text and "Optimum" in text and "Experiment 2 bis 5" in text
    big = ev.instance_data(ev.Settings("uniform", 12, 35))
    assert "bester gefundener Lösung" in ev.verdict_core(ev.Settings("uniform", 12, 35), big)[1] and "bestes von 20 Starts" in ev.verdict_core(ev.Settings("uniform", 12, 35), big)[1]
