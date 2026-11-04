"""Zeilen für die Vergleiche, Zusammenfassungen und Instanzdaten: Handwerte und Eigenschaften."""

import math

import numpy as np
import pytest

import bkr_birkhoff as Bk
import bkr_bound as B
import bkr_constants as C
import bkr_evaluation as ev
import bkr_instance as I
import bkr_landscape as L


def test_helpers():
    assert ev.de(-1.5, 1) == "−1,5" and ev.pct(0.266, 1) == "26,6 %" and ev.fam("star") == "Stern" and ev.num(27.0) == "27" and ev.num(26.5) == "26,5" and ev.num(-77.14) == "−77,1"
    assert ev.rel(10, 8) == pytest.approx(0.2) and math.isnan(ev.rel(0, 1)) and ev.rel(10, 10 + 1e-14) == 0.0 and ev.starts_for(8) == 50 and ev.starts_for(9) == 50 and ev.starts_for(12) == 20
    assert ev.median_of([{"v": 1}, {"v": 5}, {"v": 2}], "v") == 2.0 and ev.quartiles([1, 2, 3, 4, 5]) == (2.0, 3.0, 4.0)


def test_best_so_far_and_first_hit_by_hand():
    assert ev.best_so_far([9, 7, 8, 7, 5, 6]) == [9, 7, 7, 7, 5, 5] and ev.best_so_far([]) == []
    assert ev.first_hit([9, 7, 7, 5, 5], 5) == 4 and ev.first_hit([9, 7], 5) is None and ev.first_hit([5], 5) == 1 and ev.first_hit([4], 5) == 1


def test_the_hand_data_match_the_readme():
    h = ev.hand_data()
    assert [x["opt"] for x in h] == [6, 10, 6] and [x["glb"] for x in h] == [6.0, 10.0, 5.0] and [x["convex"].rounded_cost for x in h] == [6, 10, 8] and [x["bary"].cost for x in h][1:] == [10, 8]
    assert [round(x["convex"].cq, 2) for x in h][1:] == [9.88, 3.17] and 5.68 < h[0]["convex"].cq < 5.69 and [x["bary"].gap0 == 0 for x in h] == [False, True, True]


def test_row_equals_its_parts(land6):
    inst, ls = land6
    r = ev.row_of("netlist", 6, 35, starts=10)
    cv = Bk.convex_bound(inst.F, inst.D)
    runs = Bk.random_runs(inst.F, inst.D, 35, 10)
    assert r["ref"] == ls.opt and r["exact"] and r["glb"] == B.gilmore_lawler(inst.F, inst.D).glb and r["cq"] == pytest.approx(cv.cq) and r["round_cost"] == cv.rounded_cost and r["cq_valid"] and r["starts"] == 10
    assert r["start_success"] == pytest.approx(np.mean([x.cost == ls.opt for x in runs])) and r["best_cost"] == min(x.cost for x in runs) and r["swap_success"] == pytest.approx(ls.success)
    assert r["gap_cq"] == pytest.approx((ls.opt - cv.cq) / ls.opt) and r["round_over"] == pytest.approx((cv.rounded_cost - ls.opt) / ls.opt) and r["cq_negative"] == (cv.cq < 0) and r["best_hit"] == (r["best_cost"] == ls.opt)


def test_large_row_compares_the_best_start_with_the_best_found_solution():
    r = ev.large_row("uniform", 12, 100000, starts=6)
    inst = I.make_instance("uniform", 12, 100000)
    ub, _ = B.best_found(inst.F, inst.D, 100000)
    runs = Bk.random_runs(inst.F, inst.D, 100000, 6)
    best = min(x.cost for x in runs)
    assert r["ref"] == ub and not r["exact"] and r["best_cost"] == best and r["gap_best"] == pytest.approx((best - ub) / ub) and (r["faq_better"], r["faq_equal"], r["faq_worse"]) == (best < ub, best == ub, best > ub)
    assert r["median_cost"] == pytest.approx(float(np.median([x.cost for x in runs]))) and r["starts"] == 6


def test_sweep_dispatches_on_the_size_and_uses_the_fixed_seeds():
    assert [r["seed"] for r in ev.sweep("cluster", 6, seeds=(1, 2))] == [1, 2] and all(r["exact"] for r in ev.sweep("cluster", 6, seeds=(1,)))
    assert all(not r["exact"] for r in ev.sweep("cluster", 12, seeds=(1,)))
    assert len(C.SWEEP_SEEDS_N[6]) == 20 and len(C.SWEEP_SEEDS_N[8]) == 10 and C.SWEEP_SEEDS_N[8] == C.SWEEP_SEEDS[:10] and len(C.LARGE_SEEDS) == 10


def _row(ss, swap, cq=0.3, neg=False, hit=False, best=True, bary=False, stat=False, over=0.2):
    return {"gap_glb": 0.1, "gap_pb": 0.2, "gap_cq": cq, "cq_negative": neg, "cq_valid": True, "round_over": over, "round_hit": hit, "interior": 0.3, "constant": 100.0, "ref": 50.0, "start_success": ss,
            "swap_success": swap, "best_hit": best, "bary_hit": bary, "bary_stationary": stat, "steps": 100.0}


def test_family_summary_by_hand():
    rows = [_row(0.0, 0.5, cq=0.1, neg=True), _row(0.2, 0.7, cq=0.3, hit=True, bary=True), _row(1.0, 0.9, cq=0.5, best=False, stat=True)]
    s = ev.family_summary(rows)
    assert s["count"] == 3 and s["gap_cq"] == pytest.approx(0.3) and s["cq_negative"] == 1 and s["round_hit"] == 1 and s["best_hit"] == 2 and s["bary_hit"] == 1 and s["bary_stationary"] == 1
    assert s["start_success"] == 0.2 and s["start_min"] == 0.0 and s["start_max"] == 1.0 and s["swap_success"] == 0.7 and s["cq_valid"] and s["round_over"] == 0.2 and s["constant"] == 100.0 and s["steps"] == 100.0
    rows[0]["cq_valid"] = False
    assert not ev.family_summary(rows)["cq_valid"]


def test_large_summary_by_hand():
    rows = [{"gap_best": g, "gap_median": 0.05, "gap_bary": 0.1, "faq_better": b, "faq_equal": e, "faq_worse": w} for g, b, e, w in ((-0.01, True, False, False), (0.0, False, True, False), (0.02, False, False, True), (0.0, False, True, False))]
    s = ev.large_summary(rows)
    assert s["count"] == 4 and s["gap_best"] == 0.0 and s["better"] == 1 and s["equal"] == 2 and s["worse"] == 1 and s["gap_median"] == 0.05 and s["gap_bary"] == 0.1


def test_pooled_keeps_the_family_order():
    assert [r["x"] for r in ev.pooled({"star": [{"x": 1}], "netlist": [{"x": 2}, {"x": 3}]})] == [2, 3, 1]


def test_instance_data_exact_and_large(land6):
    s = ev.Settings("netlist", 6, 35)
    v = ev.instance_data(s, land6[1])
    assert v["exact"] and v["ref"] == land6[1].opt and v["perms"] == 720 and v["starts"] == 50 and len(v["costs"]) == 50 and len(v["curve"]) == 50 and v["curve"][-1] == v["best_cost"] == min(v["costs"])
    assert v["swap_success"] == pytest.approx(land6[1].success) and v["start_success"] == pytest.approx(np.mean([c == v["ref"] for c in v["costs"]])) and v["gap_cq"] == pytest.approx((v["ref"] - v["convex"].cq) / v["ref"])
    assert sorted(v["best"]) == list(range(6)) and all(v["curve"][i + 1] <= v["curve"][i] for i in range(49))
    big = ev.instance_data(ev.Settings("uniform", 12, 35))
    assert not big["exact"] and big["perms"] == math.factorial(12) and big["starts"] == 20 and big["swap_success"] is None and sorted(big["best"]) == list(range(12))


def test_landscape_of_uses_the_settings():
    assert ev.landscape_of(ev.Settings("cluster", 6, 3)).opt == L.landscape(I.make_instance("cluster", 6, 3)).opt
