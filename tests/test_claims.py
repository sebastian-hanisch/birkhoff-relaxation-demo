"""Die Zahlen der README (Abschnitte „Was die Demo zeigt“ und „Grenzen“) als Tests: Standardfall, Handbeispiele und die Vergleiche über feste Instanzen.
Prozentwerte werden auf ganze Prozent geprüft, keine Zahl liegt an einer Rundungsgrenze; Zeiten stehen nie in Tests (DEMO-PLAYBOOK §1b, Wiederholbarkeit)."""

import pytest

import bkr_constants as C
import bkr_evaluation as ev


@pytest.fixture(scope="module")
def standard():
    s = ev.Settings()
    return s, ev.instance_data(s, ev.landscape_of(s))


@pytest.fixture(scope="module")
def sweeps():
    return {n: {f: ev.sweep(f, n) for f in C.FAMILIES} for n in (6, 8, 12, 16)}


def _pct(rows, key):
    return round(ev.median_of(rows, key) * 100)


def test_standard_instance_numbers(standard):
    s, v = standard
    cv = v["convex"]
    assert (s.family, s.n, s.seed) == ("netlist", 8, 35)
    assert v["ref"] == 31 and v["glb"] == 27.0 and v["best_cost"] == 31 and cv.rounded_cost == 54
    assert cv.cq == pytest.approx(-22.069, abs=0.001) and cv.s_f == pytest.approx(5.105, abs=0.001) and cv.s_d == pytest.approx(6.828, abs=0.001) and cv.constant == pytest.approx(139.45, abs=0.01)
    assert v["pb"] == pytest.approx(13.68, abs=0.01) and cv.interior == pytest.approx(0.216, abs=0.001)
    assert v["start_success"] == pytest.approx(0.08) and v["swap_success"] == pytest.approx(0.266, abs=0.001)
    assert v["bary"].cost == 36 and v["bary"].gap0 == pytest.approx(5.0, abs=0.01) and ev.first_hit(v["curve"], v["ref"]) == 3


def test_the_three_hand_examples():
    h = ev.hand_data()
    assert [x["opt"] for x in h] == [6, 10, 6] and [x["convex"].rounded_cost for x in h] == [6, 10, 8] and [x["bary"].cost for x in h] == [6, 10, 8]
    assert [round(x["convex"].cq, 3) for x in h][1:] == [9.883, 3.172] and 5.68 < h[0]["convex"].cq < 5.69
    assert h[0]["bary"].gap0 == pytest.approx(2 / 3, abs=1e-6) and h[1]["bary"].gap0 == 0 and h[2]["bary"].gap0 == 0 and h[2]["bary"].steps == 0


def test_sweep_sizes_and_instance_counts(sweeps):
    assert {n: sum(len(r) for r in by.values()) for n, by in sweeps.items()} == {6: 120, 8: 60, 12: 60, 16: 60}
    assert all(r["cq_valid"] for n in (6, 8) for rows in sweeps[n].values() for r in rows)


@pytest.mark.parametrize("n,gap_cq,gap_glb,over,neg,hit,best,bary,stat", [(6, 74, 7, 29, 52, 3, 97, 26, 0), (8, 226, 9, 33, 35, 1, 41, 14, 0)])
def test_pooled_medians_and_counts(sweeps, n, gap_cq, gap_glb, over, neg, hit, best, bary, stat):
    rows = ev.pooled(sweeps[n])
    assert (_pct(rows, "gap_cq"), _pct(rows, "gap_glb"), _pct(rows, "round_over")) == (gap_cq, gap_glb, over)
    assert (sum(r["cq_negative"] for r in rows), sum(r["round_hit"] for r in rows), sum(r["best_hit"] for r in rows), sum(r["bary_hit"] for r in rows), sum(r["bary_stationary"] for r in rows)) == (neg, hit, best, bary, stat)


def test_the_bound_is_negative_only_for_the_unbalanced_families(sweeps):
    for n, expected in ((6, {"netlist": 0, "uniform": 0, "sparse": 17, "cluster": 0, "star": 19, "skewed": 16}), (8, {"netlist": 5, "uniform": 0, "sparse": 10, "cluster": 0, "star": 10, "skewed": 10})):
        assert {f: ev.family_summary(sweeps[n][f])["cq_negative"] for f in C.FAMILIES} == expected


def test_gaps_rounding_and_interior_per_family(sweeps):
    gap_cq = {6: (38.0, 33.4, 128.2, 22.2, 126.0, 140.0), 8: (110.0, 61.1, 372.2, 33.0, 298.3, 460.3)}
    gap_glb = {6: (7.3, 5.2, 10.3, 17.0, 7.7, 3.4), 8: (10.5, 8.0, 6.1, 9.8, 14.1, 6.4)}
    over = {6: (16, 24, 40, 40, 22, 42), 8: (22, 18, 53, 46, 27, 44)}
    inner = {6: (0.45, 0.36, 0.43, 0.36, 0.47, 0.36), 8: (0.23, 0.23, 0.21, 0.21, 0.24, 0.21)}
    for n in (6, 8):
        for k, f in enumerate(C.FAMILIES):
            sm = ev.family_summary(sweeps[n][f])
            assert sm["gap_cq"] * 100 == pytest.approx(gap_cq[n][k], abs=0.051), (n, f, "cq")
            assert sm["gap_glb"] * 100 == pytest.approx(gap_glb[n][k], abs=0.051), (n, f, "glb")
            assert round(sm["round_over"] * 100) == over[n][k], (n, f, "rundung")
            assert round(sm["interior"], 2) == inner[n][k], (n, f, "innen")


def test_the_constant_dwarfs_the_optimum_at_eight(sweeps):
    const = (134.9, 415.2, 355.6, 327.3, 440.3, 1135.7)
    ref = (41.0, 218.5, 44.0, 176.5, 68.0, 116.0)
    for k, f in enumerate(C.FAMILIES):
        sm = ev.family_summary(sweeps[8][f])
        assert sm["constant"] == pytest.approx(const[k], abs=0.051) and sm["ref"] == pytest.approx(ref[k], abs=0.051) and sm["constant"] > sm["ref"]


def test_start_spans_at_six_components_are_wide(sweeps):
    for f, top in zip(C.FAMILIES, (1.0, 0.86, 1.0, 0.86, 0.92, 1.0)):
        sm = ev.family_summary(sweeps[6][f])
        assert sm["start_min"] == 0.0 and sm["start_max"] == pytest.approx(top, abs=0.006), f


def test_start_success_per_family(sweeps):
    exp6 = {"netlist": 0.15, "uniform": 0.08, "sparse": 0.32, "cluster": 0.17, "star": 0.16, "skewed": 0.20}
    exp8 = {"netlist": 0.21, "uniform": 0.0, "sparse": 0.23, "cluster": 0.0, "star": 0.18, "skewed": 0.10}
    for n, exp in ((6, exp6), (8, exp8)):
        for f, x in exp.items():
            assert ev.family_summary(sweeps[n][f])["start_success"] == pytest.approx(x, abs=0.006), (n, f)


def test_swap_search_beats_one_start_on_every_family_at_six_and_eight(sweeps):
    exp6 = {"netlist": 0.81, "uniform": 0.69, "sparse": 0.86, "cluster": 0.58, "star": 0.83, "skewed": 0.68}
    exp8 = {"netlist": 0.32, "uniform": 0.30, "sparse": 0.44, "cluster": 0.52, "star": 0.44, "skewed": 0.31}
    for n, exp in ((6, exp6), (8, exp8)):
        for f, x in exp.items():
            sm = ev.family_summary(sweeps[n][f])
            assert sm["swap_success"] == pytest.approx(x, abs=0.006), (n, f)
            assert (sm["swap_success"] > sm["start_success"]) is True


@pytest.mark.parametrize("n,better,equal,worse,gap_best", [(12, 0, 35, 25, 0.0), (16, 1, 28, 31, 0.08)])
def test_large_instances(sweeps, n, better, equal, worse, gap_best):
    rows = ev.pooled(sweeps[n])
    assert (sum(r["faq_better"] for r in rows), sum(r["faq_equal"] for r in rows), sum(r["faq_worse"] for r in rows)) == (better, equal, worse)
    assert round(ev.median_of(rows, "gap_best") * 100, 1) == pytest.approx(gap_best, abs=0.05)


def test_large_instances_per_family_best_gap(sweeps):
    exp12 = {"netlist": 0.0, "uniform": 0.0, "sparse": 0.0, "cluster": 1.32, "star": 0.0, "skewed": 0.54}
    exp16 = {"netlist": 0.0, "uniform": 0.08, "sparse": 0.0, "cluster": 1.49, "star": 0.0, "skewed": 2.27}
    for n, exp in ((12, exp12), (16, exp16)):
        for f, x in exp.items():
            assert ev.large_summary(sweeps[n][f])["gap_best"] * 100 == pytest.approx(x, abs=0.006), (n, f)
