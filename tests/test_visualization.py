"""Abbildungen: sie entstehen für jede Eingabe, Achsen sind gesperrt, Beschriftungen und Daten stimmen mit den Werten überein; PDF-Export."""

import bkr_constants as C
import bkr_evaluation as ev
import bkr_instance as I
import bkr_visualization as V
from bkr_pdf_export import build_pdf


def _locked(fig):
    return fig.layout.xaxis.fixedrange and fig.layout.yaxis.fixedrange


def test_bounds_bar_has_the_six_values_in_order(land6):
    s = ev.Settings("netlist", 6, 35)
    v = ev.instance_data(s, land6[1])
    fig = V.build_bounds_bar(v)
    cv = v["convex"]
    assert list(fig.data[0].x) == [cv.cq, v["pb"], v["glb"], v["ref"], v["best_cost"], cv.rounded_cost] and list(fig.data[0].y)[3] == "Optimum" and _locked(fig)
    big = ev.instance_data(ev.Settings("uniform", 12, 35))
    assert list(V.build_bounds_bar(big).data[0].y)[3] == "beste gefundene Lösung"


def test_convex_trace_shows_the_bound_and_the_rounding_per_step(land6):
    s = ev.Settings("netlist", 6, 35)
    v = ev.instance_data(s, land6[1])
    fig = V.build_convex_trace(v["convex"], v["ref"])
    assert [t.name for t in fig.data] == ["untere Schranke CQ", "Kosten der gerundeten Anordnung", "Vergleichswert"] and list(fig.data[0].y) == list(v["convex"].lowers) and list(fig.data[1].y) == list(v["convex"].rounded) and _locked(fig)


def test_heatmap_shows_the_solution_and_the_board_the_rounding(land6):
    s = ev.Settings("netlist", 6, 35)
    v = ev.instance_data(s, land6[1])
    fig = V.build_x_heatmap(v["inst"], v["convex"].X)
    assert [list(r) for r in fig.data[0].z] == v["convex"].X.tolist() and list(fig.data[0].y) == list(v["inst"].names) and _locked(fig)
    assert len(V.build_board(v["inst"], v["convex"].perm, "Rundung").data[-1].x) == 6


def test_faq_traces_hist_and_curve(land6):
    s = ev.Settings("netlist", 6, 35)
    v = ev.instance_data(s, land6[1])
    fig = V.build_faq_traces(v)
    assert fig.data[0].name == "Schwerpunkt" and len(fig.data) == 1 + 8 + 1 and fig.data[-1].name == "Vergleichswert" and _locked(fig)
    hist = V.build_cost_hist(v)
    assert int(sum(hist.data[0].y)) == 50 and _locked(hist)
    curve = V.build_best_curve(v)
    assert list(curve.data[0].y) == list(v["curve"]) and list(curve.data[0].x) == list(range(1, 51)) and _locked(curve)


def _by(n=6, seeds=(1, 2, 3)):
    return {f: ev.sweep(f, n, seeds) for f in C.FAMILIES}


def test_rounding_starts_hits_and_large_charts():
    by = _by()
    r = V.build_rounding_bars(6, by, ev.median_of)
    assert [t.name for t in r.data] == ["Rundung: Kosten über dem Optimum", "Lücke der konvexen Schranke", "Lücke Gilmore–Lawler"] and len(r.data[0].x) == 6 and r.layout.yaxis.type == "log" and _locked(r)
    st = V.build_starts_bars(6, by, ev.median_of)
    assert [t.name for t in st.data] == ["steilster Abstieg (exakt gezählt)", "ein Start der FAQ-Art"] and all(0 <= y <= 100 for t in st.data for y in t.y) and _locked(st)
    hits = V.build_best_hits({6: by, 8: by})
    assert len(hits.data) == 4 and all(0 <= y <= 100 for t in hits.data for y in t.y) and _locked(hits)
    big = {12: _by(12, (1, 2)), 16: _by(16, (1,))}
    lf = V.build_large(big, ev.median_of)
    assert len(lf.data) == 6 and [t.name for t in lf.data if t.showlegend] == ["bestes von 20 Starts", "mittlerer Start", "Schwerpunkt"] and _locked(lf)


def test_charts_skip_families_without_rows():
    by = _by()
    by["star"] = []
    assert len(V.build_rounding_bars(6, by, ev.median_of).data[0].x) == 5 and len(V.build_starts_bars(6, by, ev.median_of).data[0].x) == 5 and len(V.build_best_hits({6: by}).data[0].x) == 5


def test_board_chart_for_every_family_and_size(hand):
    for fam in C.FAMILIES:
        for n in (6, 9, 12, 20):
            assert len(V.build_board(I.make_instance(fam, n, 3), list(range(n)), fam).data[-1].x) == n
    assert len(V.build_board(hand, [0, 1, 2], "Hand").data[-1].x) == 3


def test_the_pdf_export_returns_a_pdf_for_an_exact_and_a_large_instance(land6):
    s = ev.Settings("netlist", 6, 35)
    assert build_pdf(s, ev.instance_data(s, land6[1]))[:5] == b"%PDF-"
    s2 = ev.Settings("uniform", 20, 35)
    data = build_pdf(s2, ev.instance_data(s2))
    assert data[:5] == b"%PDF-" and len(data) > 1500
