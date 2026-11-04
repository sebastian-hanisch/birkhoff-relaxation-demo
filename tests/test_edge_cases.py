"""Randfälle (DEMO-PLAYBOOK §1b): kleinste Instanzen, Nullfluss, gleiche Flüsse, ungültige Eingaben."""

import numpy as np
import pytest

import bkr_birkhoff as Bk
import bkr_constants as C
import bkr_evaluation as ev
import bkr_instance as I
import bkr_landscape as L
from test_landscape import _inst


def test_zero_flow_gives_zero_bounds_and_trivial_runs():
    inst = _inst(np.zeros((4, 4)))
    cv = Bk.convex_bound(inst.F, inst.D)
    assert cv.cq == pytest.approx(0.0, abs=1e-12) and L.landscape(inst).opt == 0 and all(r.cost == 0 for r in Bk.random_runs(inst.F, inst.D, 1, 3))


def test_two_components_everything_is_sharp():
    inst = _inst([[0, 3], [3, 0]])
    cv = Bk.convex_bound(inst.F, inst.D)
    assert cv.rounded_cost == 3 and cv.cq <= 3 + 1e-9 and L.landscape(inst).opt == 3


def test_equal_flows_on_a_line_every_arrangement_costs_the_same():
    n = 5
    ones = np.ones((n, n), dtype=np.int16) - np.eye(n, dtype=np.int16)
    inst = _inst(ones)
    opt = L.landscape(inst).opt
    assert all(r.cost == opt for r in Bk.random_runs(inst.F, inst.D, 2, 5)) and Bk.convex_bound(inst.F, inst.D).rounded_cost == opt


def test_every_family_and_size_runs_through_the_measurement():
    for fam in C.FAMILIES:
        r = ev.row_of(fam, 6, 1, starts=5)
        assert r["cq_valid"] and r["round_cost"] >= r["ref"] and r["best_cost"] >= r["ref"] and 0 <= r["start_success"] <= 1
        big = ev.large_row(fam, 12, 1, starts=3)
        assert big["best_cost"] > 0 and not big["exact"]


def test_seed_extremes_and_rejections():
    for seed in (0, C.SEED_MAX):
        inst = I.make_instance("sparse", 6, seed)
        assert Bk.convex_bound(inst.F, inst.D, 20).cq <= L.landscape(inst).opt + 1e-7
    with pytest.raises(ValueError):
        I.make_instance("uniform", 7, 1)
    with pytest.raises(ValueError):
        I.make_instance("gaussian", 6, 1)


def test_verdict_helpers_survive_a_single_family_and_a_single_size():
    rows = ev.sweep("star", 6, seeds=(1, 2))
    assert ev.verdict_sweep(6, {"star": rows})[0] == "success" and "Stern" in ev.verdict_starts_sweep(6, {"star": rows})[1]
    big = {12: {"star": ev.sweep("star", 12, seeds=(1, 2))}}
    assert "n = 12" in ev.verdict_large(big)[1]
