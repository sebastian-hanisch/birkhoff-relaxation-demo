"""Relaxation auf das Birkhoff-Polytop: Frank–Wolfe, Verschiebung, zertifizierte Schranke und FAQ-Art: Handwerte, Eigenschaften, Randfälle (DEMO-PLAYBOOK §1b, ORACLE-PLAYBOOK §5)."""

from itertools import permutations

import numpy as np
import pytest

import bkr_birkhoff as Bk
import bkr_constants as C
import bkr_cost as K
import bkr_instance as I
import bkr_landscape as L


def _opt(inst):
    return min(K.cost(p, inst.F, inst.D) for p in permutations(range(inst.n)))


def _pm(p):
    n = len(p)
    X = np.zeros((n, n))
    X[np.arange(n), p] = 1.0
    return X


def test_barycenter_and_random_starts_are_doubly_stochastic_inside_and_deterministic():
    b = Bk.barycenter(5)
    assert b.shape == (5, 5) and np.allclose(b, 0.2) and Bk.is_doubly_stochastic(b)
    x = Bk.random_start(6, 17)
    assert Bk.is_doubly_stochastic(x) and x.min() >= (1 - C.LAMBDA) / 6 - 1e-12 and np.array_equal(x, Bk.random_start(6, 17)) and not np.array_equal(x, Bk.random_start(6, 18))
    assert np.allclose(Bk.random_start(4, 3, lam=1.0).sum(axis=0), 1) and set(np.unique(Bk.random_start(4, 3, lam=1.0))) == {0.0, 1.0}
    assert not Bk.is_doubly_stochastic(np.array([[1.0, 0.5], [0.0, 0.5]])) and not Bk.is_doubly_stochastic(np.array([[1.5, -0.5], [-0.5, 1.5]]))


def test_lap_min_by_hand_and_against_enumeration():
    M = np.array([[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]])
    Y, cols = Bk.lap_min(M)
    assert cols.tolist() == [1, 0, 2] and (Y * M).sum() == 5.0 and Y.sum() == 3
    rng = np.random.default_rng(0)
    for _ in range(5):
        a = rng.random((5, 5))
        Y, cols = Bk.lap_min(a)
        assert (Y * a).sum() == pytest.approx(min(sum(a[i, p[i]] for i in range(5)) for p in permutations(range(5))))


def test_rounding_gives_back_a_permutation_matrix_and_a_valid_permutation_otherwise():
    assert Bk.round_to_perm(_pm([2, 0, 3, 1])) == [2, 0, 3, 1]
    p = Bk.round_to_perm(Bk.barycenter(5))
    assert sorted(p) == list(range(5))
    X = np.array([[0.7, 0.3, 0.0], [0.2, 0.5, 0.3], [0.1, 0.2, 0.7]])
    assert Bk.round_to_perm(X) == [0, 1, 2]


def test_the_objective_equals_the_cost_at_permutation_matrices(land6):
    inst, _ = land6
    for p in list(permutations(range(6)))[::97]:
        assert Bk.objective(inst.F.astype(float), inst.D.astype(float), _pm(list(p))) == pytest.approx(K.cost(list(p), inst.F, inst.D))


def test_frank_wolfe_keeps_double_stochasticity_decreases_the_value_and_the_gap_is_nonnegative():
    inst = I.make_instance("uniform", 6, 4)
    Fm, Dm = inst.F.astype(float), inst.D.astype(float)
    for seed in range(5):
        res = Bk.frank_wolfe(Fm, Dm, Bk.random_start(6, seed), 60)
        assert Bk.is_doubly_stochastic(res.X) and res.steps <= 60 and res.gap >= 0 and min(res.gaps) >= 0
        assert all(res.values[i + 1] <= res.values[i] + 1e-9 for i in range(len(res.values) - 1)) and len(res.values) in (res.steps + 1, res.steps + 1)
        assert all(res.lowers[i + 1] >= res.lowers[i] - 1e-12 for i in range(len(res.lowers) - 1))


def test_the_barycenter_is_stationary_exactly_when_the_row_sums_of_flow_or_distance_are_constant():
    """Der Gradient am Schwerpunkt ist u wᵀ/n² (u = F e, w = D e); ⟨∇, P⟩ = Σ u_i w_{p_i}/n² ist genau dann für alle Permutationen gleich, wenn u oder w konstant ist."""
    for family in C.FAMILIES:
        for seed in range(6):
            inst = I.make_instance(family, 6, seed)
            u, w = inst.F.sum(axis=1), inst.D.sum(axis=1)
            res = Bk.frank_wolfe(inst.F, inst.D, Bk.barycenter(6), 5)
            constant = len(set(u.tolist())) == 1 or len(set(w.tolist())) == 1
            assert (res.gaps[0] < 1e-9) == constant


def test_the_hand_examples_two_and_three_have_a_stationary_barycenter(hand_grid, hand_gap, hand):
    for inst in (hand_grid, hand_gap):
        run = Bk.faq_run(inst.F, inst.D, Bk.barycenter(inst.n))
        assert run.gap0 == 0 and run.steps == 0 and np.allclose(run.X, 1 / inst.n)
    assert Bk.faq_run(hand.F, hand.D, Bk.barycenter(3)).gap0 > 0.1
    assert Bk.faq_run(hand_gap.F, hand_gap.D, Bk.barycenter(4)).cost == 8 and _opt(hand_gap) == 6          # Rundung des stationären Schwerpunkts: Kosten 8 statt 6


def test_the_shifts_make_both_matrices_positive_semidefinite_and_are_nonnegative(land6):
    inst, _ = land6
    sf, sd = Bk.shifts(inst.F, inst.D)
    assert sf > 0 and sd > 0
    for M, s in ((inst.F, sf), (inst.D, sd)):
        w = np.linalg.eigvalsh(M.astype(float) + s * np.eye(6))
        assert w.min() == pytest.approx(0.0, abs=1e-9) and w.max() > 0


def test_the_shifted_objective_equals_the_cost_plus_the_constant_for_every_permutation():
    """g'(P) = cost(P) + ½ s_F s_D n für jede Permutation (die Verschiebung ändert auf Ecken nur eine Konstante)."""
    inst = I.make_instance("star", 6, 3)
    sf, sd = Bk.shifts(inst.F, inst.D)
    const = 0.5 * sf * sd * 6
    for p in list(permutations(range(6)))[::23]:
        assert Bk.convex_value_at_permutation(inst.F, inst.D, list(p)) == pytest.approx(K.cost(list(p), inst.F, inst.D) + const)


def test_the_shifted_objective_is_convex_on_random_doubly_stochastic_pairs():
    inst = I.make_instance("cluster", 6, 2)
    Fm, Dm = inst.F.astype(float), inst.D.astype(float)
    sf, sd = Bk.shifts(Fm, Dm)
    Fp, Dp = Fm + sf * np.eye(6), Dm + sd * np.eye(6)
    for s in range(10):
        a, b = Bk.random_start(6, s, 0.9), Bk.random_start(6, s + 100, 0.9)
        mid = Bk.objective(Fp, Dp, 0.5 * (a + b))
        assert mid <= 0.5 * (Bk.objective(Fp, Dp, a) + Bk.objective(Fp, Dp, b)) + 1e-9


def test_closed_form_for_the_hand_examples_with_a_stationary_barycenter(hand_grid, hand_gap):
    """Ist der Schwerpunkt stationär, ist er das Minimum der konvexen Relaxation: g'(J/n) = ½ (eᵀF'e)(eᵀD'e)/n², CQ = g'(J/n) − ½ s_F s_D n.
    Vierer-Zeile: eᵀF'e = 8 + 4 · 2 = 16, eᵀD'e = 20 + 4 · (2 + √2); Raster 2×2: CQ ≈ 9,8833."""
    for inst, expected in ((hand_gap, 3.1716), (hand_grid, 9.8833)):
        n = inst.n
        sf, sd = Bk.shifts(inst.F, inst.D)
        closed = 0.5 * (inst.F.sum() + n * sf) * (inst.D.sum() + n * sd) / n**2 - 0.5 * sf * sd * n
        cv = Bk.convex_bound(inst.F, inst.D)
        assert cv.cq == pytest.approx(closed, abs=1e-9) and round(cv.cq, 4) == expected and np.allclose(cv.X, 1 / n) and cv.interior == pytest.approx(1 / n)
    assert Bk.shifts(hand_gap.F, hand_gap.D)[0] == pytest.approx(2.0) and Bk.shifts(hand_gap.F, hand_gap.D)[1] == pytest.approx(2 + np.sqrt(2))


def test_hand_example_one_by_hand(hand):
    """Drei Bauteile auf einer Zeile: s_F = √26, s_D = 2, Konstante ½ · √26 · 2 · 3 = 3√26 ≈ 15,30; CQ ≈ 5,69 (konvergiert langsam: 5,6875 nach 300, 5,6880 nach 2000 Schritten) ≤ Optimum 6; die Rundung ist optimal."""
    cv = Bk.convex_bound(hand.F, hand.D)
    assert cv.s_f == pytest.approx(np.sqrt(26)) and cv.s_d == pytest.approx(2.0) and cv.constant == pytest.approx(3 * np.sqrt(26)) and 5.68 < cv.cq <= 5.69 and cv.cq <= 6 and cv.rounded_cost == 6


@pytest.mark.parametrize("family", C.FAMILIES)
def test_the_certified_bound_is_valid_monotone_and_below_the_rounding(family):
    for seed in range(100000, 100004):
        inst = I.make_instance(family, 6, seed)
        ls = L.landscape(inst)
        cv = Bk.convex_bound(inst.F, inst.D)
        assert cv.cq <= ls.opt + 1e-7 and cv.rounded_cost >= ls.opt and cv.rounded_cost == K.cost(cv.perm, inst.F, inst.D) and Bk.is_doubly_stochastic(cv.X)
        assert all(cv.lowers[i + 1] >= cv.lowers[i] - 1e-9 for i in range(len(cv.lowers) - 1)) and cv.lowers[-1] == pytest.approx(cv.cq) and all(c >= ls.opt for c in cv.rounded)
        assert len(cv.lowers) == len(cv.rounded) and 0 < cv.interior <= 1


def test_faq_runs_are_valid_permutations_with_consistent_costs_and_deterministic():
    inst = I.make_instance("netlist", 8, 35)
    runs = Bk.random_runs(inst.F, inst.D, 35, 5)
    again = Bk.random_runs(inst.F, inst.D, 35, 5)
    assert len(runs) == 5 and [r.cost for r in runs] == [r.cost for r in again]
    for r in runs:
        assert sorted(r.perm) == list(range(8)) and r.cost == K.cost(r.perm, inst.F, inst.D) and r.steps <= C.FW_ITERS and Bk.is_doubly_stochastic(r.X)
    assert min(r.cost for r in runs) >= L.landscape(inst).opt


def test_a_run_from_a_permutation_start_is_valid_and_never_increases_the_relaxed_objective():
    """Ecken der Relaxation sind hier fast nie stationär (die Frank–Wolfe-Lücke ist an jeder Ecke der Handbeispiele größer als null): ein Lauf von einer Ecke bewegt sich, und der relaxierte Zielwert steigt nie."""
    inst = I.make_instance("star", 6, 100000)
    for p in list(permutations(range(6)))[::97]:
        run = Bk.faq_run(inst.F, inst.D, _pm(list(p)))
        assert run.gap0 > 0 and run.steps >= 1 and run.values[-1] <= run.values[0] + 1e-9 and sorted(run.perm) == list(range(6))


def test_edge_cases():
    F = np.array([[0, 3], [3, 0]])
    D = np.array([[0, 2], [2, 0]])
    cv = Bk.convex_bound(F, D)
    assert cv.cq <= 6 + 1e-9 and cv.rounded_cost == 6 and Bk.is_doubly_stochastic(cv.X)
    z = Bk.convex_bound(np.zeros((4, 4)), np.zeros((4, 4)))
    assert z.cq == pytest.approx(0.0, abs=1e-12) and z.constant == 0 and z.rounded_cost == 0
    one = Bk.faq_run(np.zeros((1, 1)), np.zeros((1, 1)), Bk.barycenter(1))
    assert one.perm == [0] and one.cost == 0
    with pytest.raises(ValueError):
        K.cost([0, 0], F, D)


# ----------------------------------------------------------------------------- rechnerunabhängige Gleichstände

def test_tie_quantized_makes_noise_level_ties_exact_and_leaves_zero_alone():
    M = np.array([[1.0, 1.0 + 1e-13], [2.0, 3.0]])
    Q = Bk.tie_quantized(M)
    assert Q[0, 0] == Q[0, 1] and Q[1, 1] > Q[1, 0] > Q[0, 0] and Q.max() == pytest.approx(1.0)
    assert np.array_equal(Bk.tie_quantized(np.zeros((3, 3))), np.zeros((3, 3)))
    assert Bk.tie_quantized(np.array([[1.0, 2.0], [3.0, 4.0]]), 1)[0, 0] == 0.2


def _noisy_ties(monkeypatch, noise, seed):
    """Stört die Eingabe jeder gerundeten Zuordnung (Gradient und Lösung) um relatives Rauschen `noise`: der Ersatz für einen anderen Rechner (anderes BLAS, andere Summationsreihenfolge)."""
    rng = np.random.default_rng(seed)
    orig = Bk.tie_quantized
    monkeypatch.setattr(Bk, "tie_quantized", lambda M, decimals=C.TIE_DECIMALS: orig(np.asarray(M, float) + noise * np.abs(M).max() * rng.standard_normal(np.shape(M)), decimals))


@pytest.mark.parametrize("family,n,seed", [("netlist", 8, 35), ("uniform", 8, 100000), ("cluster", 8, 100001), ("sparse", 6, 100002)])
def test_the_convex_run_and_its_rounding_do_not_depend_on_rounding_noise(monkeypatch, family, n, seed):
    inst = I.make_instance(family, n, seed)
    base = Bk.convex_bound(inst.F, inst.D)
    for s in range(3):
        _noisy_ties(monkeypatch, 1e-13, s)
        other = Bk.convex_bound(inst.F, inst.D)
        monkeypatch.undo()
        assert other.rounded_cost == base.rounded_cost and other.perm == base.perm and other.cq == pytest.approx(base.cq, abs=1e-6) and other.rounded == base.rounded


def test_the_tie_digits_change_only_the_rounding_not_the_bound(monkeypatch):
    inst = I.make_instance("netlist", 8, 35)
    out = {}
    for d in (6, 8, 10):
        monkeypatch.setattr(Bk.tie_quantized, "__defaults__", (d,))
        cv = Bk.convex_bound(inst.F, inst.D)
        out[d] = (cv.rounded_cost, cv.cq)
        monkeypatch.undo()
    assert [out[d][0] for d in (6, 8, 10)] == [46, 54, 37]
    assert max(out[d][1] for d in out) - min(out[d][1] for d in out) < 0.05 and all(v[1] <= 31 for v in out.values())
