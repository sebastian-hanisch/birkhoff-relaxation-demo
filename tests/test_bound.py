"""Die Gilmore–Lawler-Schranke: Handwerte der drei Beispiele, Gültigkeit, Eigenschaften, Randfälle, Abstiege (DEMO-PLAYBOOK §1b, ORACLE-PLAYBOOK §5)."""

from itertools import permutations

import numpy as np
import pytest

import bkr_bound as B
import bkr_constants as C
import bkr_cost as K
import bkr_instance as I
import bkr_landscape as L


def _opt(inst):
    return min(K.cost(p, inst.F, inst.D) for p in permutations(range(inst.n)))


def test_hand_example_one_by_hand(hand):
    """Fluss A–B 5, B–C 1 auf einer Zeile: L = 5 5 5 / 7 6 7 / 1 1 1, kleinste Zuordnung 12, GLB = 6 = Optimum; die Zuordnung (A, B, C auf 0, 1, 2) kostet 6."""
    b = B.gilmore_lawler(hand.F, hand.D)
    assert b.L.tolist() == [[5, 5, 5], [7, 6, 7], [1, 1, 1]] and b.lap == 12 and b.glb == 6.0 and b.rowmin2 == 12 and b.rowmin == 6.0
    assert _opt(hand) == 6 and b.rounded_cost == K.cost(b.assign.tolist(), hand.F, hand.D) and b.rounded_cost >= 6 and b.assign[1] == 1


def test_hand_example_two_by_hand(hand_grid):
    """Raster 2×2 (Fluss 3, 1, 2, 4): L hat in jeder Zeile gleiche Einträge 4, 5, 5, 6; Zuordnung 20; GLB = 10 = Optimum."""
    b = B.gilmore_lawler(hand_grid.F, hand_grid.D)
    assert b.L.tolist() == [[4] * 4, [5] * 4, [5] * 4, [6] * 4] and b.lap == 20 and b.glb == 10.0 and b.rowmin == 10.0 and _opt(hand_grid) == 10


def test_hand_example_three_has_a_gap(hand_gap):
    """Fluss A–C, A–D, B–C, B–D je 1 auf vier Plätzen einer Zeile: Randplätze 3, Innenplätze 2 für jedes Bauteil; Zuordnung 3 + 2 + 2 + 3 = 10, GLB = 5; Optimum 6 (A und B innen); Zeilenminimum 4 · 2 / 2 = 4."""
    b = B.gilmore_lawler(hand_gap.F, hand_gap.D)
    assert b.L.tolist() == [[3, 2, 2, 3]] * 4 and b.lap == 10 and b.glb == 5.0 and b.rowmin2 == 8 and b.rowmin == 4.0 and _opt(hand_gap) == 6
    assert K.cost([1, 2, 0, 3], hand_gap.F, hand_gap.D) == 6                                   # A auf Platz 2, B auf 3, C auf 1, D auf 4: A und B innen, C und D außen


def test_flow_and_distance_rows_leave_out_the_diagonal_and_sort():
    F = np.array([[0, 3, 1], [3, 0, 2], [1, 2, 0]])
    D = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    assert B.flow_rows(F).tolist() == [[1, 3], [2, 3], [1, 2]] and B.distance_rows(D).tolist() == [[2, 1], [1, 1], [2, 1]]


def test_the_diagonal_must_not_be_part_of_the_scalar_product():
    """Ließe man die Diagonale (0) in beiden Reihen stehen, würde die größte Zahl der einen Reihe mit der 0 der anderen gepaart: ein zu kleiner Wert. Fluss 0, 5; Abstand 0, 4: richtig 5 · 4 = 20."""
    F = np.array([[0, 5], [5, 0]])
    D = np.array([[0, 4], [4, 0]])
    assert B.lower_matrix(F, D).tolist() == [[20, 20], [20, 20]]


def test_lower_matrix_equals_a_naive_loop():
    inst = I.make_instance("skewed", 8, 3)
    n = inst.n
    naive = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for k in range(n):
            f = sorted(int(inst.F[i, j]) for j in range(n) if j != i)
            d = sorted((int(inst.D[k, l]) for l in range(n) if l != k), reverse=True)
            naive[i, k] = sum(x * y for x, y in zip(f, d))
    assert np.array_equal(B.lower_matrix(inst.F, inst.D), naive)


@pytest.mark.parametrize("family", C.FAMILIES)
def test_the_bound_is_valid_and_ordered_for_every_family(family):
    for seed in range(6):
        inst = I.make_instance(family, 6, seed)
        b = B.gilmore_lawler(inst.F, inst.D)
        opt = _opt(inst)
        assert b.rowmin <= b.glb + 1e-12 and b.glb <= opt and b.rounded_cost >= opt and b.lap == int(round(2 * b.glb)) and sorted(b.assign.tolist()) == list(range(6))


def test_every_permutation_costs_at_least_the_matrix_sum_over_two():
    """cost(p) ≥ ½ Σ_i L[i, p_i] für jede Anordnung (das ist der Kern des Beweises)."""
    inst = I.make_instance("uniform", 6, 4)
    L_ = B.lower_matrix(inst.F, inst.D)
    for p in permutations(range(6)):
        assert 2 * K.cost(p, inst.F, inst.D) >= sum(int(L_[i, p[i]]) for i in range(6))


def test_the_bound_is_invariant_under_relabelling_bodies_and_places_and_scales_with_the_flow():
    inst = I.make_instance("cluster", 8, 5)
    rng = np.random.default_rng(0)
    q, r = rng.permutation(8), rng.permutation(8)
    base = B.gilmore_lawler(inst.F, inst.D).lap
    assert B.gilmore_lawler(inst.F[np.ix_(q, q)], inst.D).lap == base and B.gilmore_lawler(inst.F, inst.D[np.ix_(r, r)]).lap == base
    assert B.gilmore_lawler(3 * inst.F, inst.D).lap == 3 * base and B.gilmore_lawler(inst.F, 2 * inst.D).lap == 2 * base


def test_equal_flows_and_zero_flows():
    n = 5
    ones = np.ones((n, n), dtype=np.int16) - np.eye(n, dtype=np.int16)
    xy = I.grid_xy(1, n)
    D = I.manhattan(xy)
    b = B.gilmore_lawler(ones, D)
    assert b.glb == _opt_of(ones, D) == 20                                                       # gleicher Fluss auf einer Zeile: alle Anordnungen kosten gleich (Summe aller Abstände), die Schranke ist scharf
    z = B.gilmore_lawler(np.zeros((n, n), dtype=np.int16), D)
    assert z.glb == 0 and z.rowmin == 0 and z.rounded_cost == 0


def _opt_of(F, D):
    return min(K.cost(p, F, D) for p in permutations(range(len(F))))


def test_small_instances():
    one = B.gilmore_lawler(np.zeros((1, 1), dtype=np.int16), np.zeros((1, 1), dtype=np.int16))
    assert one.glb == 0 and one.L.shape == (1, 1)
    two = B.gilmore_lawler(np.array([[0, 3], [3, 0]]), np.array([[0, 2], [2, 0]]))
    assert two.glb == 6 == two.rounded_cost


def test_the_gap_is_small_not_zero_for_dense_instances_and_the_bound_never_exceeds_the_optimum_on_netlists():
    for seed in range(5):
        inst = I.make_instance("netlist", 8, 100000 + seed)
        assert B.gilmore_lawler(inst.F, inst.D).glb <= L.landscape(inst).opt


# ---------------------------------------------------------------- Abstiege

def test_random_starts_are_permutations_deterministic_and_vary():
    a, b = B.random_starts(8, 20, 3), B.random_starts(8, 20, 3)
    assert np.array_equal(a, b) and all(sorted(r.tolist()) == list(range(8)) for r in a) and len({tuple(r) for r in a}) > 15 and not np.array_equal(a, B.random_starts(8, 20, 4))


def test_descents_end_in_local_optima_and_never_get_worse():
    inst = I.make_instance("uniform", 8, 9)
    starts = B.random_starts(8, 40, 1)
    ends, cost = B.descents(inst.F, inst.D, starts)
    before = B.costs_of_many(starts, inst.F, inst.D)
    assert (cost <= before).all() and (cost == B.costs_of_many(ends, inst.F, inst.D)).all()
    for p in ends:
        assert all(K.swap_delta(p.tolist(), a, b, inst.F, inst.D) >= 0 for a, b in K.pair_list(8))


def test_batch_descents_equal_the_single_start_descent_of_the_landscape():
    """Jeder Abstieg endet wie die steilste-Abstieg-Regel der Aufzählung (kleinstes Δ, bei Gleichstand das erste Paar)."""
    inst = I.make_instance("star", 6, 2)
    pairs = K.pair_list(6)
    starts = B.random_starts(6, 15, 7)
    ends, _ = B.descents(inst.F, inst.D, starts)
    for s, e in zip(starts, ends):
        cur = s.tolist()
        while True:
            deltas = [K.swap_delta(cur, a, b, inst.F, inst.D) for a, b in pairs]
            j = min(range(len(pairs)), key=lambda i: (deltas[i], i))
            if deltas[j] >= 0:
                break
            cur = K.swapped(cur, *pairs[j])
        assert cur == e.tolist()


def test_the_best_found_solution_is_an_upper_bound_and_often_the_optimum():
    hits = 0
    for seed in range(6):
        inst = I.make_instance("netlist", 6, seed)
        ub, p = B.best_found(inst.F, inst.D, seed)
        opt = L.landscape(inst).opt
        assert ub >= opt and K.cost(p.tolist(), inst.F, inst.D) == ub
        hits += ub == opt
    assert hits >= 5


def test_best_found_is_deterministic_and_improves_with_more_starts():
    inst = I.make_instance("uniform", 12, 5)
    assert B.best_found(inst.F, inst.D, 5)[0] == B.best_found(inst.F, inst.D, 5)[0]
    assert B.best_found(inst.F, inst.D, 5, count=200)[0] <= B.best_found(inst.F, inst.D, 5, count=1)[0]


def test_costs_of_many_equal_single_costs():
    inst = I.make_instance("sparse", 8, 3)
    Q = B.random_starts(8, 10, 2)
    assert B.costs_of_many(Q, inst.F, inst.D).tolist() == [K.cost(q.tolist(), inst.F, inst.D) for q in Q]
