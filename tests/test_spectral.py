"""Eigenwertschranke, projizierte Schranke und Zerlegung: Handwerte der drei Beispiele, Gültigkeit, Eigenschaften, Randfälle (DEMO-PLAYBOOK §1b, ORACLE-PLAYBOOK §5)."""

import math
from itertools import permutations

import numpy as np
import pytest

import bkr_bound as B
import bkr_constants as C
import bkr_cost as K
import bkr_instance as I
import bkr_landscape as L
import bkr_spectral as S


def _opt(inst):
    return min(K.cost(p, inst.F, inst.D) for p in permutations(range(inst.n)))


def test_sorted_dot_by_hand():
    """Kleinstes Skalarprodukt: (1, 2, 3) aufsteigend mit (4, 5, 6) absteigend = 1·6 + 2·5 + 3·4 = 28."""
    assert S.sorted_dot([3, 1, 2], [5, 6, 4]) == 28.0


def test_hand_example_one_by_hand(hand):
    """Drei Bauteile auf einer Zeile: Eigenwerte von F: −√26, 0, √26; von D (Zeile mit Abständen 1, 1, 2): −2, 1 − √3, 1 + √3.
    EB = ½ · (−√26 (1 + √3) + 0 · (1 − √3) + √26 (−2)) = −√26 (3 + √3)/2 ≈ −12,0644; Optimum 6; PB ≈ 5,2966 (Terme im nächsten Test)."""
    sp = S.spectral_bounds(hand.F, hand.D)
    assert sp.bkr_F == pytest.approx([-math.sqrt(26), 0, math.sqrt(26)]) and sp.bkr_D == pytest.approx([-2, 1 - math.sqrt(3), 1 + math.sqrt(3)])
    assert sp.eb == pytest.approx(-math.sqrt(26) * (3 + math.sqrt(3)) / 2, abs=1e-9) and round(sp.eb, 4) == -12.0644
    assert round(sp.pb, 4) == 5.2966 and _opt(hand) == 6


def test_hand_example_one_terms_by_hand(hand):
    """u = F e = (5, 6, 1), w = D e = (3, 2, 3), s_F = 12, s_D = 8: konstant ½ · (−12 · 8 / 9) = −16/3; linear ½ · (2/3) · (u aufsteigend (1, 5, 6) mit w absteigend (3, 3, 2)) = ⅓ · (3 + 15 + 12) = 10."""
    sp = S.spectral_bounds(hand.F, hand.D)
    assert sp.const == pytest.approx(-16 / 3) and sp.linear == pytest.approx(10.0) and sp.pb == pytest.approx(sp.const + sp.linear + sp.quad)


def test_hand_examples_two_and_three(hand_grid, hand_gap):
    g = S.spectral_bounds(hand_grid.F, hand_grid.D)
    assert round(g.eb, 4) == -17.3044 and round(g.pb, 4) == 9.8846 and _opt(hand_grid) == 10
    h = S.spectral_bounds(hand_gap.F, hand_gap.D)
    assert round(h.eb, 4) == -8.5765 and round(h.pb, 4) == 5.5858 and _opt(hand_gap) == 6 and h.pb > B.gilmore_lawler(hand_gap.F, hand_gap.D).glb          # über der Gilmore–Lawler-Schranke 5


def test_the_complement_basis_is_orthonormal_and_orthogonal_to_ones():
    for n in (2, 3, 6, 9, 20):
        V = S.complement_basis(n)
        assert V.shape == (n, n - 1) and np.allclose(V.T @ V, np.eye(n - 1)) and np.allclose(V.T @ np.ones(n), 0)
    with pytest.raises(ValueError):
        S.complement_basis(1)


def test_the_decomposition_is_an_identity_for_every_permutation_of_small_instances():
    inst = I.make_instance("skewed", 6, 3)
    for p in list(permutations(range(6)))[::23]:
        left, right = S.decomposition_terms(inst.F, inst.D, p)
        assert left == pytest.approx(right, abs=1e-8)
        assert left == pytest.approx(2 * K.cost(p, inst.F, inst.D))


def test_the_projected_bound_does_not_depend_on_the_choice_of_the_basis():
    """Jede andere orthonormale Basis des Raums senkrecht zu e (V · Q mit orthogonalem Q) gibt dieselben Eigenwerte der projizierten Matrizen und denselben Wert."""
    inst = I.make_instance("cluster", 8, 4)
    F, D = inst.F.astype(float), inst.D.astype(float)
    V = S.complement_basis(8)
    Q, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(7, 7)))
    W = V @ Q
    assert np.allclose(S.eigenvalues(V.T @ F @ V), S.eigenvalues(W.T @ F @ W)) and np.allclose(S.eigenvalues(V.T @ D @ V), S.eigenvalues(W.T @ D @ W))


def test_eigenvalues_satisfy_trace_and_norm_identities():
    inst = I.make_instance("uniform", 8, 2)
    for M in (inst.F.astype(float), inst.D.astype(float)):
        lam = S.eigenvalues(M)
        assert lam.sum() == pytest.approx(np.trace(M), abs=1e-9) and (lam ** 2).sum() == pytest.approx((M ** 2).sum()) and list(lam) == sorted(lam)


@pytest.mark.parametrize("family", C.FAMILIES)
def test_bounds_are_valid_and_ordered_for_every_family(family):
    for seed in range(6):
        inst = I.make_instance(family, 6, seed)
        sp = S.spectral_bounds(inst.F, inst.D)
        opt = _opt(inst)
        assert sp.eb <= opt + 1e-7 and sp.pb <= opt + 1e-7 and sp.eb <= sp.pb + 1e-7 and sp.pb == pytest.approx(sp.const + sp.linear + sp.quad)


def test_every_permutation_costs_at_least_the_projected_bound():
    inst = I.make_instance("star", 6, 5)
    sp = S.spectral_bounds(inst.F, inst.D)
    assert all(K.cost(p, inst.F, inst.D) >= sp.pb - 1e-9 and K.cost(p, inst.F, inst.D) >= sp.eb - 1e-9 for p in permutations(range(6)))


def test_the_bounds_are_invariant_under_relabelling_and_scale_with_the_flow():
    inst = I.make_instance("cluster", 8, 5)
    rng = np.random.default_rng(0)
    q, r = rng.permutation(8), rng.permutation(8)
    base = S.spectral_bounds(inst.F, inst.D)
    a = S.spectral_bounds(inst.F[np.ix_(q, q)], inst.D)
    b = S.spectral_bounds(inst.F, inst.D[np.ix_(r, r)])
    assert (a.eb, a.pb) == pytest.approx((base.eb, base.pb)) and (b.eb, b.pb) == pytest.approx((base.eb, base.pb))
    c = S.spectral_bounds(3 * inst.F, inst.D)
    assert (c.eb, c.pb) == pytest.approx((3 * base.eb, 3 * base.pb))


def test_the_projected_bound_is_sharp_for_equal_flows_on_a_line():
    """Gleicher Fluss zwischen allen Paaren: alle Anordnungen kosten gleich; die konstante und die lineare Schranke sind dann scharf, F̂ ist ein Vielfaches der Einheitsmatrix."""
    n = 5
    ones = np.ones((n, n), dtype=np.int16) - np.eye(n, dtype=np.int16)
    D = I.manhattan(I.grid_xy(1, n))
    sp = S.spectral_bounds(ones, D)
    assert sp.pb == pytest.approx(_opt_of(ones, D), abs=1e-9)


def _opt_of(F, D):
    return min(K.cost(p, F, D) for p in permutations(range(len(F))))


def test_the_eigenvalue_bound_is_negative_for_the_nonnegative_hand_examples(hand, hand_grid, hand_gap):
    assert all(S.spectral_bounds(i.F, i.D).eb < 0 for i in (hand, hand_grid, hand_gap))


def test_products_are_the_summands_of_the_eigenvalue_bound(hand):
    sp = S.spectral_bounds(hand.F, hand.D)
    assert sp.products.sum() == pytest.approx(sp.eb) and len(sp.products) == 3


def test_rejections():
    with pytest.raises(ValueError):
        S.eigenvalues(np.ones((2, 3)))
    with pytest.raises(ValueError):
        S.eigenvalues(np.array([[0.0, 1.0], [2.0, 0.0]]))
    with pytest.raises(ValueError):
        S.permutation_matrix([0, 0, 1])
    with pytest.raises(ValueError):
        S.decomposition_terms(np.eye(3), np.eye(3), [0, 1])


def test_two_components_work():
    F = np.array([[0, 3], [3, 0]])
    D = np.array([[0, 2], [2, 0]])
    sp = S.spectral_bounds(F, D)
    assert sp.pb == pytest.approx(6.0) and sp.eb == pytest.approx(-6.0)           # Eigenwerte von F: ±3, von D: ±2 → ½ (−3 · 2 + 3 · (−2)) = −6; die Projektion ist scharf (6 = Optimum)


def test_the_pb_never_exceeds_the_optimum_on_netlists_and_enumeration_agrees(land6):
    inst, ls = land6
    sp = S.spectral_bounds(inst.F, inst.D)
    assert sp.pb <= ls.opt + 1e-7 and L.landscape is not None
