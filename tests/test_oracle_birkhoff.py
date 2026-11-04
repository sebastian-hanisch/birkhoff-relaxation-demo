"""Orakel (ORACLE-PLAYBOOK §5): `scipy` löst das konvexe Problem unabhängig (SLSQP mit Nebenbedingungen), die Zuordnung (`linear_sum_assignment`) und die Eigenwerte; die Aufzählung ist das Orakel für die Gültigkeit der Schranke.
Ohne scipy (nur requirements-dev.txt; die App braucht scipy nicht) werden die Tests übersprungen."""

import numpy as np
import pytest

import bkr_birkhoff as Bk
import bkr_constants as C
import bkr_instance as I
import bkr_landscape as L

scipy_opt = pytest.importorskip("scipy.optimize")
linalg = pytest.importorskip("scipy.linalg")


def _slsqp_min(Fp, Dp, n, start):
    """Das konvexe Problem min ½ tr(F' X D' Xᵀ) über dem Birkhoff-Polytop mit SLSQP (unabhängig von Frank–Wolfe)."""
    f = lambda x: 0.5 * np.sum((Fp @ x.reshape(n, n) @ Dp) * x.reshape(n, n))
    grad = lambda x: (Fp @ x.reshape(n, n) @ Dp).ravel()
    cons = [{"type": "eq", "fun": lambda x, i=i: x.reshape(n, n)[i].sum() - 1.0} for i in range(n)] + [{"type": "eq", "fun": lambda x, j=j: x.reshape(n, n)[:, j].sum() - 1.0} for j in range(n)]
    res = scipy_opt.minimize(f, start.ravel(), jac=grad, bounds=[(0, 1)] * (n * n), constraints=cons, method="SLSQP", options={"maxiter": 500, "ftol": 1e-12})
    assert np.allclose(res.x.reshape(n, n).sum(0), 1, atol=1e-6) and np.allclose(res.x.reshape(n, n).sum(1), 1, atol=1e-6)
    return float(res.fun)


@pytest.mark.parametrize("family", ("uniform", "cluster", "star", "sparse"))
def test_the_frank_wolfe_bound_sandwiches_the_independent_convex_minimum(family):
    """Untere Schranke ≤ Wert jedes zulässigen Punkts (SLSQP, zulässig geprüft); der Frank–Wolfe-Endpunkt ist nicht schlechter als der allgemeine Löser."""
    for seed in (1, 2):
        inst = I.make_instance(family, 6, seed)
        n = 6
        Fm, Dm = inst.F.astype(float), inst.D.astype(float)
        sf, sd = Bk.shifts(Fm, Dm)
        Fp, Dp = Fm + sf * np.eye(n), Dm + sd * np.eye(n)
        res = Bk.frank_wolfe(Fp, Dp, Bk.barycenter(n), 300)
        slsqp = _slsqp_min(Fp, Dp, n, Bk.barycenter(n))
        obj = Bk.objective(Fp, Dp, res.X)
        # Jeder zulässige Punkt liegt über der Frank–Wolfe-Schranke; Frank–Wolfe ist höchstens 0,1 % schlechter als das allgemeine Lösungsverfahren (das früher abbrechen kann).
        assert res.lower <= slsqp + 1e-6 and res.lower <= obj + 1e-6 and obj <= slsqp * (1 + 1e-3)
        const = 0.5 * sf * sd * n
        assert res.lower - const <= L.landscape(inst).opt + 1e-7


def test_the_linear_step_equals_scipy_on_random_gradients():
    rng = np.random.default_rng(0)
    for n in (3, 5, 8):
        for _ in range(3):
            M = rng.normal(size=(n, n))
            Y, cols = Bk.lap_min(M)
            r, c = scipy_opt.linear_sum_assignment(M)
            assert (Y * M).sum() == pytest.approx(float(M[r, c].sum()), abs=1e-9)


@pytest.mark.parametrize("family", C.FAMILIES)
def test_the_shift_uses_the_scipy_eigenvalues(family):
    inst = I.make_instance(family, 8, 4)
    sf, sd = Bk.shifts(inst.F, inst.D)
    assert sf == pytest.approx(-linalg.eigh(inst.F.astype(float), eigvals_only=True).min(), abs=1e-9) and sd == pytest.approx(-linalg.eigh(inst.D.astype(float), eigvals_only=True).min(), abs=1e-9)


def test_the_bound_never_exceeds_the_enumerated_optimum_on_many_instances():
    for family in C.FAMILIES:
        for seed in range(100000, 100006):
            inst = I.make_instance(family, 6, seed)
            assert Bk.convex_bound(inst.F, inst.D, 100).cq <= L.landscape(inst).opt + 1e-7
