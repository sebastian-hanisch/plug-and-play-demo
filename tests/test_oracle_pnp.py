"""Orakel: jede Einheit gegen einen anderen Rechenweg (die Löser aus Stück 6 und 7, die SVD-Form der Landweber-Iteration, KKT-Bedingungen) und gegen die Mini-Instanz.
Linux-CI: in der letzten Stelle nichts festlegen; Schranken statt Einzelwerte."""

import math

import numpy as np
import pytest

import pnp_constants as C
import pnp_evaluation as ev
import pnp_model as m
from inverse_generator import SplitMix64, grid


def _ill():
    x, b, sigma, _ = ev.instance("phantom", 1e-2, 1000)
    return m.op(), x, b, sigma


# ----------------------------------------------------------------------------- 1. Prox-Denoiser: PnP ist ISTA (Stück 6)

def test_soft_threshold_pnp_equals_the_exact_lasso_on_a_wellposed_matrix():
    aw = m.wellposed_matrix(C.N)
    x = ev.truth_signal("phantom")
    bw = aw @ x + SplitMix64(5).normal(C.N) * 0.01
    tw = m.step_size(aw)
    for lam in (1e-3, 1e-2, 1e-1):
        xk = m.pnp_run(aw, bw[None, :], m.make_denoiser("soft", C.N, [tw * lam]), [C.WP_STEPS])[0][0, 0]
        assert xk == pytest.approx(m.lasso(aw, bw, lam), abs=1e-9)


def test_soft_threshold_pnp_objective_gap_is_never_negative_and_falls_on_the_ill_posed_operator():
    a, x, b, _ = _ill()
    lam = 1e-3
    t = m.step_size(a)
    ex = m.lasso(a, b, lam)
    fex = m.objective_l1(a, b, ex, lam)
    xs, _ = m.pnp_run(a, b[None, :], m.make_denoiser("soft", C.N, [t * lam]), [10, 100, 1000])
    gaps = [m.objective_l1(a, b, xs[i, 0], lam) - fex for i in range(3)]
    assert all(g >= -1e-12 for g in gaps) and gaps[0] > gaps[1] > gaps[2] and gaps[2] < 1e-4


def test_lasso_satisfies_the_kkt_conditions():
    rng = np.random.default_rng(8)
    for mu in (0.5, 2.0, 5.0):
        a = rng.normal(size=(20, 12))
        y = rng.normal(size=20)
        w = m.lasso(a, y, mu)
        g = a.T @ (a @ w - y)
        assert np.allclose(g[w != 0.0], -mu * np.sign(w[w != 0.0]), atol=1e-8) and (np.abs(g[w == 0.0]) <= mu + 1e-8).all()


# ----------------------------------------------------------------------------- 2. Haar: orthonormal, Prox, Zielfunktion fällt

def test_haar_matrix_is_orthonormal_and_reconstructs():
    w = m.haar_matrix(C.N, C.HAAR_LEVELS)
    assert np.abs(w @ w.T - np.eye(C.N)).max() < 1e-14
    v = np.random.default_rng(3).normal(size=C.N)
    assert w.T @ (w @ v) == pytest.approx(v, abs=1e-13)


def test_haar_denoiser_satisfies_the_prox_conditions_in_the_coefficient_domain():
    w = m.haar_matrix(C.N, C.HAAR_LEVELS)
    k = m.n_approx(C.N, C.HAAR_LEVELS)
    v = np.random.default_rng(4).normal(size=C.N)
    th = 0.4
    out = m.make_denoiser("haar", C.N, [th])(v[None, :])[0]
    c, co = w @ v, w @ out
    assert co[:k] == pytest.approx(c[:k], abs=1e-12)                                      # die Näherung bleibt
    d, do = c[k:], co[k:]
    nz = np.abs(do) > 1e-12
    assert nz.any() and (~nz).any()
    assert np.allclose(d[nz] - do[nz], th * np.sign(do[nz]), atol=1e-12) and (np.abs(d[~nz]) <= th + 1e-12).all()


@pytest.mark.parametrize("lam", [1e-3, 1e-2])
def test_haar_pnp_objective_falls_in_every_step_and_ends_at_a_fixed_point(lam):
    a, x, b, _ = _ill()
    t = m.step_size(a)
    den = m.make_denoiser("haar", C.N, [t * lam])
    xk = np.zeros((1, C.N))
    fs = []
    for _ in range(60):
        xk = m.pnp_step(a, b[None, :], xk, den, t)
        fs.append(m.objective_haar(a, b, xk[0], lam))
    assert all(fs[i + 1] <= fs[i] + 1e-12 for i in range(59))
    xl = m.pnp_run(a, b[None, :], den, [2000])[0][0]
    assert m.fixed_point_residual(a, b, den, xl)[0] < 1e-8


# ----------------------------------------------------------------------------- 3. Gauß-MMSE: Posterior-Mittel (Stück 7)

def test_gauss_mmse_fixed_point_is_the_closed_form_and_the_posterior_mean():
    a, x, b, sigma = _ill()
    t = m.step_size(a)
    p = m.prior_cov(grid(C.N))
    s = 3.0 * math.sqrt(t) * sigma
    xk = m.pnp_run(a, b[None, :], m.make_denoiser("gauss", C.N, [s]), [C.GAUSS_STEPS])[0][0, 0]
    assert xk == pytest.approx(m.gauss_closed_form(a, b, s, p, t), abs=1e-9)
    s1 = math.sqrt(t) * sigma
    x1 = m.pnp_run(a, b[None, :], m.make_denoiser("gauss", C.N, [s1]), [C.GAUSS_STEPS])[0][0, 0]
    assert x1 == pytest.approx(m.posterior_mean(a, b, sigma, p), abs=1e-4)
    assert np.abs(xk - m.posterior_mean(a, b, sigma, p)).max() > 0.01                      # ein anderes s ist ein anderes lambda


def test_gauss_mmse_pnp_matches_the_mini_instance_hand_value(mini):
    a, b = mini
    p = np.eye(3)
    x = m.gauss_closed_form(a, b, 0.1, p, 1.0)
    assert x == pytest.approx([1.153465, 1.653465, 1.153465], abs=1e-6)
    d = lambda v: v / 1.01                                                               # P = I: D(v) = v / (1 + s²)
    xk = np.zeros(3)
    for _ in range(20000):
        xk = d(xk - (a.T @ (a @ xk - b)))
    assert xk == pytest.approx(x, abs=1e-9) and m.posterior_mean(a, b, 0.1, p) == pytest.approx(x)


# ----------------------------------------------------------------------------- 4. Nicht-Prox-Denoiser

def test_haar_average_and_soft_are_nonexpansive_in_the_sample_and_nlm_is_not_at_large_strength():
    center = ev.truth_signal("phantom")
    assert m.lipschitz_estimate("haar_avg", C.N, 0.01, center, 5) <= 1.0 + 1e-9
    assert m.lipschitz_estimate("haar", C.N, 0.01, center, 5) <= 1.0 + 1e-9
    assert m.lipschitz_estimate("gauss", C.N, 0.01, center, 5) <= 1.0
    assert m.lipschitz_estimate("nlm", C.N, 0.1, center, 5) > 1.0 + 1e-3


def test_nlm_pnp_converges_in_practice():
    a, x, b, _ = _ill()
    den = m.make_denoiser("nlm", C.N, [0.03])
    xk = m.pnp_run(a, b[None, :], den, [3000])[0][0]
    assert m.fixed_point_residual(a, b, den, xk)[0] < 1e-3 and ev.rel_err(xk, x)[0] < 0.3


# ----------------------------------------------------------------------------- 5. Verschiebungsgleichheit

def test_haar_average_commutes_with_every_cyclic_shift_and_haar_alone_does_not():
    v = np.random.default_rng(6).normal(size=(1, C.N))
    avg, one = m.make_denoiser("haar_avg", C.N, [0.3]), m.make_denoiser("haar", C.N, [0.3])
    for k in (1, 5, 17, 40):
        assert avg(np.roll(v, k, axis=1)) == pytest.approx(np.roll(avg(v), k, axis=1), abs=1e-12)
    assert np.abs(one(np.roll(v, 1, axis=1)) - np.roll(one(v), 1, axis=1)).max() > 1e-3


# ----------------------------------------------------------------------------- 6. Schrittzahl: Landweber und Beschleunigung

def test_identity_pnp_is_the_landweber_iteration_in_svd_form():
    a, x, b, _ = _ill()
    t = m.step_size(a)
    u, sv, vt = np.linalg.svd(a)
    k = 50
    xk = m.pnp_run(a, b[None, :], m.make_denoiser("identity", C.N, [0.0]), [k])[0][0, 0]
    expect = vt.T @ (((1.0 - (1.0 - t * sv ** 2) ** k) / sv) * (u.T @ b))
    assert xk == pytest.approx(expect, abs=1e-10)


def test_the_momentum_reaches_the_exact_l1_minimizer_faster():
    a, x, b, _ = _ill()
    lam = 1e-3
    t = m.step_size(a)
    fex = m.objective_l1(a, b, m.lasso(a, b, lam), lam)
    den = m.make_denoiser("soft", C.N, [t * lam])
    plain = m.pnp_run(a, b[None, :], den, [200])[0][0, 0]
    fast = m.pnp_run(a, b[None, :], den, [200], accel=True)[0][0, 0]
    assert 0.0 <= m.objective_l1(a, b, fast, lam) - fex < m.objective_l1(a, b, plain, lam) - fex


# ----------------------------------------------------------------------------- 7. Welten

def test_world_signals_match_the_hand_values():
    sp, bl, wv, ph = (ev.truth_signal(w) for w in ("spikes", "blocks", "waves", "phantom"))
    assert np.flatnonzero(sp).tolist() == [12, 24, 33] and sp[[12, 24, 33]] == pytest.approx([0.04 * 48, 0.03 * 48, 0.05 * 48])
    assert np.flatnonzero(bl).tolist() == list(range(7, 14)) + list(range(20, 25)) + list(range(34, 41)) and bl[10] == 1.0 and bl[22] == 0.6 and bl[36] == 0.9
    assert wv[4] == 0.0 and wv[43] == 0.0 and wv[11] == pytest.approx(0.916667, abs=1e-6) and wv.max() <= 1.0 and wv[5] > 0.0 and wv[42] > 0.0
    assert ph.max() == pytest.approx(1.0, abs=0.02)
