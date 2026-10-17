"""Einheiten von pnp_model.py, jede an der Mini-Instanz oder an einer kleinen Aufgabe von Hand gerechnet (DEMO-PLAYBOOK §1b). Orakel gegen andere Rechenwege: test_oracle_pnp.py."""

import math

import numpy as np
import pytest

import pnp_constants as C
import pnp_model as m


# ----------------------------------------------------------------------------- Denoiser

def test_soft_hand_cases():
    assert m.soft(np.array([3.0, -0.5, 1.0]), 1.0) == pytest.approx([2.0, 0.0, 0.0])
    assert m.soft(np.array([3.0, -2.0]), 0.0).tolist() == [3.0, -2.0]
    assert m.soft(np.array([-3.0, 0.5]), np.array([1.0, 1.0])) == pytest.approx([-2.0, 0.0])


def test_haar_matrix_hand_values_for_n_4_and_two_levels():
    w = m.haar_matrix(4, 2)
    s2 = 1.0 / math.sqrt(2.0)
    assert w == pytest.approx(np.array([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, -0.5, -0.5], [s2, -s2, 0.0, 0.0], [0.0, 0.0, s2, -s2]]))
    assert w @ np.array([1.0, 2.0, 3.0, 4.0]) == pytest.approx([5.0, -2.0, -s2, -s2]) and m.n_approx(4, 2) == 1 and m.n_approx(48, 4) == 3
    assert not w.flags.writeable and m.haar_matrix(4, 2) is w


def test_haar_denoiser_hand_values(monkeypatch):
    monkeypatch.setattr(C, "HAAR_LEVELS", 2)
    out = m.make_denoiser("haar", 4, [0.5])(np.array([[1.0, 2.0, 3.0, 4.0]]))[0]
    # Koeffizienten (5, -2, -0,7071, -0,7071); Details weich um 0,5: (5, -1,5, -0,2071, -0,2071); zurück: 2,5 -+ 0,75 -+ 0,2071 / sqrt(2) = 0,146447
    assert out == pytest.approx([1.603553, 1.896447, 3.103553, 3.396447], abs=1e-6)


def test_haar_avg_with_zero_threshold_is_the_identity_and_sums_to_the_mean_of_shifts():
    v = np.random.default_rng(0).normal(size=(2, 48))
    assert m.make_denoiser("haar_avg", 48, [0.0, 0.0])(v) == pytest.approx(v, abs=1e-12)
    assert m.make_denoiser("haar", 48, [0.0, 0.0])(v) == pytest.approx(v, abs=1e-12)


def test_nlm_limits_and_convexity():
    v = np.random.default_rng(1).normal(size=(2, 48))
    assert m.make_denoiser("nlm", 48, [1e-6, 1e-6])(v) == pytest.approx(v, abs=1e-6)
    big = m.make_denoiser("nlm", 48, [1e6, 1e6])(v)
    assert big == pytest.approx(np.repeat(v.mean(axis=1, keepdims=True), 48, axis=1), abs=1e-6)
    mid = m.make_denoiser("nlm", 48, [0.5, 0.5])(v)
    assert (mid <= v.max(axis=1, keepdims=True) + 1e-12).all() and (mid >= v.min(axis=1, keepdims=True) - 1e-12).all()
    const = np.full((1, 48), 2.0)
    assert m.make_denoiser("nlm", 48, [0.1])(const) == pytest.approx(const)


def test_gauss_denoiser_is_the_matrix_p_times_inverse_p_plus_s2():
    from inverse_generator import grid

    p = m.prior_cov(grid(48))
    v = np.random.default_rng(2).normal(size=(2, 48))
    s = np.array([0.05, 0.3])
    expect = np.array([p @ np.linalg.solve(p + si ** 2 * np.eye(48), vi) for vi, si in zip(v, s)])
    assert m.make_denoiser("gauss", 48, s)(v) == pytest.approx(expect, abs=1e-9)
    assert np.linalg.norm(m.make_denoiser("gauss", 48, [10.0])(v[:1])) < 0.1 * np.linalg.norm(v[0])


def test_denoisers_use_one_strength_per_row():
    v = np.array([[3.0, -0.5, 1.0], [3.0, -0.5, 1.0]])
    out = m.make_denoiser("soft", 3, [1.0, 0.1])(v)
    assert out[0] == pytest.approx([2.0, 0.0, 0.0]) and out[1] == pytest.approx([2.9, -0.4, 0.9])
    assert m.make_denoiser("identity", 3, [5.0, 5.0])(v).tolist() == v.tolist()


def test_prior_cov_and_exp_kernel_hand_values():
    k = m.exp_kernel(np.array([0.0, 0.12, 0.24]), 0.12)
    assert k[0, 1] == pytest.approx(math.exp(-1.0)) and k[0, 2] == pytest.approx(math.exp(-2.0)) and np.allclose(np.diag(k), 1.0) and np.allclose(k, k.T)
    p = m.prior_cov(np.array([0.0, 0.12]))
    assert p[0, 0] - 0.25 == pytest.approx(1e-10, rel=1e-3) and p[0, 1] == pytest.approx(0.25 * math.exp(-1.0)) and p[1, 0] == p[0, 1]


# ----------------------------------------------------------------------------- Plug-and-Play-ISTA

def test_step_size_is_one_over_the_squared_spectral_norm(mini):
    a, b = mini
    assert m.step_size(a) == pytest.approx(1.0, abs=1e-12)
    assert m.step_size(2.0 * a) == pytest.approx(0.25, abs=1e-12)


def test_one_step_with_soft_threshold_is_the_hand_value(mini):
    a, b = mini
    den = m.make_denoiser("soft", 3, [0.1])
    assert m.pnp_step(a, b[None, :], np.zeros((1, 3)), den, 1.0)[0] == pytest.approx([1.23, 1.24, 1.23])


def test_landweber_hand_values(mini):
    a, b = mini
    den = m.make_denoiser("identity", 3, [0.0])
    xs, _ = m.pnp_run(a, b, den, [1, 2])
    assert xs[0, 0] == pytest.approx([1.33, 1.34, 1.33]) and xs[1, 0] == pytest.approx([1.3267, 1.3466, 1.3267], abs=1e-4)


def test_pnp_run_shapes_order_start_and_error_history(mini):
    a, b = mini
    den = m.make_denoiser("identity", 3, [0.0, 0.0])
    bb = np.array([b, 2.0 * b])
    xs, errs = m.pnp_run(a, bb, den, [3, 1, 0], truth=np.array([1.0, 2.0, 1.0]))
    assert xs.shape == (3, 2, 3) and errs.shape == (4, 2) and (xs[2] == 0.0).all() and errs[0] == pytest.approx([1.0, 1.0])
    x1, _ = m.pnp_run(a, bb, den, [1])
    assert xs[1] == pytest.approx(x1[0]) and errs[1, 0] == pytest.approx(np.linalg.norm(x1[0, 0] - [1, 2, 1]) / np.linalg.norm([1, 2, 1]))
    x0 = np.ones((2, 3))
    x1b, _ = m.pnp_run(a, bb, den, [0, 1], x0=x0)
    assert x1b[0].tolist() == x0.tolist() and x1b[1] == pytest.approx(x0 - ((x0 @ a.T - bb) @ a))


def test_pnp_run_error_history_starts_at_the_start_and_zero_is_a_checkpoint_without_one(mini):
    a, b = mini
    den = m.make_denoiser("identity", 3, [0.0])
    truth = np.array([1.0, 2.0, 1.0])
    x0 = np.array([[1.0, 0.0, 0.0]])
    _, errs = m.pnp_run(a, b, den, [1], truth=truth, x0=x0)
    assert errs[0, 0] == pytest.approx(np.linalg.norm(x0[0] - truth) / np.linalg.norm(truth))
    xs, _ = m.pnp_run(a, b, den, [0, 2], x0=x0)
    assert xs[0].tolist() == x0.tolist() and xs.shape == (2, 1, 3)


def test_pnp_run_with_momentum_starts_after_the_second_step(mini):
    a, b = mini
    den = m.make_denoiser("soft", 3, [0.01])
    plain, _ = m.pnp_run(a, b, den, [1, 2, 3])
    fast, _ = m.pnp_run(a, b, den, [1, 2, 3], accel=True)
    assert plain[0].tolist() == fast[0].tolist() and plain[1].tolist() == fast[1].tolist()       # der erste Impuls (t - 1) / t' ist null
    assert not np.allclose(plain[2], fast[2])


def test_fixed_point_residual_is_zero_at_a_fixed_point_of_the_identity_denoiser(mini):
    a, b = mini
    den = m.make_denoiser("identity", 3, [0.0])
    x = np.linalg.solve(a.T @ a, a.T @ b)                                  # A^T (A x - b) = 0: der Fixpunkt der Landweber-Iteration
    assert m.fixed_point_residual(a, b, den, x)[0] < 1e-12
    assert m.fixed_point_residual(a, b, den, np.zeros(3))[0] == pytest.approx(1.34)


def test_lipschitz_estimate_limits():
    center = np.linspace(0.0, 1.0, 48)
    assert m.lipschitz_estimate("identity", 48, 1.0, center, 1) == pytest.approx(1.0)
    assert m.lipschitz_estimate("soft", 48, 1e9, center, 1) == 0.0                    # D = 0
    assert m.lipschitz_estimate("soft", 48, 0.1, center, 1) <= 1.0 + 1e-12
    assert m.lipschitz_estimate("soft", 48, 0.1, center, 1) == m.lipschitz_estimate("soft", 48, 0.1, center, 1)


# ----------------------------------------------------------------------------- Vergleichslöser

def test_tikhonov_svd_is_the_hand_value_and_equals_the_gauss_fixed_point(mini):
    a, b = mini
    x = m.tikhonov_svd(a, b, [0.1, 1e-6])
    assert x.shape == (2, 1, 3) and x[0, 0] == pytest.approx([1.153465, 1.653465, 1.153465], abs=1e-6)
    assert x[1, 0] == pytest.approx(np.linalg.solve(a, b), abs=1e-4)
    assert m.tikhonov_svd(a, np.array([b, 2 * b]), [0.1])[0, 1] == pytest.approx(2 * x[0, 0])


def test_gauss_closed_form_and_posterior_mean_hand_values(mini):
    a, b = mini
    p = np.eye(3)
    assert m.gauss_closed_form(a, b, 0.1, p, 1.0) == pytest.approx([1.153465, 1.653465, 1.153465], abs=1e-6)
    assert m.posterior_mean(a, b, 0.1, p) == pytest.approx([1.153465, 1.653465, 1.153465], abs=1e-6)
    assert m.posterior_mean(np.array([[1.0]]), np.array([1.0]), 0.1, np.array([[1.0]])) == pytest.approx([1.0 / 1.01])
    assert m.posterior_mean(np.array([[2.0]]), np.array([1.0]), 1.0, np.array([[0.5]])) == pytest.approx([0.5 * 2.0 / (4.0 * 0.5 + 1.0)])


def test_lasso_hand_cases():
    assert m.lasso(np.eye(3), np.array([3.0, -0.5, 1.0]), 1.0) == pytest.approx([2.0, 0.0, 0.0])
    assert m.lasso(2.0 * np.eye(2), np.array([3.0, -3.0]), 1.0) == pytest.approx([1.25, -1.25])
    assert m.lasso(np.eye(2), np.array([1.0, 0.0]), 1.0) == pytest.approx([0.0, 0.0])


def test_objectives_hand_values(mini):
    a, b = mini
    x = np.array([1.0, 2.0, 1.0])
    r = a @ x - b
    assert m.objective_l1(a, b, x, 0.5) == pytest.approx(0.5 * float(r @ r) + 0.5 * 4.0)
    assert m.objective_l1(a, b, np.zeros(3), 0.5) == pytest.approx(0.5 * float(b @ b))


def test_objective_haar_penalizes_only_the_details(monkeypatch):
    monkeypatch.setattr(C, "HAAR_LEVELS", 2)
    a, b = np.eye(4), np.zeros(4)
    x = np.array([1.0, 2.0, 3.0, 4.0])                                           # Koeffizienten (5, -2, -0,7071, -0,7071): bestraft werden 2 + 0,7071 + 0,7071
    assert m.objective_haar(a, b, x, 0.5) == pytest.approx(0.5 * 30.0 + 0.5 * (2.0 + 2.0 * math.sqrt(0.5)))
    assert m.objective_haar(a, b, np.ones(4), 0.5) == pytest.approx(0.5 * 4.0)           # konstant: nur der Näherungskoeffizient


def test_wellposed_matrix_hand_values():
    a = m.wellposed_matrix(3)
    assert a == pytest.approx(np.array([[0.6, 0.2, 0.0], [0.2, 0.6, 0.2], [0.0, 0.2, 0.6]]))
    assert np.linalg.cond(m.wellposed_matrix(48)) < 6.0


def test_op_is_cached_and_read_only():
    a = m.op()
    assert a is m.op(48, 0.03 + 1e-9) and not a.flags.writeable and a.shape == (48, 48)
