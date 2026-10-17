"""Formatierung, Welt, Instanz, Analyse einer Einstellung und die Verzweigungen des Urteils (jede Schwelle auf der Kante)."""

import dataclasses
import math

import numpy as np
import pytest

import pnp_constants as C
import pnp_evaluation as ev
import pnp_model as m
from inverse_generator import SplitMix64


def S(**kw) -> ev.Settings:
    return ev.Settings(**kw)


@pytest.fixture(scope="module")
def std():
    return ev.analyse(S())


# ----------------------------------------------------------------------------- Formatierung

def test_de_uses_comma_and_typographic_minus():
    assert ev.de(-1.5, 1) == "−1,5" and ev.de(0.04, 3) == "0,040" and ev.de(1234.5, 0) in ("1234", "1235")


def test_sci_formats_powers_of_ten_and_rounds_up_the_mantissa():
    assert ev.sci(2.5e9) == "2,5·10⁹" and ev.sci(0.00039) == "3,9·10⁻⁴" and ev.sci(9.96e-3) == "1,0·10⁻²" and ev.sci(0.0) == "0,0" and ev.sci(math.inf) == "unendlich"
    assert ev.sci(-2.5e-3) == "−2,5·10⁻³" and ev.sci(1.0) == "1,0·10⁰"


def test_err_pct_has_four_ranges_and_a_factor_form():
    assert ev.err_pct(0.5) == "50,0 %" and ev.err_pct(1.19) == "119 %" and ev.err_pct(0.0123) == "1,2 %" and ev.err_pct(0.00123) == "0,12 %" and ev.err_pct(1e-5) == "1,0·10⁻³ %"
    assert ev.err_pct(12.0) == "1,2·10¹-fach" and ev.err_pct(math.inf) == "unendlich" and ev.err_pct(9.99) == "999 %" and ev.err_pct(0.9951) == "100 %"


def test_noise_label():
    assert ev.noise_label(0.01) == "1 %" and ev.noise_label(0.001) == "0,1 %" and ev.noise_label(0.1) == "10 %" and ev.noise_label(0.0) == "0 %"


def test_rel_err_is_relative_to_the_truth_norm_per_row():
    assert ev.rel_err(np.array([3.0, 0.0]), np.array([0.0, 4.0])) == pytest.approx(5.0 / 4.0)
    assert ev.rel_err(np.array([[1.0, 1.0], [2.0, 2.0]]), np.array([1.0, 1.0])) == pytest.approx([0.0, 1.0])


def test_small_distances_are_shown_as_a_bound():
    assert ev._small(1e-12) == "unter 10⁻⁹" and ev._small(1e-9) == "1,0·10⁻⁹" and ev._small(0.5) == "5,0·10⁻¹"


# ----------------------------------------------------------------------------- Instanz und Stärkegitter

def test_instance_noise_scale_and_determinism():
    x, b, sigma, clean = ev.instance("phantom", 1e-2, 7)
    assert clean == pytest.approx(m.op() @ x) and sigma == pytest.approx(0.01 * np.sqrt(np.mean(clean ** 2)))
    assert b - clean == pytest.approx(SplitMix64(7).normal(48) * sigma)
    assert ev.instance("phantom", 1e-2, 7) is ev.instance("phantom", 1e-2, 7)
    assert not np.allclose(b, ev.instance("phantom", 1e-2, 8)[1]) and ev.instance("phantom", 1e-1, 7)[2] == pytest.approx(10.0 * sigma)
    assert ev.noise_sigma(np.array([3.0, 4.0]), 0.1) == pytest.approx(0.1 * math.sqrt(12.5))


def test_strength_grids_and_lambda():
    t = 0.5
    soft = ev.strength_grid("soft", t)
    assert len(soft) == 13 and soft == pytest.approx(t * np.array(C.LAMBDA_GRID)) and (np.diff(soft) > 0).all()
    assert ev.strength_grid("haar", t).tolist() == soft.tolist() and ev.strength_grid("haar_avg", t).tolist() == soft.tolist()
    assert ev.strength_grid("nlm", t).tolist() == list(C.NLM_GRID) and ev.strength_grid("gauss", t).tolist() == list(C.GAUSS_GRID)
    assert ev.lambda_of("soft", 0.1, 0.5) == pytest.approx(0.2) and ev.lambda_of("haar", 0.1, 0.5) == pytest.approx(0.2) and math.isnan(ev.lambda_of("nlm", 0.1, 0.5)) and math.isnan(ev.lambda_of("gauss", 0.1, 0.5))
    assert C.LAMBDA_GRID[0] == pytest.approx(1e-6) and C.LAMBDA_GRID[-1] == pytest.approx(1.0) and C.NLM_GRID[0] == pytest.approx(1e-3) and C.NLM_GRID[-1] == pytest.approx(10 ** 0.5)
    assert C.GAUSS_GRID[0] == pytest.approx(1e-5) and C.GAUSS_GRID[-1] == pytest.approx(1.0)


def test_experiment_grid_setup_uses_fewer_streams_for_nlm():
    g, reps = ev._den_setup("nlm", 0.5)
    assert len(g) == 13 and reps == C.NLM_REPS and ev._den_setup("soft", 0.5)[1] == C.EXP_REPS and ev._den_setup("soft", 0.5)[0].tolist() == ev.strength_grid("soft", 0.5).tolist()


def test_default_settings_equal_the_standard_preset():
    assert ev.Settings(**C.PRESETS["Standard"]) == ev.Settings()


# ----------------------------------------------------------------------------- Analyse

def test_analysis_of_the_standard_instance(std):
    a = std
    x, b, sigma, _ = ev.instance("blocks", 1e-2, 7)
    assert a.x is x and a.b is b and a.sigma == sigma and a.s == pytest.approx((np.arange(48) + 0.5) / 48)
    assert a.t == pytest.approx(m.step_size(m.op())) and len(a.grid) == len(a.grid_errs) == 13 and a.best_index == int(np.argmin(a.grid_errs)) == 8
    assert a.best_strength == a.grid[a.best_index] and a.strength == a.best_strength and a.grid == pytest.approx(ev.strength_grid("haar_avg", a.t))
    assert len(a.err_hist) == 301 and a.err_hist[0] == pytest.approx(1.0) and a.err_hist[-1] == pytest.approx(a.err_pnp) and a.err_pnp == pytest.approx(float(ev.rel_err(a.x_pnp, x)))
    den = m.make_denoiser("haar_avg", 48, [a.strength])
    assert a.x_pnp == pytest.approx(m.pnp_run(m.op(), b[None, :], den, [300])[0][0, 0]) and a.grid_errs[a.best_index] == pytest.approx(a.err_pnp)
    assert a.step_in == pytest.approx(a.x_pnp - a.t * (m.op().T @ (m.op() @ a.x_pnp - b))) and a.step_out == pytest.approx(den(a.step_in[None, :])[0])
    assert a.resid == pytest.approx(float(m.fixed_point_residual(m.op(), b, den, a.x_pnp)[0])) and a.lip == pytest.approx(m.lipschitz_estimate("haar_avg", 48, a.strength, x, C.SEED_BASE + 77))
    assert not a.at_edge and a.tik_lam in C.TIK_GRID and a.l1_lam in C.L1_GRID


def test_baselines_are_the_best_of_their_grids(std):
    a = std
    tik = m.tikhonov_svd(m.op(), a.b, C.TIK_GRID)[:, 0, :]
    assert a.err_tik == pytest.approx(float(ev.rel_err(tik, a.x).min())) and a.x_tik == pytest.approx(tik[int(ev.rel_err(tik, a.x).argmin())])
    l1 = np.array([m.lasso(m.op(), a.b, lam) for lam in C.L1_GRID])
    assert a.err_l1 == pytest.approx(float(ev.rel_err(l1, a.x).min())) and a.x_l1 == pytest.approx(l1[int(ev.rel_err(l1, a.x).argmin())])


def test_the_factor_scales_the_chosen_strength_but_not_the_best_one(std):
    a = ev.analyse(S(factor=3.0))
    assert a.best_strength == std.best_strength and a.strength == pytest.approx(3.0 * std.best_strength) and a.grid_errs.tolist() == std.grid_errs.tolist()
    assert a.err_pnp != std.err_pnp


def test_steps_and_momentum_change_the_run():
    a = ev.analyse(S(steps=30))
    assert len(a.err_hist) == 31 and a.grid_errs.shape == (13,)
    plain = ev.analyse(S(world="phantom", denoiser="soft"))
    fast = ev.analyse(S(world="phantom", denoiser="soft", accel=True))
    assert fast.err_pnp > 1.5 * plain.err_pnp and plain.err_hist[0] == fast.err_hist[0]


def test_the_best_strength_is_at_the_edge_for_the_wrong_denoiser():
    a = ev.analyse(S(world="spikes"))
    assert a.at_edge and a.best_index == 0


def test_soft_threshold_analysis_has_a_lambda():
    a = ev.analyse(S(world="spikes", denoiser="soft"))
    assert a.best_index == 9 and ev.lambda_of("soft", a.strength, a.t) == pytest.approx(C.LAMBDA_GRID[9])


# ----------------------------------------------------------------------------- Verdikt: jede Verzweigung

def _v(std, **kw):
    base = dataclasses.replace(std, err_pnp=0.5, err_tik=1.0, err_l1=0.9, err_hist=np.linspace(1.0, 0.1, 301), resid=0.0, lip=1.0, at_edge=False)
    return ev.verdict_instance(dataclasses.replace(base, **kw))


def test_verdict_helps(std):
    kind, text = ev.verdict_instance(std)
    assert kind == "success" and text.startswith("Haar gemittelt, Blöcke, Rauschen 1 %, 300 Schritte: Fehler 2,2 % (Tikhonov 17,7 %, L1 exakt 56,0 %)") and "Der Denoiser hilft" in text
    assert "Lipschitz-Schätzung des Denoisers 1,000" in text and "Fixpunktrest 1,4·10⁻⁵" in text and "Stärke 1,0·10⁻² (Schwelle θ)" in text


def test_verdict_equal_and_wrong(std):
    kind, text = ev.verdict_instance(ev.analyse(S(**C.PRESETS["Haar ohne Mittelung"])))
    assert kind == "info" and "gleichwertig zu Tikhonov" in text
    kind, text = ev.verdict_instance(ev.analyse(S(**C.PRESETS["Falscher Denoiser"])))
    assert kind == "warning" and "passt nicht zur Welt" in text and "die kleinste des Gitters" in text


def test_gain_threshold_is_inclusive(monkeypatch, std):
    monkeypatch.setattr(C, "GAIN", 0.5)
    assert _v(std, err_pnp=0.5)[0] == "success" and _v(std, err_pnp=0.5000001)[0] == "info"


def test_tie_threshold_is_inclusive(std):
    assert _v(std, err_pnp=1.0)[0] == "info" and _v(std, err_pnp=1.0000001)[0] == "warning"


def test_edge_remark_only_at_the_best_strength(std):
    assert "die kleinste des Gitters" in _v(std, at_edge=True)[1]
    assert "die kleinste des Gitters" not in _v(std)[1]
    shifted = dataclasses.replace(std.settings, factor=3.0)
    assert "die kleinste des Gitters" not in _v(std, at_edge=True, settings=shifted)[1]


def test_early_stopping_remark_with_inclusive_threshold(monkeypatch, std):
    assert "nach 1 Schritten mit 50,0 % am kleinsten und steigt danach auf 60,0 %" in _v(std, err_hist=np.array([1.0, 0.5, 0.6]))[1]
    assert "der Abbruch regularisiert mit" not in _v(std, err_hist=np.array([1.0, 0.58, 0.6]))[1]
    monkeypatch.setattr(C, "EARLY", 0.5)
    assert "der Abbruch regularisiert mit" in _v(std, err_hist=np.array([1.0, 0.25, 0.5]))[1] and "der Abbruch regularisiert mit" not in _v(std, err_hist=np.array([1.0, 0.2500001, 0.5]))[1]


def test_exact_l1_remark_only_for_the_soft_threshold(std):
    soft = dataclasses.replace(std.settings, denoiser="soft")
    assert "Das exakte L1 ist besser (40,0 %)" in _v(std, settings=soft, err_pnp=0.5, err_l1=0.4)[1]
    assert "Das exakte L1 ist besser" not in _v(std, settings=soft, err_pnp=0.5, err_l1=0.5)[1]
    assert "Das exakte L1 ist besser" not in _v(std, err_pnp=0.5, err_l1=0.4)[1]


def test_fixed_point_remark_is_strict(monkeypatch, std):
    monkeypatch.setattr(C, "RES_OK", 0.5)
    assert "noch nicht am Fixpunkt" not in _v(std, resid=0.5)[1] and "noch nicht am Fixpunkt (Rest 5,0·10⁻¹)" in _v(std, resid=0.5000001)[1]


def test_lipschitz_remark_is_strict(monkeypatch, std):
    assert "Konvergenzgarantie" not in _v(std, lip=1.0)[1] and "Lipschitz-Schätzung liegt über 1 (1,100)" in _v(std, lip=1.1)[1]
    assert "Konvergenzgarantie" not in _v(std, lip=1.0 + 1e-6)[1]                 # auf drei Stellen noch 1,000: kein Hinweis
    monkeypatch.setattr(C, "LIP_ONE", 1.5)
    assert "Konvergenzgarantie" not in _v(std, lip=1.5)[1] and "Konvergenzgarantie" in _v(std, lip=1.5000001)[1]


def test_the_lipschitz_estimate_of_the_analysis_uses_the_documented_seed():
    a = ev.analyse(S(**C.PRESETS["Wellen, viel Rauschen"]))
    x = ev.truth_signal("waves")
    assert a.lip == pytest.approx(m.lipschitz_estimate("nlm", 48, a.strength, x, C.SEED_BASE + 77)) and a.lip != pytest.approx(m.lipschitz_estimate("nlm", 48, a.strength, x, C.SEED_BASE + 76), abs=1e-6)


def test_momentum_is_named_in_the_verdict(std):
    on = dataclasses.replace(std.settings, accel=True)
    assert ", mit FISTA-Impuls:" in _v(std, settings=on)[1] and ", mit FISTA-Impuls" not in _v(std)[1]


def test_by_finds_the_row_with_all_keys():
    rows = [{"a": 1, "b": 2}, {"a": 1, "b": 3}, {"a": 2, "b": 3}]
    assert ev._by(rows, a=1, b=3) is rows[1] and ev._by(rows, a=2) is rows[2]
