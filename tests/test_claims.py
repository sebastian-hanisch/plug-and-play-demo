"""Aussagen aus README, Verdikten und Hilfetexten, einzeln an den festen Strömen gemessen (Seeds ab SEED_BASE). Ändert sich eine Zahl, muss der Text mit.
Linux-CI: in der letzten Stelle nichts festlegen; Mediane mit enger Toleranz (stromabhängige Zahlen auf 0,002), Schranken statt Einzelwerte."""

import pytest

import pnp_constants as C
import pnp_evaluation as ev


@pytest.fixture(scope="module")
def prox():
    return ev.prox_experiment()


@pytest.fixture(scope="module")
def world():
    return ev.world_experiment()


@pytest.fixture(scope="module")
def steps():
    return ev.steps_experiment()


@pytest.fixture(scope="module")
def conv():
    return ev.convergence_experiment()


def cell(rows, **kw):
    return ev._by(rows, **kw)


# ----------------------------------------------------------------------------- Experiment 1

def test_prox_is_ista_on_a_wellposed_matrix_and_in_the_objective_on_the_ill_posed_one(prox):
    assert all(r["dist"] < 1e-9 for r in prox["wp"]) and prox["wp"][0]["cond"] == pytest.approx(4.975, abs=0.01)
    gaps = [r["gap"] for r in prox["l1"]]
    assert gaps == pytest.approx([1.54e-4, 1.94e-5, 6.89e-6], rel=0.05) and gaps[0] > gaps[1] > gaps[2] > 0.0
    assert [r["dist"] for r in prox["l1"]] == pytest.approx([0.630, 0.647, 0.640], abs=0.01)


def test_gauss_mmse_fixed_point_is_the_posterior_mean_at_the_right_strength(prox):
    g = {r["factor"]: r for r in prox["gauss"]}
    assert g[1.0]["dist_pm"] < 1e-5 and g[1.0]["dist_closed"] < 1e-5 and g[1.0]["err_pnp"] == pytest.approx(0.1307, abs=0.002) and g[1.0]["err_pm"] == pytest.approx(0.1307, abs=0.002)
    assert g[3.0]["dist_closed"] < 1e-9 and g[3.0]["dist_pm"] == pytest.approx(0.0647, abs=0.003) and g[3.0]["err_pnp"] == pytest.approx(0.1039, abs=0.002)
    assert g[0.3]["dist_closed"] == pytest.approx(0.0595, abs=0.005) and g[0.3]["dist_pm"] == pytest.approx(0.152, abs=0.005) and g[0.3]["err_pnp"] == pytest.approx(0.2593, abs=0.003)


def test_haar_objective_is_monotone(prox):
    assert all(r["monotone"] and r["resid"] < 1e-8 for r in prox["haar"]) and [r["f_first"] for r in prox["haar"]] == pytest.approx([0.1323, 0.1887], abs=0.002)
    assert [r["f_last"] for r in prox["haar"]] == pytest.approx([0.0073069, 0.0606996], abs=2e-6) and [r["f_first"] for r in prox["haar"]] == pytest.approx([0.132321, 0.188672], abs=2e-6)
    text = ev.verdict_prox(prox)[1]
    assert "(Kondition 5,0) liegt Plug-and-Play nach 500 Schritten unter 10⁻⁹ vom exakten Löser" in text and "(1,5·10⁻⁴ / 1,9·10⁻⁵ / 6,9·10⁻⁶ nach 100 / 1000 / 10000 Schritten)" in text
    assert "Abstand im x bleibt 0,63 bis 0,65" in text and "bei s = √t σ 2,9·10⁻⁶ vom Posterior-Mittel (Fehler 13,1 %)" in text
    assert "bei s = 3 √t σ ist er die geschlossene Form (unter 10⁻⁹), aber nicht das Posterior-Mittel (Fehler 10,4 % statt 13,1 %)" in text and "fällt in 60 Schritten monoton" in text


# ----------------------------------------------------------------------------- Experiment 2

def test_the_denoiser_decides_about_the_world(world):
    rows = world["rows"]
    assert len(rows) == 8
    sp = cell(rows, world="spikes", noise=1e-2)
    assert sp["errs"]["soft"] == pytest.approx(0.0866, abs=0.003) and sp["tik"] == pytest.approx(0.5431, abs=0.003) and sp["l1"] == pytest.approx(0.0085, abs=0.002)
    assert all(sp["errs"][d] > 0.6 for d in ("haar", "haar_avg", "nlm", "gauss")) and all(sp["errs"][d] > sp["tik"] for d in ("haar", "haar_avg", "nlm", "gauss"))
    bl = cell(rows, world="blocks", noise=1e-2)
    assert bl["errs"]["haar_avg"] == pytest.approx(0.0216, abs=0.002) and bl["errs"]["haar"] == pytest.approx(0.1724, abs=0.003) and bl["tik"] == pytest.approx(0.1894, abs=0.003)
    assert bl["errs"]["haar"] > 5 * bl["errs"]["haar_avg"] and min(bl["errs"], key=bl["errs"].get) == "haar_avg"
    wv = cell(rows, world="waves", noise=1e-1)
    assert wv["errs"]["nlm"] == pytest.approx(0.0912, abs=0.004) and wv["errs"]["gauss"] == pytest.approx(0.1060, abs=0.003) and wv["tik"] == pytest.approx(0.1619, abs=0.003)
    assert min(wv["errs"], key=wv["errs"].get) == "nlm"
    wv1 = cell(rows, world="waves", noise=1e-2)
    assert wv1["errs"]["nlm"] == pytest.approx(0.0317, abs=0.003) and wv1["tik"] == pytest.approx(0.0459, abs=0.003) and wv1["errs"]["nlm"] < 0.75 * wv1["tik"]


def test_phantom_numbers_and_the_exact_l1_comparison(world):
    ph = cell(world["rows"], world="phantom", noise=1e-2)
    assert ph["errs"]["soft"] == pytest.approx(0.0614, abs=0.003) and ph["errs"]["haar_avg"] == pytest.approx(0.0780, abs=0.003) and ph["tik"] == pytest.approx(0.1098, abs=0.003) and ph["l1"] == pytest.approx(0.523, abs=0.01)
    assert ph["errs"]["soft"] < ph["l1"] / 5                                               # derselbe Prox, aber 300 Schritte statt Fixpunkt
    ph10 = cell(world["rows"], world="phantom", noise=1e-1)
    assert ph10["errs"]["gauss"] == pytest.approx(0.1726, abs=0.003) and ph10["errs"]["soft"] == pytest.approx(0.3087, abs=0.005) and ph10["tik"] == pytest.approx(0.2102, abs=0.003)
    sp10 = cell(world["rows"], world="spikes", noise=1e-1)
    assert sp10["errs"]["soft"] == pytest.approx(0.1174, abs=0.004) and sp10["l1"] == pytest.approx(0.0855, abs=0.003)
    assert all(r["strengths"][d] > 0.0 for r in world["rows"] for d in C.DENOISERS)
    text = ev.verdict_world(world)[1]
    assert "erreicht nur die Weichschwelle 8,7 % (Tikhonov 54,3 %)" in text and "liegen bei 61,3 % bis 64,9 %" in text and "Haar gemittelt mit 2,2 % am besten, Haar ohne Mittelung hat 17,2 % (Tikhonov 18,9 %)" in text
    assert "nicht-lokale Mittel 9,1 %, Gauß-MMSE 10,6 %, Tikhonov 16,2 %" in text


# ----------------------------------------------------------------------------- Experiment 3

def test_weak_strength_gives_the_same_curve_for_every_denoiser(steps):
    weak = steps["weak"]
    assert weak["soft"][:3] == pytest.approx([0.1344, 0.1243, 0.1123], abs=0.0005)
    for i in range(3):
        assert max(weak[d][i] for d in C.DENOISERS) - min(weak[d][i] for d in C.DENOISERS) < 5e-4
    assert weak["soft"][-1] == pytest.approx(0.0640, abs=0.002) and weak["nlm"][-1] == pytest.approx(0.0774, abs=0.002) and weak["haar_avg"][-1] == pytest.approx(0.0676, abs=0.002)
    assert all(weak[d][0] > weak[d][1] > weak[d][2] > weak[d][3] > weak[d][4] > weak[d][5] for d in C.DENOISERS)


def test_the_denoiser_does_not_replace_early_stopping(steps):
    soft_ph = cell(steps["conv"], world="phantom", den="soft")["errs"]
    soft_wv = cell(steps["conv"], world="waves", den="soft")["errs"]
    assert soft_ph == pytest.approx([0.0812, 0.0614, 0.0931], abs=0.002) and soft_wv == pytest.approx([0.0424, 0.0445, 0.1048], abs=0.002)
    assert soft_ph[2] > 1.4 * soft_ph[1] and soft_wv[2] > 2.0 * soft_wv[1]
    assert cell(steps["conv"], world="waves", den="gauss")["errs"] == pytest.approx([0.0436, 0.0437, 0.0437], abs=0.0005)
    assert cell(steps["conv"], world="waves", den="nlm")["errs"] == pytest.approx([0.0371, 0.0317, 0.0923], abs=0.002)
    assert cell(steps["conv"], world="blocks", den="haar_avg")["errs"] == pytest.approx([0.0830, 0.0216, 0.0209], abs=0.002)
    assert cell(steps["conv"], world="blocks", den="haar")["errs"] == pytest.approx([0.1956, 0.1724, 0.1224], abs=0.003) and cell(steps["conv"], world="phantom", den="haar_avg")["errs"] == pytest.approx([0.1044, 0.0780, 0.0862], abs=0.003)
    assert len(steps["conv"]) == 13 and ("blocks", "nlm") not in {(r["world"], r["den"]) for r in steps["conv"]}
    assert all(len(r["errs"]) == len(r["strengths"]) == 3 for r in steps["conv"]) and cell(steps["conv"], world="phantom", den="soft")["strengths"] == pytest.approx([0.010083, 0.0031885, 0.00010083], rel=1e-4)


def test_momentum_makes_every_denoiser_worse(steps):
    acc = {r["den"]: r for r in steps["accel"]}
    assert acc["soft"]["plain"] == pytest.approx(0.0614, abs=0.003) and acc["soft"]["accel"] == pytest.approx(0.1854, abs=0.005)
    assert acc["nlm"]["accel"] == pytest.approx(0.252, abs=0.01) and acc["gauss"]["accel"] == pytest.approx(0.2355, abs=0.01)
    assert all(r["accel"] > 1.3 * r["plain"] for r in steps["accel"])
    text = ev.verdict_steps(steps)[1]
    assert "(13,4 % nach 10, 11,2 % nach 100 Schritten, Spannweite 0,03 %)" in text and "beim Phantom von 6,1 % (300 Schritte) auf 9,3 % (3000 Schritte) schlechter, bei den Wellen von 4,5 % auf 10,5 %" in text
    assert "(4,4 % / 4,4 % / 4,4 % nach 100 / 300 / 3000 Schritten)" in text and "verschlechtert die Weichschwelle von 6,1 % auf 18,5 %" in text


# ----------------------------------------------------------------------------- Experiment 4

def test_lipschitz_and_convergence_numbers(conv):
    rows = conv["rows"]
    assert len(rows) == 8
    prox_rows = [r for r in rows if r["den"] != "nlm"]
    assert all(r["lip"] <= 1.0 + 1e-9 for r in prox_rows)
    nlm = [cell(rows, den="nlm", param=h) for h in (0.01, 0.03, 0.1)]
    assert [r["lip"] for r in nlm] == pytest.approx([1.000, 1.0265, 1.0761], abs=0.003) and nlm[2]["lip"] > 1.05 and nlm[1]["lip"] > 1.0 + 1e-3
    assert max(r["resid"] for r in rows) < 4e-4 and cell(rows, den="nlm", param=0.1)["resid"] == pytest.approx(3.4e-4, abs=0.4e-4)
    assert cell(rows, den="haar")["shift_dev"] == pytest.approx(1.6713e-3, abs=2e-6) and cell(rows, den="nlm", param=0.1)["shift_dev"] == pytest.approx(0.013006, abs=2e-5)
    assert all(r["shift_dev"] < 1e-9 for r in rows if r["den"] == "haar_avg") and cell(rows, den="gauss")["shift_dev"] == pytest.approx(2.199e-4, abs=2e-6)
    assert cell(rows, den="soft")["err"] == pytest.approx(0.0887, abs=0.002) and cell(rows, den="gauss")["err"] == pytest.approx(0.1038, abs=0.002) and cell(rows, den="haar")["err"] == pytest.approx(0.2226, abs=0.003)
    assert all(r["limit_dist"] < 5e-3 + 1e-4 for r in rows)
    text = ev.verdict_convergence(conv)[1]
    assert "höchstens bei 1,000 (in der Stichprobe nichtexpansiv)" in text and "h = 0,01 / 0,03 / 0,10 bei 1,000 / 1,027 / 1,076: über 1" in text and "höchstens 3,4·10⁻⁴" in text
    assert "Haar ohne Mittelung verfehlt die Verschiebungsgleichheit um 1,7·10⁻³, Haar gemittelt um unter 10⁻⁹" in text
