"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Reglergrenzen und die dokumentierten Ausgänge."""

import dataclasses

import numpy as np
import pytest

import pnp_constants as C
import pnp_evaluation as ev
import pnp_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_on_the_choice_lists():
    for name, p in C.PRESETS.items():
        assert p["world"] in C.WORLDS and p["noise"] in C.NOISE_CHOICES and p["denoiser"] in C.DENOISERS and p["factor"] in C.FACTOR_CHOICES, name
        assert p["steps"] in C.STEPS_CHOICES and isinstance(p["accel"], bool) and 0 <= p["seed"] <= C.SEED_MAX, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standard"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)


def test_every_default_is_a_valid_widget_value():
    for key, spec in P.SETTING_SPECS.items():
        if spec.choices is not None:
            assert spec.default in spec.choices, key
        else:
            assert spec.lo <= spec.default <= spec.hi, key


def test_settings_fields_follow_the_preset_keys():
    assert [f.name for f in dataclasses.fields(ev.Settings)] == list(P.PRESET_KEYS)


def test_to_bool_accepts_only_true_and_false():
    assert P._to_bool("True") is True and P._to_bool("False") is False
    with pytest.raises(KeyError):
        P._to_bool("1")


def test_the_presets_show_the_documented_outcomes():
    res = {name: ev.analyse(ev.Settings(**p)) for name, p in C.PRESETS.items()}
    std, wrong, right, plain_haar, waves, fix, fast = (res[k] for k in C.PRESETS)
    assert ev.verdict_instance(std)[0] == "success" and std.err_pnp == pytest.approx(0.0225, abs=0.002) and std.err_tik == pytest.approx(0.1771, abs=0.002)
    assert ev.verdict_instance(wrong)[0] == "warning" and wrong.err_pnp == pytest.approx(0.650, abs=0.005) and wrong.err_tik == pytest.approx(0.534, abs=0.005) and wrong.err_l1 == pytest.approx(0.0144, abs=0.002) and wrong.at_edge
    assert ev.verdict_instance(right)[0] == "success" and right.err_pnp == pytest.approx(0.0873, abs=0.003) and right.err_l1 < right.err_pnp
    assert ev.verdict_instance(plain_haar)[0] == "info" and plain_haar.err_pnp == pytest.approx(0.1672, abs=0.003) and plain_haar.err_pnp < plain_haar.err_tik
    assert ev.verdict_instance(waves)[0] == "success" and waves.err_pnp == pytest.approx(0.1034, abs=0.003) and waves.err_tik == pytest.approx(0.1702, abs=0.003) and waves.lip == pytest.approx(1.101, abs=0.01)
    k = int(np.argmin(fix.err_hist))
    assert ev.verdict_instance(fix)[0] == "success" and fix.err_pnp == pytest.approx(0.0739, abs=0.003) and 600 <= k <= 800 and fix.err_hist[k] == pytest.approx(0.0525, abs=0.003)
    k = int(np.argmin(fast.err_hist))
    assert ev.verdict_instance(fast)[0] == "warning" and fast.err_pnp == pytest.approx(0.1288, abs=0.004) and 40 <= k <= 80 and fast.err_pnp > fast.err_tik


def test_experiment_constants_are_consistent():
    assert C.N == 48 and C.N % 2 ** C.HAAR_LEVELS == 0 and C.HAAR_SHIFTS == 2 ** C.HAAR_LEVELS and C.GRID_SIZE == 13 == len(C.LAMBDA_GRID) == len(C.NLM_GRID) == len(C.GAUSS_GRID)
    assert C.EXP_REPS >= 6 and C.NLM_REPS >= 4 and set(C.WORLD_NOISES) <= set(C.NOISE_CHOICES) and C.WEAK_NOISE in C.NOISE_CHOICES and C.STEP_NOISE in C.NOISE_CHOICES and C.CONV_NOISE in C.NOISE_CHOICES
    assert set(C.STEP_WORLDS) <= set(C.WORLDS) and set(C.WEAK_STRENGTH) == set(C.CONV_STRENGTHS) == set(C.DENOISERS) and C.DEFAULT_STEPS in C.STEPS_CHOICES and C.DEFAULT_FACTOR in C.FACTOR_CHOICES
    assert C.GAIN == 0.8 and C.TIE == 1.0 and C.EARLY == 0.95 and set(C.SHOW_RANGE) == set(C.WORLDS) and set(C.WORLD_LABELS) == set(C.WORLD_SHORT) == set(C.WORLDS)
    assert set(C.DENOISER_LABELS) == set(C.DENOISER_SHORT) == set(C.DENOISER_UNIT) == set(C.DENOISERS) and set(C.FACTOR_LABELS) == set(C.FACTOR_CHOICES) and C.PROX_CHECKS[-1] > C.PROX_CHECKS[0]
