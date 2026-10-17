"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Preset-Button, fährt jeden
Regler an seine Grenzen und prüft, dass kein Python-Fehler auftritt (insbesondere StreamlitDuplicateElementId bei st.plotly_chart ohne eindeutiges key=)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 600
AUTOPLAY_MARKER = "▶"


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_option_of_every_select_slider_does_not_raise():
    import pnp_constants as C

    for key, options in (("noise_select", C.NOISE_CHOICES), ("factor_select", C.FACTOR_CHOICES), ("steps_select", C.STEPS_CHOICES)):
        for value in options:
            at = _fresh_app()
            at.select_slider(key=key).set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (key, value, [str(e) for e in at.exception])


def test_every_world_and_denoiser_does_not_raise():
    import pnp_constants as C

    for world in C.WORLDS:
        for den in C.DENOISERS:
            at = _fresh_app()
            at.selectbox(key="world_select").set_value(world)
            at.selectbox(key="denoiser_select").set_value(den)
            at.run(timeout=TIMEOUT)
            assert not at.exception, (world, den, [str(e) for e in at.exception])


def test_the_corners_of_the_grid_do_not_raise():
    at = _fresh_app()
    at.selectbox(key="world_select").set_value("spikes")
    at.selectbox(key="denoiser_select").set_value("nlm")
    at.select_slider(key="noise_select").set_value(1e-1)
    at.select_slider(key="steps_select").set_value(3000)
    at.select_slider(key="factor_select").set_value(10.0)
    at.toggle(key="accel_toggle").set_value(True)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    at = _fresh_app()
    at.selectbox(key="world_select").set_value("waves")
    at.selectbox(key="denoiser_select").set_value("gauss")
    at.select_slider(key="noise_select").set_value(1e-3)
    at.select_slider(key="steps_select").set_value(30)
    at.select_slider(key="factor_select").set_value(0.1)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


def test_every_preset_is_applied_to_the_widgets():
    import pnp_constants as C

    labels = {b.label: i for i, b in enumerate(_fresh_app().button)}
    for name, preset in C.PRESETS.items():
        at = _fresh_app()
        at.button[labels[name]].click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        assert at.selectbox(key="world_select").value == preset["world"]
        assert at.select_slider(key="noise_select").value == preset["noise"]
        assert at.selectbox(key="denoiser_select").value == preset["denoiser"]
        assert at.select_slider(key="factor_select").value == preset["factor"]
        assert at.select_slider(key="steps_select").value == preset["steps"]
        assert at.toggle(key="accel_toggle").value == preset["accel"]
        assert at.number_input(key="seed_input").value == preset["seed"]


def test_the_standard_instance_shows_the_success_verdict_and_four_experiments():
    at = _fresh_app()
    assert any("Der Denoiser hilft" in s.value for s in at.success)
    headers = [s.value for s in at.subheader]
    assert sum("🔬" in h for h in headers) == 4


def test_the_wrong_denoiser_preset_shows_a_warning():
    labels = {b.label: i for i, b in enumerate(_fresh_app().button)}
    at = _fresh_app()
    at.button[labels["Falscher Denoiser"]].click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any("passt nicht zur Welt" in w.value for w in at.warning)


def test_pdf_is_created_only_after_the_click():
    at = _fresh_app()
    assert "pdf_made" not in at.session_state
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    settings, data = at.session_state["pdf_made"]
    assert data[:5] == b"%PDF-" and len(data) > 1000


def test_pdf_of_the_other_presets_is_created():
    labels = {b.label: i for i, b in enumerate(_fresh_app().button)}
    for name in ("Passender Denoiser", "Wellen, viel Rauschen", "Mit Beschleunigung"):
        at = _fresh_app()
        at.button[labels[name]].click().run(timeout=TIMEOUT)
        at.button(key="pdf_make").click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        assert at.session_state["pdf_made"][1][:5] == b"%PDF-"


def test_permalink_values_are_clipped_and_unknown_choices_ignored():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["world"] = "kreis"
    at.query_params["r"] = "0.5"
    at.query_params["den"] = "median"
    at.query_params["fac"] = "2"
    at.query_params["steps"] = "7"
    at.query_params["accel"] = "ja"
    at.query_params["seed"] = "99999999"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.selectbox(key="world_select").value == "blocks"
    assert at.select_slider(key="noise_select").value == 1e-2
    assert at.selectbox(key="denoiser_select").value == "haar_avg"
    assert at.select_slider(key="factor_select").value == 1.0
    assert at.select_slider(key="steps_select").value == 300
    assert at.toggle(key="accel_toggle").value is False
    assert at.number_input(key="seed_input").value == 999999


def test_valid_permalink_values_are_taken_over():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["world"] = "waves"
    at.query_params["den"] = "nlm"
    at.query_params["accel"] = "True"
    at.query_params["steps"] = "100"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.selectbox(key="world_select").value == "waves" and at.selectbox(key="denoiser_select").value == "nlm"
    assert at.toggle(key="accel_toggle").value is True and at.select_slider(key="steps_select").value == 100


def test_the_address_bar_mirrors_the_settings():
    at = _fresh_app()
    at.selectbox(key="denoiser_select").set_value("soft").run(timeout=TIMEOUT)
    assert at.query_params["den"] == ["soft"] or at.query_params["den"] == "soft"
