"""Plug-and-Play: Der Denoiser ist der Prior - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Dreizehntes Stück der Linie Inverse Probleme und Regularisierung. Auf der Welt aus Stück 1 bis 8 (lineare Entfaltung, Gauß-Unschärfe) wird der Prox von ISTA durch einen Denoiser ersetzt:
Ist er ein Prox, ist es Stück 6; ist er der Gauß-MMSE-Denoiser, ist es Stück 7; sonst ist es neu, und es gibt keine Zielfunktion mehr.
Kern: pnp_model.py, Generator: inverse_generator.py (wortgleich in allen Repos der Linie).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import pnp_constants as C
from pnp_evaluation import (
    Settings,
    analyse,
    convergence_experiment,
    de,
    err_pct,
    lambda_of,
    noise_label,
    prox_experiment,
    sci,
    steps_experiment,
    verdict_convergence,
    verdict_instance,
    verdict_prox,
    verdict_steps,
    verdict_world,
    world_experiment,
)
from pnp_pdf_export import build_pdf
from pnp_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from pnp_visualization import (
    build_gauss_chart,
    build_history_chart,
    build_lipschitz_chart,
    build_prox_chart,
    build_recon_chart,
    build_step_chart,
    build_stop_chart,
    build_strength_chart,
    build_table,
    build_weak_chart,
    build_world_chart,
    style_table,
)

st.set_page_config(page_title="Plug-and-Play – Sebastian Hanisch", layout="wide")

SPINNER = "Experiment rechnet über feste Zufallsströme (beim ersten Aufruf rund eine Minute) …"


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=SPINNER)
def _prox():
    return prox_experiment()


@st.cache_data(show_spinner=SPINNER)
def _world():
    return world_experiment()


@st.cache_data(show_spinner=SPINNER)
def _steps():
    return steps_experiment()


@st.cache_data(show_spinner=SPINNER)
def _convergence():
    return convergence_experiment()


def verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


def _small(v):
    return "< 10⁻⁹" if v < 1e-9 else sci(v)


st.title("🔌 Plug-and-Play: Der Denoiser ist der Prior")
st.markdown(
    """
**Der Denoiser ist der Prior, und das Verfahren dahinter ist ISTA.** Dies ist das dreizehnte Stück der Linie **Inverse Probleme und Regularisierung** und baut auf **Sparsity und Total Variation**,
**Bayes-Sicht: MAP** und **Blinde Entfaltung** auf: dieselbe Welt (lineare Entfaltung, Gauß-Unschärfe), aber im Iterationsschritt von ISTA steht statt der Weichschwelle ein **Denoiser**. Ist der
Denoiser ein Prox, ist das **exakt Stück 6**; mit dem Gauß-MMSE-Denoiser ist der Fixpunkt **exakt das Posterior-Mittel aus Stück 7**; neu wird es erst, wenn der Denoiser **kein Prox** ist. Gefragt wird,
**wann ein Denoiser hilft** (nur wenn er zur Welt passt), **wie es mit Schrittzahl und Stärke steht** (der Denoiser ersetzt den Abbruch nicht) und **was ohne Zielfunktion bleibt**
(keine Garantie). Die Welt ist simuliert; die Wahrheit ist bekannt und dient nur als Maßstab.

Weiter unten: vier Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("Wie Plug-and-Play funktioniert", expanded=True):
    st.markdown(
        r"""
1. **ISTA.** Ein Schritt ist $x \leftarrow \operatorname{prox}\bigl(x - t A^\top(Ax - b)\bigr)$ mit $t = 1/\|A\|^2$; bei der Weichschwelle ist der Prox die Strafe $\lambda\|x\|_1$ (Stück 6).
2. **Plug-and-Play.** Der Prox wird durch einen beliebigen Denoiser $D$ ersetzt: $x \leftarrow D\bigl(x - t A^\top(Ax - b)\bigr)$. Ist $D$ der Prox einer Funktion, gibt es eine Zielfunktion (MAP); sonst nicht.
3. **Gauß-MMSE.** Für einen Gauß-Prior ist $D(v) = P(P + s^2 I)^{-1} v$ linear; sein Fixpunkt ist bei $s^2 = t\sigma^2$ das Posterior-Mittel (Stück 7).
4. **Zwei Parameter.** Die **Stärke** des Denoisers und die **Schrittzahl**: bei schwacher Stärke regularisiert allein die Schrittzahl (Stück 5).
5. **Translationsinvarianz.** Ein Wavelet-Denoiser hängt an der Lage seines Gitters; über Verschiebungen gemittelt ist er es nicht mehr (aber auch kein Prox).
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Welt**")
    world = st.selectbox("Welche Wahrheit?", C.WORLDS, key="world_select", format_func=lambda v: C.WORLD_LABELS[v],
                         help="Der Denoiser hilft nur in der Welt, zu der er passt: Spitzen, Blöcke und Wellen haben verschiedene Struktur.")
    noise = st.select_slider("Rauschen (relativ)", options=C.NOISE_CHOICES, key="noise_select", format_func=noise_label,
                             help="Standardabweichung des weißen Rauschens relativ zum Effektivwert der exakten Daten.")
    st.markdown("**Verfahren**")
    denoiser = st.selectbox("Welcher Denoiser?", C.DENOISERS, key="denoiser_select", format_func=lambda v: C.DENOISER_LABELS[v],
                            help="Weichschwelle, Haar und Gauß-MMSE sind Prox-Abbildungen (Haar und Weichschwelle) beziehungsweise linear; Haar gemittelt und nicht-lokale Mittel sind kein Prox.")
    factor = st.select_slider("Stärke relativ zur besten Stärke", options=C.FACTOR_CHOICES, key="factor_select", format_func=lambda v: C.FACTOR_LABELS[v],
                              help="Die beste Stärke wird auf einem Gitter von 13 Werten mit der Wahrheit gesucht (Orakel, wie in Stück 4); hier wird sie um den Faktor verstimmt.")
    steps = st.select_slider("Schritte", options=C.STEPS_CHOICES, key="steps_select", help="Zahl der Plug-and-Play-Schritte; bei 3000 Schritten ist die Iteration nahe am Fixpunkt.")
    accel = st.toggle("FISTA-Impuls", key="accel_toggle", help="Beschleunigung nach Stück 6: erreicht den Fixpunkt schneller, und der Fixpunkt ist oft nicht die beste Lösung.")
    seed = st.number_input("Zufalls-Seed des Rauschens", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neu würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"world_select": world, "noise_select": float(noise), "denoiser_select": denoiser, "factor_select": float(factor), "steps_select": int(steps), "accel_toggle": bool(accel),
                   "seed_input": int(seed)})

settings = Settings(world, float(noise), denoiser, float(factor), int(steps), bool(accel), int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Fehler Plug-and-Play", err_pct(a.err_pnp), delta=f"Tikhonov {err_pct(a.err_tik)}", delta_color="off",
          help="Abstand zur Wahrheit relativ zur Signalnorm; darunter das beste Tikhonov-λ des Gitters (Orakel).")
c2.metric("L1 exakt", err_pct(a.err_l1), delta=f"λ = {sci(a.l1_lam)}", delta_color="off", help="Exakte L1-Lösung (Aktivmenge, Stück 6) beim besten λ des Gitters: der Grenzwert der Weichschwelle bei unendlich vielen Schritten.")
lam_eff = lambda_of(denoiser, a.strength, a.t)
c3.metric("Stärke", sci(a.strength), delta=f"{C.DENOISER_UNIT[denoiser]}" + (f", λ = {sci(lam_eff)}" if lam_eff == lam_eff else ""),
          delta_color="off", help="Die gewählte Stärke (Faktor mal beste Stärke); bei den Prox-Denoisern entspricht die Schwelle θ einer L1-Gewichtung λ = θ/t.")
c4.metric("Fixpunktrest", sci(a.resid), delta=f"Lipschitz-Schätzung {de(a.lip, 3)}", delta_color="off",
          help="max |x − D(x − t Aᵀ(A x − b))|: null am Fixpunkt. Darunter eine Stichproben-Untergrenze der Lipschitz-Konstante des Denoisers (über 1: nicht nichtexpansiv).")
st.caption(f"Beste Stärke des Gitters {sci(a.best_strength)}; Schrittweite t = {de(a.t, 3)}; Tikhonov λ = {sci(a.tik_lam)}; wahres Rauschen σ = {sci(a.sigma)}.")

tab_rec, tab_hist, tab_str, tab_step, tab_tab = st.tabs(["Rekonstruktion", "Fehler über die Schritte", "Fehler über die Stärke", "Ein Schritt", "Tabelle"])
with tab_rec:
    st.plotly_chart(build_recon_chart(a), width="stretch", key="recon_chart")
    st.caption("Grün die Wahrheit, blau Plug-and-Play, orange gestrichelt Tikhonov, rot gepunktet die exakte L1-Lösung (beide beim besten Parameter des Gitters). Die Achse ist fest.")
with tab_hist:
    st.plotly_chart(build_history_chart(a), width="stretch", key="history_chart")
    st.caption("Fehler über die Schritte (logarithmisch); das Kreuz markiert den kleinsten Fehler. Steigt die Kurve danach, regularisiert der Abbruch mit (Stück 5).")
with tab_str:
    st.plotly_chart(build_strength_chart(a), width="stretch", key="strength_chart")
    st.caption("Fehler nach den gewählten Schritten über die Stärke des Denoisers; das Kreuz ist die beste Stärke. Liegt es am linken Rand, trägt der Denoiser nichts bei.")
with tab_step:
    st.plotly_chart(build_step_chart(a), width="stretch", key="step_chart")
    st.caption("Der letzte Schritt: grau der Eingang des Denoisers (Gradientenschritt, noch verrauscht), blau sein Ausgang. Der Denoiser ist das, was den Eingang zur Lösung glättet.")
with tab_tab:
    st.dataframe(style_table(build_table(a)), hide_index=True, width="stretch", height=560)
    st.caption("Die ersten Punkte der Instanz.")

if st.button("📄 PDF der aktuellen Instanz erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="plug-and-play.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Hilft der Denoiser?")
verdict(verdict_instance(a))
st.caption("Das Urteil vergleicht den Fehler mit Tikhonov (hilft: unter 80 % davon) und nennt, ob die beste Stärke am Rand liegt (der Denoiser trägt nichts bei), ob der Fehler früher am kleinsten war (der Abbruch regularisiert "
           "mit), den Fixpunktrest und die Lipschitz-Schätzung.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Plug-and-Play ist ISTA")
res1 = _prox()
st.caption("Links: ISTA mit der Weichschwelle gegen den exakten L1-Löser (Stück 6) auf dem schlecht gestellten Operator, Phantom, Rauschen 1 %. Rechts: der Gauß-MMSE-Denoiser gegen die geschlossene Form und das "
           "Posterior-Mittel (Stück 7).")
ce1, ce2 = st.columns(2)
with ce1:
    st.plotly_chart(build_prox_chart(res1), width="stretch", key="prox_chart")
with ce2:
    st.plotly_chart(build_gauss_chart(res1), width="stretch", key="gauss_chart")
verdict(verdict_prox(res1))
st.dataframe([{"λ": sci(r["lam"], 1), "Kondition": de(r["cond"], 1), "max-Abstand zum exakten L1 nach 500 Schritten": _small(r["dist"])} for r in res1["wp"]], hide_index=True, width="stretch")
st.dataframe([{"Schritte": r["steps"], "Zielfunktionslücke": sci(r["gap"]), "Abstand im x": de(r["dist"], 2)} for r in res1["l1"]], hide_index=True, width="stretch")
st.dataframe([{"s / (√t σ)": de(r["factor"], 1), "Abstand zur geschlossenen Form": _small(r["dist_closed"]), "Abstand zum Posterior-Mittel": _small(r["dist_pm"]), "Fehler Plug-and-Play": err_pct(r["err_pnp"]),
               "Fehler Posterior-Mittel": err_pct(r["err_pm"])} for r in res1["gauss"]], hide_index=True, width="stretch")
st.dataframe([{"λ": sci(r["lam"], 1), "Zielfunktion fällt monoton (60 Schritte)": "ja" if r["monotone"] else "nein", "F nach 1 Schritt": sci(r["f_first"]), "F nach 60 Schritten": sci(r["f_last"]),
               "Fixpunktrest nach 2000": _small(r["resid"])} for r in res1["haar"]], hide_index=True, width="stretch")
st.caption("Die Weichschwelle ist ISTA: auf einer gut gestellten Matrix liegt sie nach wenigen hundert Schritten auf der exakten Lösung. Auf dem schlecht gestellten Operator fällt die Zielfunktion, aber das x bleibt weit "
           "weg, weil die Lösung in flachen Richtungen liegt (das hat Stück 6 schon gesehen). Der Gauß-MMSE-Denoiser ist Stück 7: bei s = √t σ der Fixpunkt das Posterior-Mittel, bei anderem s das Posterior-Mittel mit einem anderen λ.")

st.subheader("🔬 Experiment 2: Der Denoiser gehört zur Welt")
res2 = _world()
st.caption(f"Je Welt und Rauschstufe der beste Median-Fehler (gemeinsame Stärke über {C.EXP_REPS} Ströme, {C.NLM_REPS} bei den nicht-lokalen Mitteln) nach {C.WORLD_STEPS} Schritten, gegen Tikhonov und das exakte L1.")
st.plotly_chart(build_world_chart(res2), width="stretch", key="world_chart")
verdict(verdict_world(res2))
st.dataframe([{"Welt": C.WORLD_SHORT[r["world"]], "Rauschen": noise_label(r["noise"]), "Tikhonov": err_pct(r["tik"]), "L1 exakt": err_pct(r["l1"]), **{C.DENOISER_SHORT[d]: err_pct(r["errs"][d]) for d in C.DENOISERS}}
              for r in res2["rows"]], hide_index=True, width="stretch")
st.caption("Der Denoiser entscheidet über die Welt: bei Spitzen hilft nur die Weichschwelle, bei Blöcken die über Verschiebungen gemittelte Wavelet-Schwelle (ohne Mittelung kaum besser als Tikhonov), bei Wellen mit viel Rauschen "
           "die nicht-lokalen Mittel, die die Wiederholung nutzen. Ein „besserer“ Denoiser ist keiner, der in jeder Welt gewinnt.")

st.subheader("🔬 Experiment 3: Schrittzahl und Stärke")
res3 = _steps()
st.caption("Links: schwache Stärke, Phantom, Rauschen 0,1 %: alle Denoiser liegen übereinander (Landweber, Stück 5). Rechts: Wellen, Rauschen 1 %, die beste Stärke je Schrittzahl: bis zum Fixpunkt gerechnet wird es schlechter.")
cs1, cs2 = st.columns(2)
with cs1:
    st.plotly_chart(build_weak_chart(res3), width="stretch", key="weak_chart")
with cs2:
    st.plotly_chart(build_stop_chart(res3), width="stretch", key="stop_chart")
verdict(verdict_steps(res3))
st.dataframe([{"Welt": C.WORLD_SHORT[r["world"]], "Denoiser": C.DENOISER_SHORT[r["den"]], **{f"{k} Schritte": err_pct(e) for k, e in zip(C.STEP_CHECKS, r["errs"])}} for r in res3["conv"]],
             hide_index=True, width="stretch", height=420)
st.dataframe([{"Denoiser": C.DENOISER_SHORT[r["den"]], "300 Schritte": err_pct(r["plain"]), "300 Schritte mit FISTA-Impuls": err_pct(r["accel"])} for r in res3["accel"]], hide_index=True, width="stretch")
st.caption("Plug-and-Play hat zwei Parameter, Stärke und Schrittzahl, und sie sind gekoppelt: die beste Stärke liegt oft am Rand des Gitters (kein Denoiser, nur Abbruch). Der Gauß-MMSE-Denoiser ist über die Schrittzahl "
           "stabil, weil sein Fixpunkt ein ordentlicher Schätzer ist; die Weichschwelle und die Wavelets sind es nicht. Der FISTA-Impuls kommt schneller zum Fixpunkt und verschlechtert dadurch das Ergebnis.")

st.subheader("🔬 Experiment 4: Ohne Prox gibt es keine Garantie")
res4 = _convergence()
st.caption(f"Phantom, Rauschen 1 %, {C.CONV_STEPS} Schritte: Lipschitz-Schätzung der Denoiser (Stichprobe aus {C.LIP_TRIALS} Paaren), Fixpunktrest, Abstand der Grenzwerte aus zwei Starts, Verschiebungsabweichung.")
st.plotly_chart(build_lipschitz_chart(res4), width="stretch", key="lipschitz_chart")
verdict(verdict_convergence(res4))
st.dataframe([{"Denoiser": C.DENOISER_SHORT[r["den"]], "Stärke (λ beziehungsweise h, s)": de(r["param"], 3), "Lipschitz-Schätzung": de(r["lip"], 3), "Fixpunktrest": sci(r["resid"]),
               "Abstand der Grenzwerte (Start 0 und Daten)": sci(r["limit_dist"]), "Fehler": err_pct(r["err"]), "Verschiebungsabweichung": _small(r["shift_dev"]) if r["shift_dev"] < 1e-9 else sci(r["shift_dev"])}
              for r in res4["rows"]], hide_index=True, width="stretch")
st.caption("Die nicht-lokalen Mittel sind kein Prox und bei großer Stärke nicht nichtexpansiv; die Iteration konvergiert trotzdem, eine Garantie gibt es nicht (die Theorie verlangt nichtexpansive Denoiser). Der Abstand der "
           "Grenzwerte ist nicht null, weil auf dem schlecht gestellten Operator auch 3000 Schritte nicht ganz am Fixpunkt sind.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
sp2 = next(r for r in res2["rows"] if r["world"] == "spikes" and r["noise"] == 1e-2)
st.markdown(
    f"""
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Ein besserer Denoiser hilft überall** | Bei Spitzen (Rauschen 1 %) hilft nur die Weichschwelle ({err_pct(sp2['errs']['soft'])}); die anderen vier liegen bei {err_pct(min(sp2['errs'][d] for d in ('haar', 'haar_avg', 'nlm', 'gauss')))} bis {err_pct(max(sp2['errs'][d] for d in ('haar', 'haar_avg', 'nlm', 'gauss')))}, Tikhonov bei {err_pct(sp2['tik'])}. | Stück 16 (Verfahrensvergleich) |
| **Der Denoiser ersetzt den Abbruch** | Bis zum Fixpunkt gerechnet wird die Weichschwelle schlechter; Stärke und Schrittzahl sind zwei gekoppelte Parameter. | entrollte Netze (Stück 14) |
| **Es gibt eine Garantie** | Die nicht-lokalen Mittel sind kein Prox und bei großer Stärke nicht nichtexpansiv; gemessen wurde keine Divergenz, aber bewiesen ist nichts. | – |
| **Ein Operator, eine Dimension** | Toeplitz-Faltung mit w = 0,03, n = 48, ein Gitter von 13 Stärken, wenige Ströme. | – |
| **Klassische Denoiser** | Kein trainiertes Netz, kein Rauschniveau pro Schritt, kein PnP-ADMM; ein Score als Denoiser ist Stück 15. | Stück 15 |
| **Rauschen weiß, Daten b = A x + e** | Das Modell stimmt (kein Gitterfehler); ehrliche Daten sind nicht gemessen. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** $b = A x + e$, $e \sim \mathcal N(0, \sigma^2 I)$, $A$ die Gauß-Faltungsmatrix ($n = 48$, $w = 0{,}03$), Schrittweite $t = 1/\|A\|_2^2$.

**Plug-and-Play-ISTA.** $x_{k+1} = D\bigl(x_k - t A^\top(A x_k - b)\bigr)$. Ist $D = \operatorname{prox}_{tR}$, so ist das ISTA für $\tfrac12\|Ax - b\|^2 + R(x)$; für die Weichschwelle mit Schwelle $\theta$ ist
$R = (\theta/t)\|x\|_1$. Mit FISTA-Impuls: $y_{k+1} = x_{k+1} + \tfrac{\tau_k - 1}{\tau_{k+1}}(x_{k+1} - x_k)$.

**Haar-Wavelet.** Die orthonormale Haar-Zerlegung $W$ mit vier Ebenen; $D(v) = W^\top \operatorname{soft}_\theta(W v)$ auf den Detailkoeffizienten (die Näherungskoeffizienten bleiben): der Prox von $\theta\|W_d x\|_1$.
**Gemittelt:** $D_{\mathrm{avg}}(v) = \tfrac1{16}\sum_{k=0}^{15} S_{-k} D(S_k v)$ mit zyklischen Verschiebungen $S_k$ (kein Prox).

**Nicht-lokale Mittel.** $D(v)_i = \sum_j w_{ij} v_j / \sum_j w_{ij}$ mit $w_{ij} = \exp(-\|p_i - p_j\|^2/(5 h^2))$ und den Fenstern $p_i$ der Länge 5 (kein Prox).

**Gauß-MMSE.** $D(v) = P(P + s^2 I)^{-1} v$ mit dem exponentiellen Prior $P$ aus Stück 8. Fixpunkt: $x^\ast =(s^2/t\, P^{-1} + A^\top A)^{-1} A^\top b$;
bei $s^2 = t\sigma^2$ ist das das Posterior-Mittel $P A^\top(A P A^\top + \sigma^2 I)^{-1} b$ (Stück 7).

**Literatur.** Das Beispiel ist erfunden. Plug-and-Play: S. V. Venkatakrishnan, C. A. Bouman, B. Wohlberg (GlobalSIP 2013); Konvergenz mit nichtexpansiven Denoisern: E. K. Ryu u. a. (ICML 2019); Denoiser als Regularisierung:
Y. Romano, M. Elad, P. Milanfar (2017); nicht-lokale Mittel: A. Buades, B. Coll, J.-M. Morel (2005); Mittelung über Verschiebungen (cycle spinning): R. Coifman und D. Donoho (1995); ISTA und FISTA: I. Daubechies, M. Defrise,
C. De Mol (2004), A. Beck und M. Teboulle (2009) (alles aus dem Gedächtnis, nicht geprüft; die Zahlen hier sind gemessen, nicht der Literatur entnommen).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Inverse Probleme: vom Rauschen zur Verfahrenswahl](https://sebastianhanisch.net/konzepte-inverse-probleme.html)."
)
