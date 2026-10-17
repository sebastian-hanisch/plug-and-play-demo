"""Plotly-Figuren und die Tabelle der Instanz. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import pnp_constants as C
from pnp_evaluation import Analysis, de, noise_label, sci


def _fixed(fig: go.Figure, height: int = 360, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _ref_line(fig: go.Figure, value: float, line: dict, text: str, position: str, axis: str) -> None:
    """Referenzlinie mit Beschriftung auf einer logarithmischen Achse: die Linie liegt auf dem Datenwert, die Beschriftung braucht log10 des Werts
    (add_hline und add_vline setzen beide auf denselben Wert, die Beschriftung sitzt dann falsch und dehnt die Achse)."""
    if not value > 0.0:
        return
    pos = math.log10(value)
    top, left = "top" in position, "left" in position
    if axis == "y":
        fig.add_hline(y=value, line=line)
        fig.add_annotation(xref="paper", yref="y", x=0 if left else 1, y=pos, text=text, showarrow=False, xanchor="left" if left else "right", yanchor="bottom" if top else "top")
    else:
        fig.add_vline(x=value, line=line)
        fig.add_annotation(xref="x", yref="paper", x=pos, y=1 if top else 0, text=text, showarrow=False, xanchor="right" if left else "left", yanchor="top" if top else "bottom")


def build_table(a: Analysis, rows: int = 40) -> pd.DataFrame:
    k = min(rows, C.N)
    return pd.DataFrame({"Punkt": [f"{i + 1}" for i in range(k)], "Ort s": a.s[:k], "Daten b": a.b[:k], "Wahrheit x": a.x[:k], "Plug-and-Play": a.x_pnp[:k], "Tikhonov": a.x_tik[:k], "L1 exakt": a.x_l1[:k]})


def style_table(df: pd.DataFrame):
    return df.style.format({c: (lambda v: de(v, 4)) for c in df.columns if c != "Punkt"})


def build_recon_chart(a: Analysis) -> go.Figure:
    """Wahrheit, Plug-and-Play, Tikhonov und L1 (exakt) bei ihren besten Parametern; die Achse ist fest (je Welt)."""
    lo, hi = C.SHOW_RANGE[a.settings.world]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a.s, y=a.x, mode="lines", name="Wahrheit", line=dict(color=C.COLOR_TRUTH, width=3), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=a.s, y=a.x_tik, mode="lines", name=f"Tikhonov (λ = {sci(a.tik_lam)})", line=dict(color=C.COLOR_TIK, width=2, dash="dash"), hovertemplate="s %{x:.3f}<br>x̂ %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=a.s, y=a.x_l1, mode="lines", name=f"L1 exakt (λ = {sci(a.l1_lam)})", line=dict(color=C.COLOR_L1, width=1.5, dash="dot"), hovertemplate="s %{x:.3f}<br>x̂ %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=a.s, y=a.x_pnp, mode="lines", name="Plug-and-Play", line=dict(color=C.COLOR_PNP, width=2.5), hovertemplate="s %{x:.3f}<br>x̂ %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title="Ort s")
    fig.update_yaxes(title="Wert", range=[lo, hi])
    return _fixed(fig, 400)


def build_history_chart(a: Analysis) -> go.Figure:
    """Fehler über die Schritte (logarithmisch), mit dem kleinsten Fehler markiert und Tikhonov und L1 als Linien."""
    steps = np.arange(len(a.err_hist))
    k = int(np.argmin(a.err_hist))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps[1:], y=a.err_hist[1:], mode="lines", name="Plug-and-Play", line=dict(color=C.COLOR_PNP, width=3), hovertemplate="Schritt %{x}<br>Fehler %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[max(k, 1)], y=[float(a.err_hist[max(k, 1)])], mode="markers", name=f"kleinster Fehler (Schritt {k})", marker=dict(symbol="x", size=12, color=C.COLOR_L1, line=dict(width=2))))
    _ref_line(fig, a.err_tik, dict(color=C.COLOR_TIK, width=2, dash="dash"), "Tikhonov", "top left", "y")
    _ref_line(fig, a.err_l1, dict(color=C.COLOR_L1, width=1.5, dash="dot"), "L1 exakt", "bottom left", "y")
    fig.update_xaxes(title="Schritt", type="log")
    fig.update_yaxes(title="relativer Fehler", type="log", exponentformat="power")
    return _fixed(fig, 340)


def build_strength_chart(a: Analysis) -> go.Figure:
    """Fehler nach den gewählten Schritten über die Stärke des Denoisers (Gitter), mit bester und gewählter Stärke."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a.grid, y=a.grid_errs, mode="lines+markers", name="Plug-and-Play", line=dict(color=C.COLOR_PNP, width=3), marker=dict(size=6), hovertemplate="Stärke %{x:.2e}<br>Fehler %{y:.3f}<extra></extra>"))
    _ref_line(fig, a.err_tik, dict(color=C.COLOR_TIK, width=2, dash="dash"), "Tikhonov", "top left", "y")
    fig.add_trace(go.Scatter(x=[a.best_strength], y=[float(a.grid_errs[a.best_index])], mode="markers", name="beste Stärke", marker=dict(symbol="x", size=12, color=C.COLOR_L1, line=dict(width=2))))
    if a.settings.factor != 1.0:
        _ref_line(fig, a.strength, dict(color=C.COLOR_GREY, width=1.5, dash="dot"), "gewählt", "top right", "x")
    fig.update_xaxes(title=f"Stärke ({C.DENOISER_UNIT[a.settings.denoiser]})", type="log", exponentformat="power")
    fig.update_yaxes(title="relativer Fehler", type="log", exponentformat="power")
    return _fixed(fig, 340)


def build_step_chart(a: Analysis) -> go.Figure:
    """Ein Schritt am Ende: Eingang des Denoisers (Gradientenschritt) und sein Ausgang, mit der Wahrheit; die Achse ist fest (je Welt)."""
    lo, hi = C.SHOW_RANGE[a.settings.world]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a.s, y=a.x, mode="lines", name="Wahrheit", line=dict(color=C.COLOR_TRUTH, width=3), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=a.s, y=a.step_in, mode="lines", name="Eingang x − t Aᵀ(A x − b)", line=dict(color=C.COLOR_GREY, width=1.5), hovertemplate="s %{x:.3f}<br>%{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=a.s, y=a.step_out, mode="lines", name="Ausgang D(…)", line=dict(color=C.COLOR_PNP, width=2.5), hovertemplate="s %{x:.3f}<br>%{y:.3f}<extra></extra>"))
    fig.update_xaxes(title="Ort s")
    fig.update_yaxes(title="Wert", range=[lo, hi])
    return _fixed(fig, 340)


# ----------------------------------------------------------------------------- Experimente

_COLORS = {"soft": "#c0392b", "haar": "#8a8a8a", "haar_avg": "#2F6B65", "nlm": "#4c78a8", "gauss": "#D68A2E"}


def build_prox_chart(res: dict) -> go.Figure:
    """Zielfunktionslücke von ISTA mit der Weichschwelle gegen den exakten L1-Löser über die Schritte (schlecht gestellter Operator)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[r["steps"] for r in res["l1"]], y=[r["gap"] for r in res["l1"]], mode="lines+markers", name="Zielfunktionslücke F(x_K) − F(x*)", line=dict(color=C.COLOR_PNP, width=3), marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=[r["steps"] for r in res["l1"]], y=[r["dist"] for r in res["l1"]], mode="lines+markers", name="Abstand max |x_K − x*|", line=dict(color=C.COLOR_L1, width=3), marker=dict(size=8)))
    fig.update_xaxes(title="Schritte K", type="log", tickvals=[r["steps"] for r in res["l1"]])
    fig.update_yaxes(title="Weichschwelle gegen exaktes L1", type="log", exponentformat="power")
    return _fixed(fig, 340)


def build_gauss_chart(res: dict) -> go.Figure:
    """Abstand des Gauß-MMSE-Fixpunkts zum Posterior-Mittel und zur geschlossenen Form je Stärke s = Faktor mal sqrt(t) sigma."""
    xs = [f"{de(r['factor'], 1)} √t σ" for r in res["gauss"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=xs, y=[max(r["dist_pm"], 1e-16) for r in res["gauss"]], name="Abstand zum Posterior-Mittel", marker_color=C.COLOR_PNP))
    fig.add_trace(go.Bar(x=xs, y=[max(r["dist_closed"], 1e-16) for r in res["gauss"]], name="Abstand zur geschlossenen Form", marker_color=C.COLOR_TIK))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Stärke s des Gauß-MMSE-Denoisers", type="category")
    fig.update_yaxes(title="max-Abstand nach 20000 Schritten", type="log", exponentformat="power")
    return _fixed(fig, 340)


def build_world_chart(res: dict) -> go.Figure:
    """Bester Fehler nach Welt bei Rauschen 1 %: Tikhonov, L1 (exakt) und die fünf Denoiser."""
    rows = {r["world"]: r for r in res["rows"] if r["noise"] == 1e-2}
    worlds = [w for w in C.WORLDS if w in rows]
    labels = [C.WORLD_SHORT[w] for w in worlds]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[100 * rows[w]["tik"] for w in worlds], name="Tikhonov", marker_color="#222222"))
    fig.add_trace(go.Bar(x=labels, y=[100 * rows[w]["l1"] for w in worlds], name="L1 exakt", marker_color="#bbbbbb"))
    for den in C.DENOISERS:
        fig.add_trace(go.Bar(x=labels, y=[100 * rows[w]["errs"][den] for w in worlds], name=C.DENOISER_SHORT[den], marker_color=_COLORS[den]))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Welt (Rauschen 1 %)", type="category")
    fig.update_yaxes(title="Fehler (%), bester Parameter", range=[0, 100])
    return _fixed(fig, 380)


def build_weak_chart(res: dict) -> go.Figure:
    """Fehler über die Schritte bei schwacher Stärke: die Kurven der fünf Denoiser liegen übereinander (Landweber)."""
    fig = go.Figure()
    for den in C.DENOISERS:
        fig.add_trace(go.Scatter(x=list(C.WEAK_CHECKS), y=[100 * v for v in res["weak"][den]], mode="lines+markers", name=C.DENOISER_SHORT[den], line=dict(color=_COLORS[den], width=2), marker=dict(size=6)))
    fig.update_xaxes(title="Schritte", type="log", tickvals=list(C.WEAK_CHECKS))
    fig.update_yaxes(title="Fehler (%), Phantom, Rauschen 0,1 %", range=[0, 15])
    return _fixed(fig, 340)


def build_stop_chart(res: dict) -> go.Figure:
    """Bester Fehler nach 100, 300 und 3000 Schritten (beste Stärke je Schrittzahl), Wellen, Rauschen 1 %."""
    rows = [r for r in res["conv"] if r["world"] == "waves"]
    fig = go.Figure()
    for r in rows:
        fig.add_trace(go.Scatter(x=[str(k) for k in C.STEP_CHECKS], y=[100 * v for v in r["errs"]], mode="lines+markers", name=C.DENOISER_SHORT[r["den"]], line=dict(color=_COLORS[r["den"]], width=3), marker=dict(size=8)))
    fig.update_xaxes(title="Schritte", type="category")
    fig.update_yaxes(title="Fehler (%), Wellen, Rauschen 1 %, beste Stärke je Schrittzahl", range=[0, 12])
    return _fixed(fig, 340)


def build_lipschitz_chart(res: dict) -> go.Figure:
    """Lipschitz-Schätzung der Denoiser (Stichprobe); über der Linie 1 ist der Denoiser sicher nicht nichtexpansiv."""
    labels = [f"{C.DENOISER_SHORT[r['den']]} ({de(r['param'], 3)})" for r in res["rows"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[r["lip"] for r in res["rows"]], marker_color=[_COLORS[r["den"]] for r in res["rows"]], text=[de(r["lip"], 3) for r in res["rows"]], textposition="outside"))
    fig.add_hline(y=1.0, line=dict(color=C.COLOR_REF, width=1.5, dash="dash"))
    fig.update_xaxes(title="Denoiser (Stärke)", type="category")
    fig.update_yaxes(title="Lipschitz-Schätzung", range=[0.9, 1.15])
    return _fixed(fig, 340, showlegend=False)
