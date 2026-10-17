"""PDF-Export der aktuellen Instanz (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import pnp_constants as C
from pnp_evaluation import Analysis, de, err_pct, noise_label, sci

_REPLACE = {"→": "->", "–": "-", "—": "-", "σ": "sigma", "λ": "lambda", "θ": "theta", "Γ": "Gamma", "×": "x", "÷": "/", "τ": "tau", "−": "-", "€": "EUR", "≥": ">=", "·": "*", "⁻": "-", "⁰": "0",
            "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}
MAX_ROWS = 40


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Plug-and-Play"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"Welt: {C.WORLD_SHORT[s.world]}, Rauschen {noise_label(s.noise)} des Effektivwerts, Denoiser: {C.DENOISER_SHORT[s.denoiser]}, Stärke {sci(a.strength)} "
                            f"({C.DENOISER_UNIT[s.denoiser]}, Faktor {de(s.factor, 1)} zur besten), {s.steps} Schritte{', mit FISTA-Impuls' if s.accel else ''}, Seed {s.seed}. Alles simuliert."),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    lines = [
        f"Fehler: Plug-and-Play {err_pct(a.err_pnp)}, Tikhonov {err_pct(a.err_tik)} (lambda = {sci(a.tik_lam)}), L1 exakt {err_pct(a.err_l1)} (lambda = {sci(a.l1_lam)}).",
        f"Beste Stärke des Gitters {sci(a.best_strength)}; Fixpunktrest {sci(a.resid)}; Lipschitz-Schätzung des Denoisers {de(a.lip, 3)}; wahres Rauschen sigma = {sci(a.sigma)}.",
    ]
    for line in lines:
        pdf.multi_cell(0, 5, _s(line), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    head, widths = ["Punkt", "Ort s", "Wahrheit x", "Plug-and-Play", "Tikhonov", "L1 exakt"], [16, 26, 34, 40, 34, 34]
    pdf.set_font("Helvetica", "B", 9)
    for h, w in zip(head, widths):
        pdf.cell(w, 6, _s(h), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for i in range(min(C.N, MAX_ROWS)):
        row = [f"{i + 1}", de(a.s[i], 3), de(a.x[i], 3), de(a.x_pnp[i], 3), de(a.x_tik[i], 3), de(a.x_l1[i], 3)]
        for v, w in zip(row, widths):
            pdf.cell(w, 5.5, _s(v), border=1)
        pdf.ln()
    if C.N > MAX_ROWS:
        pdf.ln(1)
        pdf.cell(0, 5, _s(f"... {C.N - MAX_ROWS} weitere Zeilen nicht gezeigt."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Plug-and-Play: ISTA mit austauschbarem Denoiser. Quelle: plug-and-play-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
