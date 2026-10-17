"""Auswertung: Formatierung, Welt und Instanz, die Analyse einer Einstellung, die vier Experimente über feste Ströme und die Verdikte. Nur numpy, kein Streamlit."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import pnp_constants as C
import pnp_model as model
from inverse_generator import SplitMix64, grid, phantom

_SUPER = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


# ----------------------------------------------------------------------------- Formatierung

def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def sci(x: float, digits: int = 1) -> str:
    """Zahl in Zehnerpotenz-Schreibweise, z. B. 2,5·10⁹."""
    if x == 0.0 or not math.isfinite(x):
        return de(x, digits) if math.isfinite(x) else "unendlich"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10.0 ** e
    if round(abs(m), digits) >= 10.0:
        m, e = m / 10.0, e + 1
    return f"{de(m, digits)}·10{str(e).translate(_SUPER)}"


def err_pct(err: float) -> str:
    """Fehler als Prozent der Signalnorm: ab 100 Prozent ohne, ab 1 Prozent mit einer, ab 0,01 Prozent mit zwei Stellen, darunter als Zehnerpotenz;
    ab dem Zehnfachen der Signalnorm als Vielfaches."""
    if not math.isfinite(err):
        return "unendlich"
    if err >= 10.0:
        return f"{sci(err)}-fach"
    x = 100.0 * err
    if x >= 99.5:
        return f"{de(x, 0)} %"
    if x >= 0.95:
        return f"{de(x, 1)} %"
    if x >= 0.01:
        return f"{de(x, 2)} %"
    return f"{sci(x)} %"


def noise_label(v: float) -> str:
    """Rauschstufe als Prozent des Effektivwerts, z. B. 0,1 %."""
    return de(100.0 * v, 4).rstrip("0").rstrip(",") + " %" if v > 0.0 else "0 %"


def rel_err(x: np.ndarray, xt: np.ndarray) -> np.ndarray:
    """Fehler in der 2-Norm relativ zur Norm der Wahrheit (je Zeile für Blöcke)."""
    return np.linalg.norm(np.asarray(x) - xt, axis=-1) / np.linalg.norm(xt)


# ----------------------------------------------------------------------------- Welt und Instanz

@lru_cache(maxsize=16)
def truth_signal(world: str, n: int = C.N) -> np.ndarray:
    """Die Wahrheit x: `phantom` (zwei Hügel und ein Kasten), `spikes` (drei Impulse; Masse mal n in der Zelle floor(Ort mal n)), `blocks` (drei Kästen) oder `waves`
    (Dreieckswelle mit vier Perioden zwischen 0,1 und 0,9)."""
    s = grid(n)
    if world == "phantom":
        x = phantom(s)
    elif world == "spikes":
        x = np.zeros(n)
        for pos, mass in C.SPIKES:
            x[int(math.floor(pos * n))] += mass * n
    elif world == "blocks":
        x = np.zeros(n)
        for left, right, height in C.BLOCKS:
            x = x + height * ((s >= left) & (s < right))
    else:
        lo, hi = C.WAVE_RANGE
        x = np.abs((C.WAVE_PERIODS * s) % 1.0 - 0.5) * 2.0 * ((s > lo) & (s < hi))
    x.flags.writeable = False
    return x


def noise_sigma(clean: np.ndarray, noise: float) -> float:
    """Absolute Standardabweichung des Rauschens: `noise` mal Effektivwert der exakten Daten."""
    return float(noise) * float(np.sqrt(np.mean(np.asarray(clean) ** 2)))


@lru_cache(maxsize=512)
def instance(world: str, noise: float, seed: int):
    """(x, b, sigma, clean): Wahrheit, Daten b = A x + e (das Modell stimmt, kein Gitterfehler), absolutes Rauschen, rauschfreie Daten."""
    x = truth_signal(world)
    clean = model.op() @ x
    sigma = noise_sigma(clean, noise)
    return x, clean + SplitMix64(seed).normal(C.N) * sigma, sigma, clean


def strength_grid(den: str, t: float) -> np.ndarray:
    """Gitter der Stärken (13 Werte) des Denoisers: Schwelle theta = t mal lambda (Weichschwelle, Haar), Gewichtsbreite h (nicht-lokale Mittel), Rauschniveau s (Gauß-MMSE)."""
    if den in ("soft", "haar", "haar_avg"):
        return t * np.array(C.LAMBDA_GRID)
    return np.array(C.NLM_GRID if den == "nlm" else C.GAUSS_GRID)


def lambda_of(den: str, strength: float, t: float) -> float:
    """Die zugehörige L1-Gewichtung lambda = theta / t der Prox-Denoiser; nan bei den anderen."""
    return strength / t if den in ("soft", "haar", "haar_avg") else math.nan


@dataclass(frozen=True)
class Settings:
    world: str = "blocks"
    noise: float = C.DEFAULT_NOISE
    denoiser: str = "haar_avg"
    factor: float = C.DEFAULT_FACTOR
    steps: int = C.DEFAULT_STEPS
    accel: bool = False
    seed: int = C.DEFAULT_SEED


@dataclass
class Analysis:
    settings: Settings
    s: np.ndarray
    x: np.ndarray
    b: np.ndarray
    sigma: float
    t: float                  # Schrittweite
    x_pnp: np.ndarray
    x_tik: np.ndarray
    x_l1: np.ndarray
    err_pnp: float
    err_tik: float
    err_l1: float
    tik_lam: float
    l1_lam: float
    strength: float           # gewählte Stärke (Faktor mal beste)
    best_strength: float
    best_index: int           # Index der besten Stärke im Gitter
    grid: np.ndarray          # Gitter der Stärken
    grid_errs: np.ndarray     # Fehler nach `steps` Schritten je Stärke des Gitters
    err_hist: np.ndarray      # Fehler über die Schritte (steps + 1)
    resid: float              # Fixpunktrest nach den Schritten
    lip: float                # Lipschitz-Schätzung des Denoisers bei der gewählten Stärke
    step_in: np.ndarray       # Eingang des Denoisers im letzten Schritt
    step_out: np.ndarray      # Ausgang (der nächste Iterierte)
    at_edge: bool             # beste Stärke ist die kleinste des Gitters: der Denoiser trägt nichts bei


def analyse(settings: Settings) -> Analysis:
    st = settings
    n = C.N
    x, b, sigma, _ = instance(st.world, st.noise, st.seed)
    a = model.op()
    t = model.step_size(a)
    g = strength_grid(st.denoiser, t)
    rows = model.make_denoiser(st.denoiser, n, g)
    xs, _ = model.pnp_run(a, np.tile(b, (len(g), 1)), rows, [st.steps], accel=st.accel)
    grid_errs = rel_err(xs[0], x)
    k = int(np.argmin(grid_errs))
    strength = float(g[k]) * st.factor
    den = model.make_denoiser(st.denoiser, n, [strength])
    xp, hist = model.pnp_run(a, b[None, :], den, [st.steps], accel=st.accel, truth=x)
    x_pnp = xp[0, 0]
    tik = model.tikhonov_svd(a, b, C.TIK_GRID)[:, 0, :]
    tik_errs = rel_err(tik, x)
    kt = int(np.argmin(tik_errs))
    l1 = np.array([model.lasso(a, b, lam) for lam in C.L1_GRID])
    l1_errs = rel_err(l1, x)
    kl = int(np.argmin(l1_errs))
    step_in = x_pnp - t * (a.T @ (a @ x_pnp - b))
    return Analysis(
        settings=st, s=grid(n), x=x, b=b, sigma=sigma, t=t, x_pnp=x_pnp, x_tik=tik[kt], x_l1=l1[kl], err_pnp=float(rel_err(x_pnp, x)), err_tik=float(tik_errs[kt]), err_l1=float(l1_errs[kl]),
        tik_lam=float(C.TIK_GRID[kt]), l1_lam=float(C.L1_GRID[kl]), strength=strength, best_strength=float(g[k]), best_index=k, grid=g, grid_errs=grid_errs, err_hist=hist[:, 0],
        resid=float(model.fixed_point_residual(a, b, den, x_pnp)[0]), lip=model.lipschitz_estimate(st.denoiser, n, strength, x, C.SEED_BASE + 77), step_in=step_in, step_out=den(step_in[None, :])[0],
        at_edge=k == 0,
    )


# ----------------------------------------------------------------------------- Experiment 1: PnP ist ISTA

def prox_experiment() -> dict:
    """Teil 1: Weichschwelle auf einer gut gestellten Matrix gegen den exakten L1-Löser (Abstand nach WP_STEPS Schritten). Teil 2: dieselbe Weichschwelle auf dem schlecht gestellten Operator:
    Zielfunktionslücke und Abstand zum exakten Löser nach PROX_CHECKS Schritten. Teil 3: Gauß-MMSE-Denoiser, Fixpunkt gegen die geschlossene Form und das Posterior-Mittel (Stück 7), bei
    s = Faktor mal sqrt(t) sigma. Teil 4: Haar-Denoiser, Zielfunktion über die Schritte (monoton) und Fixpunktrest."""
    n = C.N
    a = model.op()
    t = model.step_size(a)
    x, b, sigma, _ = instance("phantom", 1e-2, C.SEED_BASE)
    aw = model.wellposed_matrix(n)
    bw = aw @ x + SplitMix64(C.SEED_BASE + 5).normal(n) * 0.01
    tw = model.step_size(aw)
    wp = []
    for lam in C.WP_LAMBDAS:
        den = model.make_denoiser("soft", n, [tw * lam])
        xk = model.pnp_run(aw, bw[None, :], den, [C.WP_STEPS])[0][0, 0]
        wp.append({"lam": lam, "dist": float(np.abs(xk - model.lasso(aw, bw, lam)).max()), "cond": float(np.linalg.cond(aw))})
    lam = C.PROX_LAMBDA
    ex = model.lasso(a, b, lam)
    den = model.make_denoiser("soft", n, [t * lam])
    xs, _ = model.pnp_run(a, b[None, :], den, C.PROX_CHECKS)
    fex = model.objective_l1(a, b, ex, lam)
    l1 = [{"steps": k, "gap": model.objective_l1(a, b, xs[i, 0], lam) - fex, "dist": float(np.abs(xs[i, 0] - ex).max())} for i, k in enumerate(C.PROX_CHECKS)]
    p = model.prior_cov(grid(n))
    pm = model.posterior_mean(a, b, sigma, p)
    gauss = []
    for fac in C.GAUSS_FACTORS:
        s = fac * math.sqrt(t) * sigma
        xk = model.pnp_run(a, b[None, :], model.make_denoiser("gauss", n, [s]), [C.GAUSS_STEPS])[0][0, 0]
        closed = model.gauss_closed_form(a, b, s, p, t)
        gauss.append({"factor": fac, "dist_closed": float(np.abs(xk - closed).max()), "dist_pm": float(np.abs(xk - pm).max()), "err_pnp": float(rel_err(xk, x)), "err_pm": float(rel_err(pm, x))})
    haar = []
    for lam in (1e-3, 1e-2):
        den = model.make_denoiser("haar", n, [t * lam])
        xk = np.zeros((1, n))
        fs = []
        for _ in range(C.HAAR_STEPS):
            xk = model.pnp_step(a, b[None, :], xk, den, t)
            fs.append(model.objective_haar(a, b, xk[0], lam))
        xl = model.pnp_run(a, b[None, :], den, [2000])[0][0]
        haar.append({"lam": lam, "monotone": bool(all(fs[i + 1] <= fs[i] + 1e-12 for i in range(len(fs) - 1))), "f_first": fs[0], "f_last": fs[-1], "resid": float(model.fixed_point_residual(a, b, den, xl)[0])})
    return {"wp": wp, "l1": l1, "gauss": gauss, "haar": haar, "lam": lam}


# ----------------------------------------------------------------------------- Experiment 2: der Denoiser gehört zur Welt

def _den_setup(den: str, t: float):
    """(Gitter der Stärken im Experiment, Zahl der Ströme): nicht-lokale Mittel mit weniger Strömen (teuer)."""
    return strength_grid(den, t), (C.NLM_REPS if den == "nlm" else C.EXP_REPS)


def _best_over_grid(a, bs: np.ndarray, x: np.ndarray, den: str, g: np.ndarray, checkpoints, accel: bool = False):
    """Median-Fehler je Stärke und Schrittzahl: (Fehler (len(checkpoints), len(g)), Index der besten Stärke je Schrittzahl)."""
    reps = len(bs)
    rows = np.repeat(g, reps)
    xs, _ = model.pnp_run(a, np.tile(bs, (len(g), 1)), model.make_denoiser(den, C.N, rows), checkpoints, accel=accel)
    errs = np.array([np.median(rel_err(xk, x).reshape(len(g), reps), axis=1) for xk in xs])
    return errs, errs.argmin(axis=1)


def _streams(world: str, noise: float, reps: int):
    ins = [instance(world, noise, C.SEED_BASE + r) for r in range(reps)]
    return ins[0][0], np.array([i[1] for i in ins])


def _baselines(a, bs: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """(Tikhonov, L1 exakt): bester gemeinsamer Parameter, Median-Fehler über die Ströme."""
    tik = rel_err(model.tikhonov_svd(a, bs, C.TIK_GRID), x)
    l1 = np.array([[np.linalg.norm(model.lasso(a, bb, lam) - x) / np.linalg.norm(x) for bb in bs] for lam in C.L1_GRID])
    return float(np.median(tik, axis=1).min()), float(np.median(l1, axis=1).min())


def world_experiment() -> dict:
    """Je Welt und Rauschstufe: bester Median-Fehler (gemeinsame Stärke über die Ströme) jedes Denoisers nach WORLD_STEPS Schritten gegen Tikhonov und L1 (exakt)."""
    a = model.op()
    t = model.step_size(a)
    rows = []
    for world in C.WORLDS:
        for noise in C.WORLD_NOISES:
            x, bs = _streams(world, noise, C.EXP_REPS)
            tik, l1 = _baselines(a, bs, x)
            errs, strengths = {}, {}
            for den in C.DENOISERS:
                g, reps = _den_setup(den, t)
                e, k = _best_over_grid(a, bs[:reps], x, den, g, [C.WORLD_STEPS])
                errs[den], strengths[den] = float(e[0, k[0]]), float(g[k[0]])
            rows.append({"world": world, "noise": noise, "tik": tik, "l1": l1, "errs": errs, "strengths": strengths})
    return {"rows": rows}


# ----------------------------------------------------------------------------- Experiment 3: Schrittzahl und Stärke

def steps_experiment() -> dict:
    """Teil 1: schwache Stärke, Fehler nach WEAK_CHECKS Schritten (Phantom, Rauschen 0,1 %): alle Denoiser gleich (Landweber). Teil 2: bester Fehler je Schrittzahl (STEP_CHECKS, beste Stärke je
    Schrittzahl) für Phantom, Blöcke und Wellen bei 1 % Rauschen (nicht-lokale Mittel nur bei den Wellen). Teil 3: FISTA-Impuls bei der besten Stärke nach 300 Schritten (Phantom)."""
    a = model.op()
    t = model.step_size(a)
    x, bs = _streams("phantom", C.WEAK_NOISE, C.EXP_REPS)
    weak = {}
    for den in C.DENOISERS:
        s = C.WEAK_STRENGTH[den] * (t if den in ("soft", "haar", "haar_avg") else 1.0)
        xs, _ = model.pnp_run(a, bs, model.make_denoiser(den, C.N, np.full(len(bs), s)), C.WEAK_CHECKS)
        weak[den] = [float(np.median(rel_err(xk, x))) for xk in xs]
    conv = []
    for world in C.STEP_WORLDS:
        x, bs = _streams(world, C.STEP_NOISE, C.EXP_REPS)
        for den in C.DENOISERS:
            if den == "nlm" and world != "waves":
                continue
            g, reps = _den_setup(den, t)
            if den in ("haar_avg", "nlm"):
                g = g[::2]
            e, k = _best_over_grid(a, bs[:reps], x, den, g, C.STEP_CHECKS)
            conv.append({"world": world, "den": den, "errs": [float(e[i, k[i]]) for i in range(len(C.STEP_CHECKS))], "strengths": [float(g[k[i]]) for i in range(len(C.STEP_CHECKS))]})
    x, bs = _streams("phantom", C.STEP_NOISE, C.EXP_REPS)
    accel = []
    for den in C.DENOISERS:
        g, reps = _den_setup(den, t)
        e, k = _best_over_grid(a, bs[:reps], x, den, g, [C.WORLD_STEPS])
        ea, _ = _best_over_grid(a, bs[:reps], x, den, g[k[0]:k[0] + 1], [C.WORLD_STEPS], accel=True)
        accel.append({"den": den, "plain": float(e[0, k[0]]), "accel": float(ea[0, 0])})
    return {"weak": weak, "conv": conv, "accel": accel}


# ----------------------------------------------------------------------------- Experiment 4: ohne Prox

def convergence_experiment() -> dict:
    """Je Denoiser und Stärke (Phantom, Rauschen 1 %): Lipschitz-Schätzung, Fixpunktrest nach CONV_STEPS Schritten, Abstand der Grenzwerte aus zwei Starts (0 und die Daten), Fehler. Dazu die
    Verschiebungsabweichung |D(S v) - S D(v)| / |D(v)| für eine zyklische Verschiebung um 1."""
    n = C.N
    a = model.op()
    t = model.step_size(a)
    x, b, sigma, _ = instance("phantom", C.CONV_NOISE, C.SEED_BASE)
    rows = []
    for den, strengths in C.CONV_STRENGTHS.items():
        for sv in strengths:
            s = sv * (t if den in ("soft", "haar", "haar_avg") else 1.0)
            d = model.make_denoiser(den, n, [s])
            xa = model.pnp_run(a, b[None, :], d, [C.CONV_STEPS])[0][0, 0]
            xb = model.pnp_run(a, b[None, :], d, [C.CONV_STEPS], x0=b[None, :])[0][0, 0]
            v = x + 0.3 * SplitMix64(C.SEED_BASE + 3).normal(n)
            dv, dsv = d(v[None, :])[0], d(np.roll(v, 1)[None, :])[0]
            rows.append({"den": den, "param": sv, "strength": s, "lip": model.lipschitz_estimate(den, n, s, x, C.SEED_BASE + 77), "resid": float(model.fixed_point_residual(a, b, d, xa)[0]),
                         "limit_dist": float(np.abs(xa - xb).max()), "err": float(rel_err(xa, x)), "shift_dev": float(np.linalg.norm(dsv - np.roll(dv, 1)) / np.linalg.norm(dv))})
    return {"rows": rows}


# ----------------------------------------------------------------------------- Verdikte: (Art, Text)

def _small(v: float) -> str:
    """Abstände unter 10⁻⁹ als Schranke, sonst in Zehnerpotenz-Schreibweise."""
    return "unter 10⁻⁹" if v < 1e-9 else sci(v)


def _by(rows: list, **kw) -> dict:
    return next(r for r in rows if all(r[k] == v for k, v in kw.items()))


def verdict_instance(a: Analysis) -> tuple[str, str]:
    """Hilft der Denoiser? Fehler gegen Tikhonov (die Standardlösung der Linie) mit der Schwelle GAIN; dazu Hinweise auf den Rand des Stärkegitters, den Fehlerverlauf (Abbruch regularisiert mit),
    das exakte L1 bei der Weichschwelle, den Fixpunktrest und die Lipschitz-Schätzung."""
    st = a.settings
    lead = (f"{C.DENOISER_SHORT[st.denoiser]}, {C.WORLD_SHORT[st.world]}, Rauschen {noise_label(st.noise)}, {st.steps} Schritte{', mit FISTA-Impuls' if st.accel else ''}: Fehler {err_pct(a.err_pnp)} "
            f"(Tikhonov {err_pct(a.err_tik)}, L1 exakt {err_pct(a.err_l1)}); Stärke {sci(a.strength)} ({C.DENOISER_UNIT[st.denoiser]}), Fixpunktrest {sci(a.resid)}, "
            f"Lipschitz-Schätzung des Denoisers {de(a.lip, 3)}.")
    if a.err_pnp <= C.GAIN * a.err_tik:
        kind, text = "success", f"{lead} Der Denoiser hilft: der Fehler liegt unter 80 % des Fehlers von Tikhonov."
    elif a.err_pnp <= C.TIE * a.err_tik:
        kind, text = "info", f"{lead} Der Denoiser ist gleichwertig zu Tikhonov, aber nicht deutlich besser."
    else:
        kind, text = "warning", f"{lead} Der Denoiser passt nicht zur Welt: der Fehler ist größer als bei Tikhonov."
    if a.at_edge and st.factor == 1.0:
        text += " Die beste Stärke ist die kleinste des Gitters: der Denoiser trägt nichts bei, regularisiert hat allein die Schrittzahl."
    k = int(np.argmin(a.err_hist))
    if a.err_hist[k] <= C.EARLY * a.err_hist[-1]:
        text += f" Der Fehler war nach {k} Schritten mit {err_pct(float(a.err_hist[k]))} am kleinsten und steigt danach auf {err_pct(float(a.err_hist[-1]))}: der Abbruch regularisiert mit."
    if st.denoiser == "soft" and a.err_l1 < a.err_pnp:
        text += f" Das exakte L1 ist besser ({err_pct(a.err_l1)}): die Weichschwelle ist ISTA für L1 und nähert sich ihm mit mehr Schritten (Stück 6)."
    if a.resid > C.RES_OK:
        text += f" Die Iteration ist nach {st.steps} Schritten noch nicht am Fixpunkt (Rest {sci(a.resid)})."
    if a.lip > C.LIP_ONE:
        text += f" Die Lipschitz-Schätzung liegt über 1 ({de(a.lip, 3)}): der Denoiser ist nicht nichtexpansiv, eine Konvergenzgarantie gibt es nicht."
    return kind, text


def verdict_prox(res: dict) -> tuple[str, str]:
    wp, l1, g, h = res["wp"], res["l1"], res["gauss"], res["haar"]
    g1, g3 = _by(g, factor=1.0), _by(g, factor=3.0)
    return "info", (f"Die Weichschwelle ist ISTA für L1: auf der gut gestellten Matrix (Kondition {de(wp[0]['cond'], 1)}) liegt Plug-and-Play nach {C.WP_STEPS} Schritten {_small(max(r['dist'] for r in wp))} vom "
                    f"exakten Löser; auf dem schlecht gestellten Operator fällt die Zielfunktionslücke ({sci(l1[0]['gap'])} / {sci(l1[1]['gap'])} / {sci(l1[2]['gap'])} nach 100 / 1000 / 10000 Schritten), aber der "
                    f"Abstand im x bleibt {de(min(r['dist'] for r in l1), 2)} bis {de(max(r['dist'] for r in l1), 2)}. Der Gauß-MMSE-Denoiser liegt bei s = √t σ {_small(g1['dist_pm'])} vom Posterior-Mittel "
                    f"(Fehler {err_pct(g1['err_pnp'])}); bei s = 3 √t σ ist er die geschlossene Form ({_small(g3['dist_closed'])}), aber nicht das Posterior-Mittel (Fehler {err_pct(g3['err_pnp'])} statt "
                    f"{err_pct(g3['err_pm'])}). Haar: die Zielfunktion fällt in {C.HAAR_STEPS} Schritten {'monoton' if all(r['monotone'] for r in h) else 'nicht monoton'}.")


def verdict_world(res: dict) -> tuple[str, str]:
    rows = res["rows"]
    sp = _by(rows, world="spikes", noise=1e-2)
    bl = _by(rows, world="blocks", noise=1e-2)
    wv = _by(rows, world="waves", noise=1e-1)
    others = [sp["errs"][d] for d in ("haar", "haar_avg", "nlm", "gauss")]
    return "warning", (f"Bei Spitzen (Rauschen 1 %) erreicht nur die Weichschwelle {err_pct(sp['errs']['soft'])} (Tikhonov {err_pct(sp['tik'])}); Haar, Haar gemittelt, nicht-lokale Mittel und Gauß-MMSE "
                       f"liegen bei {err_pct(min(others))} bis {err_pct(max(others))}. Bei Blöcken ist Haar gemittelt mit {err_pct(bl['errs']['haar_avg'])} am besten, Haar ohne Mittelung hat "
                       f"{err_pct(bl['errs']['haar'])} (Tikhonov {err_pct(bl['tik'])}). Bei Wellen mit 10 % Rauschen: nicht-lokale Mittel {err_pct(wv['errs']['nlm'])}, Gauß-MMSE {err_pct(wv['errs']['gauss'])}, "
                       f"Tikhonov {err_pct(wv['tik'])}.")


def verdict_steps(res: dict) -> tuple[str, str]:
    weak = res["weak"]
    cp = {d: weak[d] for d in C.DENOISERS}
    i10, i100 = C.WEAK_CHECKS.index(10), C.WEAK_CHECKS.index(100)
    spread = max(max(cp[d][i] for d in C.DENOISERS) - min(cp[d][i] for d in C.DENOISERS) for i in range(i100 + 1))
    ph = _by(res["conv"], world="phantom", den="soft")["errs"]
    wv = _by(res["conv"], world="waves", den="soft")["errs"]
    gw = _by(res["conv"], world="waves", den="gauss")["errs"]
    ac = _by(res["accel"], den="soft")
    return "warning", (f"Bei schwacher Stärke liefern alle fünf Denoiser bis 100 Schritte dieselbe Fehlerkurve ({err_pct(cp['soft'][i10])} nach 10, {err_pct(cp['soft'][i100])} nach 100 Schritten, Spannweite "
                       f"{err_pct(spread)}): es regularisiert die Schrittzahl (Stück 5). Mit bester Stärke je Schrittzahl wird die Weichschwelle beim Phantom von {err_pct(ph[1])} (300 Schritte) auf "
                       f"{err_pct(ph[2])} (3000 Schritte) schlechter, bei den Wellen von {err_pct(wv[1])} auf {err_pct(wv[2])}; nur der Gauß-MMSE-Denoiser bleibt stabil ({err_pct(gw[0])} / {err_pct(gw[1])} / "
                       f"{err_pct(gw[2])} nach 100 / 300 / 3000 Schritten). Der FISTA-Impuls verschlechtert die Weichschwelle von {err_pct(ac['plain'])} auf {err_pct(ac['accel'])}.")


def verdict_convergence(res: dict) -> tuple[str, str]:
    rows = res["rows"]
    nlm = [r for r in rows if r["den"] == "nlm"]
    prox = [r for r in rows if r["den"] != "nlm"]
    return "info", (f"Die Lipschitz-Schätzung liegt bei Weichschwelle, Haar, Haar gemittelt und Gauß-MMSE höchstens bei {de(max(r['lip'] for r in prox), 3)} (in der Stichprobe nichtexpansiv), bei den "
                    f"nicht-lokalen Mitteln mit h = {' / '.join(de(r['param'], 2) for r in nlm)} bei {' / '.join(de(r['lip'], 3) for r in nlm)}: über 1. Trotzdem ist der Fixpunktrest nach {C.CONV_STEPS} Schritten "
                    f"höchstens {sci(max(r['resid'] for r in rows))}: keine Divergenz gemessen, aber auch keine Garantie. Haar ohne Mittelung verfehlt die Verschiebungsgleichheit um "
                    f"{sci(_by(rows, den='haar')['shift_dev'])}, Haar gemittelt um {_small(_by(rows, den='haar_avg', param=1e-3)['shift_dev'])}.")
