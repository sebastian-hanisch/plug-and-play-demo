"""Kern: Denoiser (Weichschwelle, Haar-Wavelet, Haar über Verschiebungen gemittelt, nicht-lokale Mittel, Gauß-MMSE), Plug-and-Play-ISTA, Vergleichslöser (Tikhonov, L1 per Aktivmenge),
Lipschitz-Schätzung und Fixpunktrest. Nur numpy, kein Streamlit, kein scipy.

Jede Funktion ist eine kleine, einzeln prüfbare Einheit (DEMO-PLAYBOOK §1b): `tests/test_model.py` rechnet sie an der 3 x 3-Mini-Instanz von Hand und
`tests/test_oracle_pnp.py` gegen die Löser aus Stück 6 und 7 (Weichschwelle gegen L1 auf einer gut gestellten Matrix, Gauß-MMSE gegen das Posterior-Mittel), die Orthonormalität von Haar,
die Verschiebungsgleichheit der Mittelung und die Landweber-Iteration.

Plug-and-Play-ISTA (Schrittweite t = 1 / |A|_2^2):   x <- D(x - t A^T (A x - b)),   D der Denoiser. Ist D der Prox von t R, ist das ISTA für 1/2 |A x - b|^2 + R (Stück 6);
für D = Weichschwelle mit Schwelle theta ist R = (theta / t) |x|_1. Der Gauß-MMSE-Denoiser D = P (P + s^2 I)^-1 hat den Fixpunkt x = (s^2 / t P^-1 + A^T A)^-1 A^T b: bei s^2 = t sigma^2 das
Posterior-Mittel des Gauß-Modells (Stück 7).

Alle Denoiser arbeiten auf Zeilenblöcken v (m, n) mit je einer Stärke je Zeile: so laufen mehrere Stärken oder Ströme in einer Schleife.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

import pnp_constants as C
from inverse_generator import SplitMix64, grid, operator

_TINY = 1e-300


# ----------------------------------------------------------------------------- Denoiser

def soft(v, t):
    """Weiche Schwelle sign(v) max(|v| - t, 0): die Prox-Abbildung von t |.|_1."""
    v = np.asarray(v, dtype=float)
    return np.sign(v) * np.maximum(np.abs(v) - t, 0.0)


def n_approx(n: int, levels: int) -> int:
    """Zahl der Näherungskoeffizienten der Haar-Zerlegung: n / 2^levels."""
    return n // 2 ** levels


@lru_cache(maxsize=16)
def _haar_matrix(n: int, levels: int) -> np.ndarray:
    def fwd(v):
        out, cur = [], v.copy()
        for _ in range(levels):
            out.append((cur[0::2] - cur[1::2]) / math.sqrt(2.0))
            cur = (cur[0::2] + cur[1::2]) / math.sqrt(2.0)
        return np.concatenate([cur] + out[::-1])

    w = np.array([fwd(e) for e in np.eye(n)]).T
    w.flags.writeable = False
    return w


def haar_matrix(n: int, levels: int) -> np.ndarray:
    """Orthonormale Haar-Transformation (levels Ebenen) als Matrix: Zeilen = Koeffizienten, zuerst die n / 2^levels Näherungskoeffizienten, dann die Details von grob nach fein."""
    return _haar_matrix(int(n), int(levels))


def _haar(v: np.ndarray, th: np.ndarray, n: int) -> np.ndarray:
    w = haar_matrix(n, C.HAAR_LEVELS)
    c = v @ w.T
    k = n_approx(n, C.HAAR_LEVELS)
    c[:, k:] = soft(c[:, k:], th[:, None])
    return c @ w


def _haar_avg(v: np.ndarray, th: np.ndarray, n: int) -> np.ndarray:
    acc = np.zeros_like(v)
    for k in range(C.HAAR_SHIFTS):
        acc += np.roll(_haar(np.roll(v, k, axis=1), th, n), -k, axis=1)
    return acc / C.HAAR_SHIFTS


def _nlm(v: np.ndarray, h: np.ndarray, n: int) -> np.ndarray:
    """Nicht-lokale Mittel in 1D: Gewicht exp(-d2 / h^2) mit d2 = mittlerer quadratischer Abstand der Fenster (reflektierter Rand), gemittelt über alle Orte; die Abstände über die
    Gram-Matrix der Fenster (|p_i|^2 + |p_j|^2 - 2 p_i . p_j)."""
    r = C.NLM_HALF
    pad = np.pad(v, ((0, 0), (r, r)), mode="reflect")
    pat = np.stack([pad[:, k:k + n] for k in range(2 * r + 1)], axis=2)
    sq = (pat ** 2).sum(axis=2)
    d2 = sq[:, :, None] + sq[:, None, :] - 2.0 * np.einsum("mik,mjk->mij", pat, pat)
    d2 *= -1.0 / ((2 * r + 1) * h[:, None, None] ** 2)
    np.minimum(d2, 0.0, out=d2)
    np.exp(d2, out=d2)
    return np.einsum("mij,mj->mi", d2, v) / d2.sum(axis=2)


def exp_kernel(s: np.ndarray, ell: float) -> np.ndarray:
    """Korrelationsmatrix exp(-|s_i - s_j| / ell) (Stück 8)."""
    s = np.asarray(s, dtype=float)
    return np.exp(-np.abs(s[:, None] - s[None, :]) / ell)


def prior_cov(s: np.ndarray) -> np.ndarray:
    """Prior-Kovarianz tau^2 C + 1e-10 I (der Zuschlag hält die Zerlegung stabil)."""
    return C.PRIOR_TAU ** 2 * exp_kernel(s, C.PRIOR_ELL) + 1e-10 * np.eye(len(s))


@lru_cache(maxsize=8)
def _prior_eig(n: int):
    ev, q = np.linalg.eigh(prior_cov(grid(n)))
    return np.maximum(ev, 0.0), q


def _gauss(v: np.ndarray, s: np.ndarray, n: int) -> np.ndarray:
    """MMSE-Denoiser des Gauß-Priors: D(v) = P (P + s^2 I)^-1 v, über die Eigenzerlegung von P (jede Zeile mit eigenem s)."""
    ev, q = _prior_eig(n)
    return ((v @ q) * (ev[None, :] / (ev[None, :] + s[:, None] ** 2))) @ q.T


def make_denoiser(name: str, n: int, strengths):
    """Gibt eine Funktion zurück, die einen Block v (m, n) mit den Stärken `strengths` (m,) entrauscht. name: identity, soft, haar, haar_avg, nlm, gauss."""
    st = np.asarray(strengths, dtype=float).reshape(-1)
    if name == "identity":
        return lambda v: np.asarray(v, dtype=float)
    fn = {"soft": lambda v: soft(v, st[:, None]), "haar": lambda v: _haar(v, st, n), "haar_avg": lambda v: _haar_avg(v, st, n), "nlm": lambda v: _nlm(v, st, n),
          "gauss": lambda v: _gauss(v, st, n)}[name]
    return lambda v: fn(np.asarray(v, dtype=float))


# ----------------------------------------------------------------------------- Plug-and-Play-ISTA

def step_size(a: np.ndarray) -> float:
    """t = 1 / |A|_2^2."""
    return 1.0 / float(np.linalg.norm(a, 2) ** 2)


def pnp_step(a: np.ndarray, b: np.ndarray, x: np.ndarray, den, t: float) -> np.ndarray:
    """Ein Schritt D(x - t A^T (A x - b)) für Blöcke x, b (m, n)."""
    return den(x - t * ((x @ a.T - b) @ a))


def pnp_run(a: np.ndarray, b: np.ndarray, den, checkpoints, accel: bool = False, truth: np.ndarray | None = None, x0: np.ndarray | None = None):
    """PnP-ISTA (accel: mit FISTA-Impuls) für Blöcke b (m, n), Schrittweite 1 / |A|_2^2, Start 0 oder x0. Gibt (Iterierte an den Schrittzahlen `checkpoints`, (len, m, n); Fehlerverlauf)
    zurück; der Fehlerverlauf (Schritte + 1, m) ist der relative Abstand zu `truth` (Schritt 0 = Start) oder None."""
    b = np.atleast_2d(np.asarray(b, dtype=float))
    t = step_size(a)
    cps = sorted(set(int(c) for c in checkpoints))
    x = np.zeros_like(b) if x0 is None else np.array(x0, dtype=float).reshape(b.shape)
    y, mom = x.copy(), 1.0
    errs = None if truth is None else [np.linalg.norm(x - truth[None, :], axis=1) / np.linalg.norm(truth)]
    out = {}
    last = cps[-1]
    for k in range(1, last + 1):
        xn = pnp_step(a, b, y, den, t)
        if accel:
            mn = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * mom * mom))
            y = xn + ((mom - 1.0) / mn) * (xn - x)
            mom = mn
        else:
            y = xn
        x = xn
        if errs is not None:
            errs.append(np.linalg.norm(x - truth[None, :], axis=1) / np.linalg.norm(truth))
        if k in cps:
            out[k] = x.copy()
    if 0 in cps:
        out[0] = np.zeros_like(b) if x0 is None else np.array(x0, dtype=float).reshape(b.shape)
    return np.array([out[c] for c in [int(c) for c in checkpoints]]), (None if errs is None else np.array(errs))


def fixed_point_residual(a: np.ndarray, b: np.ndarray, den, x: np.ndarray) -> np.ndarray:
    """max |x - D(x - t A^T (A x - b))| je Zeile: null genau am Fixpunkt."""
    b = np.atleast_2d(b)
    x = np.atleast_2d(x)
    return np.abs(x - pnp_step(a, b, x, den, step_size(a))).max(axis=1)


def lipschitz_estimate(name: str, n: int, strength: float, center: np.ndarray, seed: int) -> float:
    """Größter Quotient |D(u) - D(v)| / |u - v| über C.LIP_TRIALS Paare (je ein Drittel pro Abstand in C.LIP_SCALES); u = center + Rauschen, v = u + Abstand mal Rauschen. Eine Untergrenze der
    Lipschitz-Konstante: über 1 heißt sicher nicht nichtexpansiv, 1 heißt nur „in der Stichprobe nichtexpansiv“."""
    rng = SplitMix64(seed)
    per = C.LIP_TRIALS // len(C.LIP_SCALES)
    u = np.asarray(center, dtype=float)[None, :] + C.LIP_CENTER_NOISE * rng.normal(per * len(C.LIP_SCALES) * n).reshape(-1, n)
    dv = rng.normal(per * len(C.LIP_SCALES) * n).reshape(-1, n) * np.repeat(C.LIP_SCALES, per)[:, None]
    den = make_denoiser(name, n, np.full(len(u), strength))
    num = np.linalg.norm(den(u) - den(u + dv), axis=1)
    return float((num / np.linalg.norm(dv, axis=1)).max())


# ----------------------------------------------------------------------------- Vergleichslöser

def tikhonov_svd(a: np.ndarray, b: np.ndarray, lams) -> np.ndarray:
    """Tikhonov-Lösungen x = V diag(s / (s^2 + lam^2)) U^T b für Blöcke b (m, n) und alle lam: Form (len(lams), m, n)."""
    u, sv, vt = np.linalg.svd(a)
    b = np.atleast_2d(b)
    return np.array([((b @ u) * (sv / (sv ** 2 + lam ** 2))) @ vt for lam in lams])


def lasso(m: np.ndarray, y: np.ndarray, lam: float, maxit: int = C.LASSO_MAXIT, tol: float = 1e-10) -> np.ndarray:
    """Exakte Lösung von min 1/2 |m w - y|^2 + lam |w|_1 per Aktivmenge (wie in Stück 6 und 12)."""
    p = m.shape[1]
    mtm, mty = m.T @ m, m.T @ y
    w = np.zeros(p)
    active = np.zeros(p, dtype=bool)
    sgn = np.zeros(p)
    for _ in range(maxit):
        idx = np.flatnonzero(active)
        if idx.size:
            s = sgn[idx]
            sol = np.linalg.solve(mtm[idx][:, idx], mty[idx] - lam * s)
            bad = sol * s < 0.0
            if bad.any():
                w_old = w[idx]
                with np.errstate(divide="ignore", invalid="ignore"):
                    t = np.where(bad, w_old / (w_old - sol), np.inf)
                j = int(np.argmin(t))
                w[idx[j]] = 0.0
                active[idx[j]] = False
                continue
            w[idx] = sol
        g = mtm @ w - mty
        viol = np.where(active, 0.0, np.abs(g) - lam)
        j = int(np.argmax(viol))
        if viol[j] <= tol * max(lam, _TINY):
            return w
        active[j] = True
        sgn[j] = -np.sign(g[j])
        w[j] = 0.0
    return w


def objective_l1(a: np.ndarray, b: np.ndarray, x: np.ndarray, lam: float) -> float:
    """1/2 |A x - b|^2 + lam |x|_1."""
    return 0.5 * float(np.sum((a @ x - b) ** 2)) + lam * float(np.abs(x).sum())


def objective_haar(a: np.ndarray, b: np.ndarray, x: np.ndarray, lam: float) -> float:
    """1/2 |A x - b|^2 + lam |W_d x|_1: nur die Detailkoeffizienten der orthonormalen Haar-Zerlegung sind bestraft (die Zielfunktion von PnP mit dem Haar-Denoiser, lam = theta / t)."""
    n = len(x)
    c = haar_matrix(n, C.HAAR_LEVELS) @ x
    return 0.5 * float(np.sum((a @ x - b) ** 2)) + lam * float(np.abs(c[n_approx(n, C.HAAR_LEVELS):]).sum())


def wellposed_matrix(n: int, kernel=C.WP_KERNEL) -> np.ndarray:
    """Gut gestellte Faltungsmatrix (Kern 0,2 / 0,6 / 0,2, Nullränder): die Kondition ist klein, ISTA konvergiert schnell auch im x."""
    k = (len(kernel) - 1) // 2
    a = np.zeros((n, n))
    for d in range(-k, k + 1):
        idx = np.arange(max(d, 0), min(n, n + d))
        a[idx, idx - d] = kernel[d + k]
    return a


def gauss_closed_form(a: np.ndarray, b: np.ndarray, s: float, p: np.ndarray, t: float) -> np.ndarray:
    """Fixpunkt von PnP mit dem Gauß-MMSE-Denoiser: x = (s^2 / t P^-1 + A^T A)^-1 A^T b."""
    return np.linalg.solve(s * s / t * np.linalg.inv(p) + a.T @ a, a.T @ b)


def posterior_mean(a: np.ndarray, b: np.ndarray, sigma: float, p: np.ndarray) -> np.ndarray:
    """P A^T (A P A^T + sigma^2 I)^-1 b (Stück 7)."""
    return p @ a.T @ np.linalg.solve(a @ p @ a.T + sigma ** 2 * np.eye(len(b)), b)


@lru_cache(maxsize=4)
def _op(n: int, w: float) -> np.ndarray:
    a = operator(n, w)
    a.flags.writeable = False
    return a


def op(n: int = C.N, w: float = C.W) -> np.ndarray:
    """Der Operator des Generators, zwischengespeichert."""
    return _op(int(n), round(float(w), 6))
