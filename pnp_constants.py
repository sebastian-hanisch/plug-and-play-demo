"""Konstanten: Regler, feste Annahmen, Experimente, Farben und Presets."""

# Welt: die lineare Entfaltung aus Stück 1 bis 8 (n = 48 Gitterpunkte, Gauß-Unschärfe w = 0,03), Daten b = A x + e (das Modell stimmt).
N = 48
W = 0.03                                                         # Breite der Unschärfe (Anteil der Länge 1), fest
NOISE_CHOICES = (1e-3, 1e-2, 1e-1)                               # Rauschen relativ zum Effektivwert der exakten Daten
DEFAULT_NOISE = 1e-2
WORLDS = ("phantom", "spikes", "blocks", "waves")
WORLD_LABELS = {"phantom": "Phantom (zwei Hügel und ein Kasten)", "spikes": "drei Spitzen", "blocks": "drei Blöcke", "waves": "Dreieckswelle (vier Perioden)"}
WORLD_SHORT = {"phantom": "Phantom", "spikes": "Spitzen", "blocks": "Blöcke", "waves": "Wellen"}
SPIKES = ((0.25, 0.04), (0.5, 0.03), (0.7, 0.05))                # Spitzen: (Ort als Anteil der Länge, Masse); die Höhe in der Zelle ist Masse mal n
BLOCKS = ((0.15, 0.30, 1.0), (0.42, 0.52, 0.6), (0.70, 0.85, 0.9))   # Blöcke: (links, rechts, Höhe) wie in Stück 6
WAVE_PERIODS, WAVE_RANGE = 4, (0.1, 0.9)                         # Dreieckswelle mit vier Perioden zwischen 0,1 und 0,9
DENOISERS = ("soft", "haar", "haar_avg", "nlm", "gauss")
DENOISER_LABELS = {"soft": "Weichschwelle (Prox von L1)", "haar": "Haar-Wavelet-Weichschwelle (Prox)", "haar_avg": "Haar über 16 Verschiebungen gemittelt (kein Prox)",
                   "nlm": "Nicht-lokale Mittel (kein Prox)", "gauss": "Gauß-MMSE (linear, Prior aus Stück 8)"}
DENOISER_SHORT = {"soft": "Weichschwelle", "haar": "Haar", "haar_avg": "Haar gemittelt", "nlm": "nicht-lokale Mittel", "gauss": "Gauß-MMSE"}
DENOISER_UNIT = {"soft": "Schwelle θ", "haar": "Schwelle θ", "haar_avg": "Schwelle θ", "nlm": "Gewichtsbreite h", "gauss": "Rauschniveau s"}
FACTOR_CHOICES = (0.1, 0.3, 1.0, 3.0, 10.0)                      # Stärke = Faktor mal beste Stärke des Gitters
FACTOR_LABELS = {0.1: "÷10", 0.3: "÷3", 1.0: "×1 (beste Stärke)", 3.0: "×3", 10.0: "×10"}
DEFAULT_FACTOR = 1.0
STEPS_CHOICES = (30, 100, 300, 1000, 3000)
DEFAULT_STEPS = 300
SEED_MAX, DEFAULT_SEED = 999999, 7

# feste Modellannahmen
PRIOR_TAU, PRIOR_ELL = 0.5, 0.12                                 # Prior des Gauß-MMSE-Denoisers wie in Stück 8
HAAR_LEVELS = 4                                                  # n = 48 = 3 mal 2^4
HAAR_SHIFTS = 16                                                 # Verschiebungen der Mittelung: 2^4, damit die Mittelung für jede zyklische Verschiebung exakt gleichartig ist
NLM_HALF = 2                                                     # Fenster der nicht-lokalen Mittel: 2 * 2 + 1 = 5 Punkte
GRID_SIZE = 13                                                   # Gitter der Stärken je Denoiser (halbe Dekade)
LAMBDA_GRID = tuple(10.0 ** (-6.0 + 0.5 * k) for k in range(GRID_SIZE))      # λ = θ / t für Weichschwelle und Haar: 1e-6 bis 1
NLM_GRID = tuple(10.0 ** (-3.0 + 3.5 * k / (GRID_SIZE - 1)) for k in range(GRID_SIZE))   # h: 1e-3 bis 10^0,5
GAUSS_GRID = tuple(10.0 ** (-5.0 + 5.0 * k / (GRID_SIZE - 1)) for k in range(GRID_SIZE))  # s: 1e-5 bis 1
TIK_GRID = tuple(10.0 ** (-6.0 + 0.5 * k) for k in range(GRID_SIZE))         # Tikhonov-λ für die Vergleichslösung
L1_GRID = tuple(10.0 ** (-6.0 + 0.5 * k) for k in range(GRID_SIZE))          # L1-λ für die Vergleichslösung (exakter Löser)
LASSO_MAXIT = 5000
LIP_TRIALS = 300                                                 # Paare der Lipschitz-Schätzung (je Abstand ein Drittel)
LIP_SCALES = (1e-3, 1e-1, 1.0)                                   # Abstände der Paare
LIP_CENTER_NOISE = 0.3                                           # Streuung der Mittelpunkte der Paare um die Wahrheit

# Verdikt-Schwellen
GAIN = 0.8                                                       # der Denoiser „hilft“: Fehler höchstens 80 % des Fehlers von Tikhonov
TIE = 1.0                                                        # „gleichwertig“: Fehler höchstens so groß wie bei Tikhonov
EARLY = 0.95                                                     # der Fehlerverlauf hat sein Minimum früher: kleinster Fehler höchstens 95 % des Fehlers am Ende
RES_OK = 1e-3                                                    # Fixpunktrest (max |x − D(x − t Aᵀ(A x − b))|), ab dem die Iteration als nicht am Fixpunkt gilt
LIP_ONE = 1.001                                                  # Lipschitz-Schätzung über 1 (auf drei Stellen sichtbar): der Denoiser ist (nach Stichprobe) nicht nichtexpansiv

# Experimente: feste Ströme (Seeds aus SEED_BASE), nie ein einzelner Seed als Befund.
SEED_BASE = 1000
EXP_REPS = 6                                                     # Ströme je Zelle
NLM_REPS = 4                                                     # Ströme je Zelle bei den nicht-lokalen Mitteln (teuer)
WORLD_NOISES = (1e-2, 1e-1)                                      # Experiment 2
WORLD_STEPS = 300
WP_KERNEL = (0.2, 0.6, 0.2)                                      # gut gestellte Vergleichsmatrix (Experiment 1)
WP_LAMBDAS = (1e-3, 1e-2, 1e-1)
WP_STEPS = 500
PROX_LAMBDA, PROX_CHECKS = 1e-3, (100, 1000, 10000)              # Experiment 1: Zielfunktionslücke von ISTA gegen den exakten Löser
GAUSS_FACTORS = (0.3, 1.0, 3.0)                                  # Experiment 1: s = Faktor mal sqrt(t) mal sigma
GAUSS_STEPS = 20000
HAAR_STEPS = 60
WEAK_NOISE, WEAK_CHECKS = 1e-3, (10, 30, 100, 300, 1000, 3000)  # Experiment 3, Teil 1: schwache Stärke
WEAK_STRENGTH = {"soft": 1e-5, "haar": 1e-5, "haar_avg": 1e-5, "nlm": 3e-3, "gauss": 1e-4}   # θ = t λ beziehungsweise h, s
STEP_WORLDS = ("phantom", "blocks", "waves")                     # Experiment 3, Teil 2
STEP_CHECKS = (100, 300, 3000)
STEP_NOISE = 1e-2
CONV_STEPS = 3000                                                # Experiment 4
CONV_STRENGTHS = {"soft": (1e-3,), "haar": (1e-3,), "haar_avg": (1e-3, 1e-2), "nlm": (0.01, 0.03, 0.1), "gauss": (1e-2,)}   # λ beziehungsweise h, s
CONV_NOISE = 1e-2

COLOR_TRUTH = "#2F6B65"
COLOR_PNP = "#4c78a8"
COLOR_TIK = "#D68A2E"
COLOR_L1 = "#c0392b"
COLOR_GREY = "#8a8a8a"
COLOR_REF = "#222222"
SHOW_RANGE = {"phantom": (-0.4, 1.9), "spikes": (-0.6, 3.0), "blocks": (-0.4, 1.9), "waves": (-0.4, 1.9)}   # feste Achse der Rekonstruktion je Welt

_BASE = {"world": "blocks", "noise": DEFAULT_NOISE, "denoiser": "haar_avg", "factor": DEFAULT_FACTOR, "steps": DEFAULT_STEPS, "accel": False, "seed": DEFAULT_SEED}
PRESETS = {
    "Standard": dict(_BASE),
    "Falscher Denoiser": dict(_BASE, world="spikes"),
    "Passender Denoiser": dict(_BASE, world="spikes", denoiser="soft"),
    "Haar ohne Mittelung": dict(_BASE, denoiser="haar"),
    "Wellen, viel Rauschen": dict(_BASE, world="waves", noise=1e-1, denoiser="nlm"),
    "Bis zum Fixpunkt": dict(_BASE, world="phantom", denoiser="soft", steps=3000),
    "Mit Beschleunigung": dict(_BASE, world="phantom", denoiser="soft", accel=True),
}
PRESET_HELP = {
    "Standard": "Blöcke, Rauschen 1 %, Haar über 16 Verschiebungen gemittelt: 2,2 % Fehler gegen 17,7 % bei Tikhonov; der Denoiser passt zur Welt.",
    "Falscher Denoiser": "Derselbe Denoiser bei drei Spitzen: 65,0 % Fehler, schlechter als Tikhonov (53,4 %); die beste Stärke liegt am Rand des Gitters, der Denoiser trägt nichts bei. Das exakte L1 hat 1,4 %.",
    "Passender Denoiser": "Spitzen mit der Weichschwelle (Prox von L1): 8,7 % gegen 53,4 % bei Tikhonov; das exakte L1 (der Grenzwert bei unendlich vielen Schritten) hat 1,4 %.",
    "Haar ohne Mittelung": "Blöcke mit dem Haar-Denoiser ohne Mittelung: 16,7 % gegen 17,7 % bei Tikhonov, gleichwertig; erst die Mittelung über Verschiebungen bringt die 2,2 %.",
    "Wellen, viel Rauschen": "Dreieckswelle, Rauschen 10 %, nicht-lokale Mittel: 10,3 % gegen 17,0 % bei Tikhonov; sie nutzen die Wiederholung. Die Lipschitz-Schätzung liegt über 1 (1,10): keine Konvergenzgarantie.",
    "Bis zum Fixpunkt": "Phantom, Weichschwelle, 3000 Schritte: 7,4 % am Ende, nach etwa 700 Schritten waren es 5,3 %; der Denoiser ersetzt den Abbruch nicht.",
    "Mit Beschleunigung": "Phantom, Weichschwelle, FISTA-Impuls, 300 Schritte: 12,9 % statt 6,1 % ohne Impuls (der kleinste Fehler lag bei Schritt 61): der schnelle Weg zum Fixpunkt ist schlechter.",
}
