# Plug-and-Play: Der Denoiser ist der Prior – Streamlit-Demo

Dreizehntes Stück der Linie **Inverse Probleme und Regularisierung** in der „Konzepte“-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine Learning). Es baut auf [sparsity-tv-demo](https://github.com/sebastian-hanisch/sparsity-tv-demo), [bayes-map-demo](https://github.com/sebastian-hanisch/bayes-map-demo), [iterativ-demo](https://github.com/sebastian-hanisch/iterativ-demo), [tikhonov-demo](https://github.com/sebastian-hanisch/tikhonov-demo) und [blinde-entfaltung-demo](https://github.com/sebastian-hanisch/blinde-entfaltung-demo) auf und nutzt dieselbe Welt: eine lineare Entfaltung (n = 48) mit Gauß-Unschärfe (w = 0,03) und weißem Rauschen. Die Wahrheit ist bekannt, weil das Signal selbst die Wahrheit ist; sie dient hier nur als Maßstab.

## Kernfrage

Was passiert, wenn man im Schritt von ISTA den Prox durch einen **Denoiser** ersetzt? Wann **hilft** er (nur wenn er zur Welt passt), wie stehen **Stärke und Schrittzahl** zueinander (der Denoiser ersetzt den Abbruch nicht), und was bleibt, wenn der Denoiser **kein Prox** ist (keine Zielfunktion, keine Garantie)?

## Modell

Generator `inverse_generator.py`, wortgleich in allen Stücken der Linie (Phantom, Gitter, Operator, Rauschen):

- **Daten:** b = A x + e mit der Gauß-Faltungsmatrix A (n = 48, w = 0,03), e ~ N(0, σ² I), Rauschen 0,1 / 1 / 10 % des Effektivwerts, Daten ohne Gitterfehler (das Modell stimmt).
- **Vier Welten:** Phantom (Hügel und Kasten), **drei Spitzen**, **drei Blöcke** (wie Stück 6) und eine **Dreieckswelle** mit vier Perioden (neu: wiederholte Struktur).
- **Plug-and-Play-ISTA:** x ← D(x − t Aᵀ(A x − b)), t = 1/‖A‖₂² (≈ 1,01), Start 0; wahlweise mit FISTA-Impuls (Stück 6).
- **Fünf Denoiser:** **Weichschwelle** (Prox von L1), **Haar-Wavelet-Weichschwelle** (orthonormal, vier Ebenen, Prox von θ‖W_d x‖₁), **Haar über 16 Verschiebungen gemittelt** (kein Prox), **nicht-lokale Mittel** in 1D (Fenster 5, kein Prox) und **Gauß-MMSE** D(v) = P(P + s² I)⁻¹v mit dem exponentiellen Prior aus Stück 8 (linear).
- **Stärke:** je Denoiser ein Gitter von 13 Werten (halbe Dekaden); in der App als Faktor ÷10 bis ×10 zur besten Stärke des Gitters (Orakel mit der Wahrheit, wie in Stück 4).
- **Vergleich:** Tikhonov (SVD) und L1 exakt (Aktivmenge, Stück 6), jeweils beim besten λ des Gitters.

Alles ist erfunden; die Wahrheit ist das simulierte Signal.

## Methodik

- **Verdikt:** „hilft“, wenn der Fehler höchstens 80 % von Tikhonov beträgt; „gleichwertig“ bis 100 %; sonst „passt nicht zur Welt“. Dazu Hinweise: beste Stärke am Rand des Gitters (der Denoiser trägt nichts bei), der Fehler war früher am kleinsten (der Abbruch regularisiert mit), das exakte L1 ist besser (bei der Weichschwelle), Fixpunktrest über 10⁻³, Lipschitz-Schätzung über 1.
- **Experimente** laufen über feste Zufallsströme (SplitMix64, Seeds ab 1000); nie ein einzelner Seed: 6 Ströme (4 bei den nicht-lokalen Mitteln), jeweils der Median beim besten gemeinsamen Parameter.
- **Rechenaufwand:** eine Instanz rund 1 s, die vier Experimente zusammen rund eine Minute beim ersten Aufruf (danach zwischengespeichert).
- **Gleichstände und letzte Stellen** werden nicht festgelegt; die Tests prüfen Schranken, Mediane und enge Toleranzen (0,002 für stromabhängige Zahlen).

## Befunde (gemessen, keine Behauptungen)

Instanz n = 48, w = 0,03, Rauschen 1 %, 300 Schritte, wenn nicht anders genannt. Jede Zahl steht in `tests/test_claims.py` oder `tests/test_presets.py`.

| Frage | Befund | Test |
|---|---|---|
| Ist Plug-and-Play mit Prox exakt Stück 6 (Experiment 1)? | **Ja, in der Zielfunktion:** auf einer gut gestellten Matrix (Kondition 5,0) liegt die Weichschwelle nach 500 Schritten unter 10⁻⁹ vom exakten L1-Löser. Auf dem schlecht gestellten Operator fällt die Zielfunktionslücke (1,5·10⁻⁴ / 1,9·10⁻⁵ / 6,9·10⁻⁶ nach 100 / 1000 / 10000 Schritten), aber der **Abstand im x bleibt 0,63 bis 0,65**. Haar: die Zielfunktion fällt in 60 Schritten monoton. | `test_prox_is_ista_…`, `test_haar_objective_is_monotone` |
| Ist der Gauß-MMSE-Fixpunkt exakt Stück 7? | **Ja, bei s = √t σ:** 2,9·10⁻⁶ vom Posterior-Mittel (Fehler 13,1 %). Bei s = 3 √t σ ist er die geschlossene Form (unter 10⁻⁹), aber **nicht** das Posterior-Mittel (Abstand 0,065, Fehler 10,4 % statt 13,1 %): ein anderes λ. | `test_gauss_mmse_fixed_point_…` |
| Hilft ein Denoiser überall (Experiment 2)? | **Nein, er gehört zur Welt:** **Spitzen** (Rauschen 1 %) nur die Weichschwelle **8,7 %** (Tikhonov 54,3 %, exaktes L1 0,9 %), Haar, Haar gemittelt, nicht-lokale Mittel und Gauß-MMSE liegen bei **61,3 bis 64,9 %**; **Blöcke** Haar gemittelt **2,2 %**, Haar ohne Mittelung **17,2 %**, Tikhonov 18,9 %; **Wellen** bei 10 % Rauschen nicht-lokale Mittel **9,1 %**, Gauß-MMSE 10,6 %, Tikhonov 16,2 % (bei 1 %: 3,2 gegen 4,6 %). | `test_the_denoiser_decides_about_the_world` |
| Und das Phantom? | Weichschwelle 6,1 %, Haar gemittelt 7,8 %, Tikhonov 11,0 %, **exaktes L1 52,3 %**: derselbe Prox wie L1, aber nach 300 Schritten statt am Fixpunkt (Stück 6 hat das schon gesehen). | `test_phantom_numbers_…` |
| Ersetzt der Denoiser den Abbruch (Experiment 3)? | **Nein.** Bei schwacher Stärke liegen alle fünf Denoiser bis 100 Schritte **übereinander** (13,4 / 12,4 / 11,2 % nach 10 / 30 / 100 Schritten, Spannweite 0,03 %): es regularisiert die Schrittzahl (Stück 5). Mit bester Stärke je Schrittzahl wird die Weichschwelle beim Phantom von **6,1 % (300 Schritte) auf 9,3 % (3000 Schritte)** schlechter, bei den Wellen von 4,5 auf **10,5 %**; nicht-lokale Mittel bei den Wellen 3,7 / 3,2 / 9,2 %; nur der **Gauß-MMSE-Denoiser bleibt stabil** (4,4 % bei 100, 300 und 3000 Schritten). Haar gemittelt bei Blöcken: 8,3 / 2,2 / 2,1 %. | `test_weak_strength_…`, `test_the_denoiser_does_not_replace_early_stopping` |
| Hilft der FISTA-Impuls? | **Nein, er schadet:** Weichschwelle 6,1 → **18,5 %**, Haar 10,7 → 17,6 %, Haar gemittelt 7,8 → 15,4 %, nicht-lokale Mittel 9,9 → 25,2 %, Gauß-MMSE 10,9 → 23,6 %: der schnelle Weg zum Fixpunkt ist der schlechtere. | `test_momentum_makes_every_denoiser_worse` |
| Gibt es eine Garantie ohne Prox (Experiment 4)? | **Nein, gemessen wurde keine Divergenz:** Lipschitz-Schätzung (300 Paare) bei Weichschwelle, Haar, Haar gemittelt und Gauß-MMSE höchstens 1,000; **nicht-lokale Mittel 1,000 / 1,027 / 1,076** bei h = 0,01 / 0,03 / 0,10, also über 1; trotzdem Fixpunktrest nach 3000 Schritten höchstens 3,4·10⁻⁴. Haar ohne Mittelung verfehlt die Verschiebungsgleichheit um 1,7·10⁻³, Haar gemittelt um unter 10⁻⁹. | `test_lipschitz_and_convergence_numbers` |
| Einzelstichproben (Presets) | **Standard** (Blöcke, Haar gemittelt): 2,2 % gegen 17,7 % (Tikhonov). **Falscher Denoiser** (Spitzen): 65,0 % gegen 53,4 %, beste Stärke am Rand. **Passender Denoiser** (Spitzen, Weichschwelle): 8,7 %. **Haar ohne Mittelung**: 16,7 % gegen 17,7 %. **Wellen, 10 %, nicht-lokale Mittel:** 10,3 % gegen 17,0 %. **Bis zum Fixpunkt** (Phantom, 3000 Schritte): 7,4 %, nach etwa 700 Schritten 5,3 %. **FISTA-Impuls:** 12,9 % statt 6,1 %. | `tests/test_presets.py` |
| Mini-Instanz von Hand | n = 3, A symmetrisch (Eigenwerte 1 / 0,5 / 0,1), b = (1,3, 1,4, 1,3): ein Schritt mit Weichschwelle 0,1 gibt (1,23, 1,24, 1,23); Landweber nach 1 und 2 Schritten (1,33, 1,34, 1,33) und (1,3267, 1,3466, 1,3267); Gauß-MMSE-Fixpunkt (1,153465, 1,653465, 1,153465) = Posterior-Mittel = Tikhonov bei λ = 0,1. | `tests/test_model.py` |

## Befunde und Korrekturen gegenüber dem Plan

- **„MAP-artig“ stimmt nur für Prox-Denoiser.** Der Weichschwellen-Prox ist MAP (Stück 6); der Gauß-MMSE-Denoiser ist Stück 7 (dort ist MAP = Mittel). Ein Plug-and-Play mit nicht-lokalen Mitteln ist keines von beiden: kein Prox, keine Zielfunktion, Lipschitz bis 1,08.
- **Ein besserer Denoiser macht kein besseres Ergebnis.** Bei Spitzen sind die „guten“ Denoiser (Wavelet, nicht-lokale Mittel) siebenmal schlechter als die Weichschwelle; bei Blöcken entscheidet die Mittelung über Verschiebungen (17,2 gegen 2,2 %).
- **Plug-and-Play löst die Parameterwahl nicht.** Es hat zwei gekoppelte Parameter, Stärke und Schrittzahl; die beste Stärke liegt oft am Rand des Gitters (kein Denoiser, nur Abbruch).
- **Das Orakel „ISTA mit L1“ trägt in der Zielfunktion, nicht im x:** auf dem schlecht gestellten Operator bleibt der Abstand im x nach 10000 Schritten über 0,6.
- **Nicht-lokale Mittel brauchen ihre Wiederholung:** bei den Wellen gewinnen sie (36 % besser als Tikhonov bei 10 % Rauschen), beim Phantom und bei Blöcken nicht.

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Ein besserer Denoiser hilft überall** | Bei Spitzen (Rauschen 1 %) hilft nur die Weichschwelle (8,7 %); die anderen vier liegen bei 61,3 bis 64,9 %, Tikhonov bei 54,3 %. | Stück 16 (Verfahrensvergleich) |
| **Der Denoiser ersetzt den Abbruch** | Bis zum Fixpunkt gerechnet wird die Weichschwelle schlechter; Stärke und Schrittzahl sind zwei gekoppelte Parameter. | entrollte Netze (Stück 14) |
| **Es gibt eine Garantie** | Die nicht-lokalen Mittel sind kein Prox und bei großer Stärke nicht nichtexpansiv; gemessen wurde keine Divergenz, aber bewiesen ist nichts. | – |
| **Ein Operator, eine Dimension** | Toeplitz-Faltung mit w = 0,03, n = 48, ein Gitter von 13 Stärken, wenige Ströme. | – |
| **Klassische Denoiser** | Kein trainiertes Netz, kein Rauschniveau pro Schritt, kein PnP-ADMM; ein Score als Denoiser ist Stück 15. | Stück 15 |
| **Rauschen weiß, Daten b = A x + e** | Das Modell stimmt (kein Gitterfehler); ehrliche Daten sind nicht gemessen. | – |

Die Zahlen gelten für diese erfundene Welt und zeigen nichts über echte Messungen.

## Bezug zu den Nachbarn im Portfolio

- [sparsity-tv-demo](https://github.com/sebastian-hanisch/sparsity-tv-demo): Stück 6; ISTA, FISTA, Weichschwelle und der exakte L1-Löser sind das Orakel der Prox-Denoiser.
- [bayes-map-demo](https://github.com/sebastian-hanisch/bayes-map-demo): Stück 7; das Posterior-Mittel ist das Orakel des Gauß-MMSE-Denoisers.
- [iterativ-demo](https://github.com/sebastian-hanisch/iterativ-demo): Stück 5; die Schrittzahl als Regularisierung (Landweber) kehrt in Experiment 3 zurück.
- [blinde-entfaltung-demo](https://github.com/sebastian-hanisch/blinde-entfaltung-demo): Stück 12; dort fehlte die Annahme über x, die hier der Denoiser liefert.

## Tests

Mini-Instanz von Hand (Weichschwelle, Haar für n = 4, Landweber, Gauß-MMSE-Fixpunkt, Schrittweite, ein Schritt, Impuls, Fixpunktrest, Lipschitz-Grenzfälle, Tikhonov, Lasso, Zielfunktionen), Orakel (Weichschwelle gegen den exakten Löser auf der gut gestellten Matrix, Zielfunktionslücke auf dem schlecht gestellten, KKT von L1, Orthonormalität und Prox-Bedingungen von Haar, Haar-Zielfunktion fällt in jedem Schritt, Gauß-MMSE gegen geschlossene Form und Posterior-Mittel, Lipschitz der Denoiser, Verschiebungsgleichheit der Mittelung, Landweber in SVD-Form, Impuls schneller, Welten gegen Handwerte), Generator, Auswertung (jede Schwelle des Urteils auf der Kante), jede README-Zahl in `test_claims.py`, AppTest (Presets, Regler-Grenzen, jede Welt und jeder Denoiser, PDF, Permalink), Footer, Presets und Laufzeit-Importe.

```bash
python -m pytest tests -q
```

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `inverse_generator.py` | gemeinsamer Generator der Linie (SplitMix64, Phantom, Gitter, Operator) |
| `pnp_model.py` | Denoiser, Plug-and-Play-ISTA, Tikhonov, L1-Aktivmenge, Lipschitz-Schätzung, Fixpunktrest |
| `pnp_evaluation.py` | Welt, Instanz, Analyse, Experimente, Verdikte |
| `pnp_visualization.py`, `pnp_pdf_export.py` | Figuren, PDF |
| `pnp_constants.py`, `pnp_presets.py` | Regler, Presets, Permalink |
| `tests/` | Mini-Instanz, Orakel, Claims, App |

## Bewusst nicht umgesetzt

- **TV als Denoiser:** exakter Prox, aber das ist Stück 6 und hier teuer (30 ms je Prox).
- **PnP-ADMM und andere Aufteilungen:** nur ISTA.
- **Trainierte Denoiser, Rauschniveau pro Schritt:** Stück 14 und 15.
- **Mehrere Operatoren, 2D-Bilder:** ein Operator, eine Dimension.
- **Ehrliche Daten (analytisch):** b = A x + e, das Modell stimmt.
- **Praxisdaten:** nicht gebaut; alle Zahlen sind erzeugt.

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python 3.12.

Gebaut mit Streamlit, Plotly, numpy, pandas und fpdf2.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Inverse Probleme: vom Rauschen zur Verfahrenswahl](https://sebastianhanisch.net/konzepte-inverse-probleme.html).
