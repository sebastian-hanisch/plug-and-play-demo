"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini():
    """3 x 3-Matrix aus Stück 1 bis 12 (symmetrisch, Zeilensumme 1, Eigenwerte 1, 0,5, 0,1 zu den Vektoren (1,1,1), (1,0,-1), (1,-2,1)), b = (1,3, 1,4, 1,3) = (4/3)(1,1,1) - (1/30)(1,-2,1),
    also ohne Anteil von (1,0,-1). |A|² = 1, die Schrittweite ist t = 1. Handrechnung:
    - ein Schritt aus 0 mit Weichschwelle 0,1: A^T b = (1,33, 1,34, 1,33), nach der Schwelle (1,23, 1,24, 1,23);
    - Landweber (Denoiser = Identität) nach k Schritten: (4/3)(1,1,1) + 10 (1 - 0,99^k)(-1/30)(1,-2,1): k = 1 (1,33, 1,34, 1,33), k = 2 (1,3267, 1,3466, 1,3267);
    - Fixpunkt des Gauß-MMSE-Denoisers mit P = I und s = 0,1: (s² I + A²) x = A b, also x = (1,153465, 1,653465, 1,153465) (Komponenten (4/3)/1,01 = 1,320132 und 0,1 (-1/30)/(0,01 + 0,01) = -0,166667);
      das ist das Posterior-Mittel (Stück 7) und die Tikhonov-Lösung mit lambda = 0,1."""
    a = np.array([[0.6, 0.3, 0.1], [0.3, 0.4, 0.3], [0.1, 0.3, 0.6]])
    b = np.array([1.3, 1.4, 1.3])
    return a, b
