"""Konstanten: Größen, Raster, Familien der Flussmatrix, feste Reihen der Experimente, Farben und Presets."""

EXACT_SIZES = (6, 8, 9)                               # alle n! Anordnungen werden aufgezählt: das Optimum ist bekannt; n = 9 nur auf Knopfdruck
LARGE_SIZES = (12, 16, 20)                            # nicht aufgezählt: die beste gefundene Lösung ersetzt das Optimum
SIZES = EXACT_SIZES + LARGE_SIZES
DEFAULT_N = 8
SWEEP_SIZES = (6, 8)                                  # Vergleiche mit aufgezähltem Optimum
LARGE_SWEEP_SIZES = (12, 16)                          # Vergleiche gegen die beste gefundene Lösung
GRID = {6: (2, 3), 8: (2, 4), 9: (3, 3), 12: (3, 4), 16: (4, 4), 20: (4, 5)}    # Plätze als Raster (Zeilen, Spalten), Abstand = Manhattan
SEED_MAX, DEFAULT_SEED = 999999, 35
SWEEP_SEEDS = tuple(range(100000, 100020))            # feste Instanzen: nie ein einzelner Seed als Befund
SWEEP_SEEDS_N = {6: SWEEP_SEEDS, 8: SWEEP_SEEDS[:10]}  # n = 8: Aufzählung und 50 Starts kosten je Instanz einige Sekunden, darum nur zehn
LARGE_SEEDS = SWEEP_SEEDS[:10]                        # große Instanzen: zehn feste Instanzen je Familie
STARTS = 200                                          # Zahl der steilsten Abstiege aus Zufallsstarts für die beste gefundene Lösung
FW_ITERS = 100                                        # höchstens so viele Frank–Wolfe-Schritte je Lauf der FAQ-Art
CONVEX_ITERS = 300                                    # Frank–Wolfe-Schritte für die konvexe Relaxation (jeder Schritt liefert eine gültige untere Schranke)
LAMBDA = 0.6                                          # Zufallsstart: λ · (zufällige Permutationsmatrix) + (1 − λ) · J/n
STARTS_SWEEP = 50                                     # Zufallsstarts je Instanz in den Vergleichen n = 6, 8
STARTS_LARGE = 20                                     # Zufallsstarts je Instanz bei n ≥ 12
STARTS_LIVE = 50                                      # Zufallsstarts der gewählten Instanz bis n = 9 (darüber 20)

FAMILIES = ("netlist", "uniform", "sparse", "cluster", "star", "skewed")
DEFAULT_FAMILY = "netlist"
FAMILY_NAMES = {"netlist": "Netzliste", "uniform": "Gleichverteilt", "sparse": "Dünn besetzt", "cluster": "Zwei Cluster", "star": "Stern", "skewed": "Wenige starke Flüsse"}
FAMILY_HELP = {
    "netlist": "Netzliste aus Dreiergruppen (IC, Kondensator, Widerstand), jedes Paar in einem Netz zählt Fluss 1: die Familie aus Stück 1.",
    "uniform": "Jedes Paar von Bauteilen hat einen zufälligen Fluss von 0 bis 9, gleichverteilt: dichte Matrix ohne Struktur.",
    "sparse": "Nur drei von zehn Paaren haben Fluss (1 bis 9), die übrigen keinen: dünn besetzt ohne weitere Struktur.",
    "cluster": "Zwei Gruppen von Bauteilen: innerhalb einer Gruppe starker Fluss (8 bis 10), zwischen den Gruppen schwacher (0 bis 2).",
    "star": "Ein Bauteil (die Nabe) hat starken Fluss (1 bis 9) zu allen anderen; zwischen den übrigen höchstens 1.",
    "skewed": "Die Flüsse sind schief verteilt: viele Paare ohne oder mit kleinem Fluss, wenige mit sehr großem (bis 60).",
}

# Netzliste (wie in Stück 1 bis 5)
N_NETS_PER_COMPONENT = 1.5
NET_SIZES = (2, 3, 4)
NET_WEIGHTS = (5, 3, 2)
LOCAL_NET_PERCENT = 70
GROUP = 3
# Dünn besetzt, Cluster, Stern, schief (wie in Stück 2)
SPARSE_PERCENT = 30
UNIFORM_MAX = 9
CLUSTER_STRONG, CLUSTER_WEAK_MAX = 8, 2
SKEW_VALUES = (0, 1, 2, 4, 8, 16, 32, 60)
SKEW_WEIGHTS = (45, 22, 13, 8, 6, 3, 2, 1)           # ganzzahlige Gewichte (Summe 100), schweres Ende

HAND_FLOWS = ((0, 1, 5), (1, 2, 1))                   # Handbeispiel 1: drei Bauteile auf einer Zeile, Fluss A–B 5, B–C 1
HAND_GRID = (1, 3)
HAND_GAP_FLOWS = ((0, 2, 1), (0, 3, 1), (1, 2, 1), (1, 3, 1))   # Handbeispiel 3: vier Bauteile auf einer Zeile, A und B gegen C und D
HAND_GAP_GRID = (1, 4)
HAND_GRID_FLOWS = ((0, 1, 3), (0, 2, 1), (1, 3, 2), (2, 3, 4))  # Handbeispiel 2 (aus Stück 1): Raster 2×2
HAND_GRID_GRID = (2, 2)

COLORS = {"best": "#2F6B65", "local": "#c0504d", "other": "rgba(150,150,150,0.6)", "line": "#4c78a8", "ok": "#2F6B65", "warn": "#b08a2e", "bad": "#b34a4a",
          "family": {"netlist": "#4c78a8", "uniform": "#9c755f", "sparse": "#b08a2e", "cluster": "#2F6B65", "star": "#8e6bb0", "skewed": "#c0504d"}}

_BASE = {"family": DEFAULT_FAMILY, "n": DEFAULT_N, "seed": DEFAULT_SEED}
PRESETS = {
    "Standardfall: Netzliste, acht Bauteile": dict(_BASE),
    "Gleichverteilte Flüsse": dict(_BASE, family="uniform"),
    "Zwei Cluster": dict(_BASE, family="cluster"),
    "Dünn besetzt": dict(_BASE, family="sparse"),
    "Stern: eine Nabe": dict(_BASE, family="star"),
    "Sechs Bauteile": dict(_BASE, n=6),
    "Zwölf Bauteile: ohne Aufzählung": dict(_BASE, n=12),
}
PRESET_HELP = {
    "Standardfall: Netzliste, acht Bauteile": "Die Familie aus Stück 1 mit acht Bauteilen auf dem 2×4-Raster: 40 320 Anordnungen, alle werden gerechnet; die Relaxationen werden gegen das aufgezählte Optimum gemessen.",
    "Gleichverteilte Flüsse": "Jedes Paar hat einen zufälligen Fluss von 0 bis 9: die dichte Matrix ohne Struktur.",
    "Zwei Cluster": "Zwei Gruppen mit starkem Fluss innerhalb und schwachem zwischen den Gruppen: hier ist ein einzelner Start der FAQ-Art besonders selten erfolgreich.",
    "Dünn besetzt": "Nur ein Teil der Paare hat überhaupt Fluss: viele Nullen in der Matrix; die konvexe Schranke ist hier meist negativ.",
    "Stern: eine Nabe": "Ein Bauteil hängt mit allen anderen stark zusammen, die übrigen kaum untereinander.",
    "Sechs Bauteile": "Sechs Bauteile auf dem 2×3-Raster: 720 Anordnungen, alle Rechnungen dauern Sekundenbruchteile.",
    "Zwölf Bauteile: ohne Aufzählung": "Zwölf Bauteile auf dem 3×4-Raster: 479 001 600 Anordnungen, nicht aufzählbar; verglichen wird mit der besten von 200 steilsten Abstiegen.",
}


def n_text(n: int) -> str:
    """Anzahl mit Leerzeichen als Tausendertrenner: 362880 → „362 880“."""
    return f"{n:,}".replace(",", " ")


assert set(PRESETS) == set(PRESET_HELP) and set(FAMILY_NAMES) == set(FAMILIES) == set(FAMILY_HELP)
assert all(p["n"] in SIZES and p["family"] in FAMILIES and 0 <= p["seed"] <= SEED_MAX for p in PRESETS.values())
assert all(r * c == n for n, (r, c) in GRID.items()) and sum(SKEW_WEIGHTS) == 100 and set(GRID) == set(SIZES) and set(SWEEP_SEEDS_N) == set(SWEEP_SIZES)
