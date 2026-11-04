# Doppeltstochastische Relaxation – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-birkhoff-relaxation-demo.streamlit.app/)** -->

Stück 6 von 15 der **Konzepte-Reihe „Anordnen: relaxieren und runden“** für die Website „Sebastian Hanisch – Operations Research und Machine Learning“ (die Konzepte-Seite der Reihe gibt es noch nicht). Stück 3, [Die Gilmore–Lawler-Schranke](https://github.com/sebastian-hanisch/gilmore-lawler-demo), und Stück 5, [Linearisierung und Dual Ascent](https://github.com/sebastian-hanisch/linearisierung-schranke-demo), verändern die **Zielfunktion**, um eine Schranke zu bekommen. Dieses Stück macht das Gegenteil und **relaxiert die Menge** (Kante der Reihe: *Kontrast*): Statt einer Anordnung darf jede **doppeltstochastische Matrix** gewählt werden. Es zeigt die beiden Gesichter dieser Relaxation: die **konvexe** als Schranke und die **indefinite** als Näherungsverfahren in der Art von FAQ.

## Warum dieses Problem

Eine Anordnung ist eine Permutationsmatrix `X` (`X[i, p_i] = 1`), die Kosten sind `½ tr(F X D Xᵀ)`. Die Permutationsmatrizen sind die Ecken des **Birkhoff-Polytops**, der Menge aller nichtnegativen Matrizen, deren Zeilen und Spalten sich zu 1 summieren. Lässt man alle Punkte dieser Menge zu, entsteht ein stetiges Problem, das man mit Frank–Wolfe lösen kann, wobei jeder Schritt eine **lineare Zuordnung** ist. Die Kernfrage: **Was bringt die konvexe Fassung als Schranke, wie gut ist ihre gerundete Lösung, und wie viele Starts braucht die indefinite Fassung?**

Die Antwort dieser Messung: Die konvexe Schranke (mit einer einfachen Spektralverschiebung) ist **gültig, aber grob** und liegt bei vielen Instanzen unter 0; die Lösung liegt im Inneren des Polytops und wird gerundet fast nie zum Optimum. Die indefinite Fassung findet das Optimum meist, **wenn man genug Starts macht**; ein einzelner Start trifft es selten. Das sind **Befunde an sechs erfundenen Familien und an einfachen Eigenfassungen**, keine Sätze.

## Modell

- **Aufgabe** (wie in Stück 1 bis 5): `n` Bauteile auf `n` Plätze, `cost(p) = ½ Σ_{i≠k} F[i,k] · D[p_i, p_k]` (alle geordneten Paare). Kern (Kosten, Aufzählung, Zufall, sechs Familien der Flussmatrix, Gilmore–Lawler-Schranke, beste gefundene Lösung, projizierte Schranke, Ungarische Methode) als Kopie aus Stück 3 bis 5 (`bkr_cost.py`, `bkr_landscape.py`, `bkr_rng.py`, `bkr_instance.py`, `bkr_bound.py`, `bkr_spectral.py`, `bkr_assignment.py`; Tests vergleichen die Instanzen mit den Schwester-Repos, wenn sie nebenan liegen). Die Familien sind **eigene, einfache Erfindungen**, keine echten Platinen und keine QAPLIB-Instanzen; Plätze liegen auf einem Raster (Manhattan-Abstand).
- **Frank–Wolfe mit Zuordnung** (`bkr_birkhoff.py`): aus `X` die Permutation `Y` bestimmen, die `⟨F X D, Y⟩` minimiert (lineare Zuordnung), dann auf der Strecke von `X` nach `Y` die **exakte Schrittweite** nehmen (die Zielfunktion ist dort eine Parabel). Die **Frank–Wolfe-Lücke** ist null genau an stationären Punkten.
- **Konvex (Schranke):** die Diagonalen von `F` und `D` sind null, also ändert die Verschiebung `F' = F + s_F·I`, `D' = D + s_D·I` mit `s_F = −λ_min(F)`, `s_D = −λ_min(D)` die Kosten jeder Permutation nur um die Konstante `½ s_F s_D n`. Beide Matrizen sind danach positiv semidefinit, die Zielfunktion ist **konvex**, und jeder Frank–Wolfe-Schritt liefert eine **gültige untere Schranke** `CQ`. Das ist eine **einfache Spektralverschiebung** (eigene Herleitung), nicht die stärkere Schranke von Anstreicher und Brixius.
- **Indefinit (FAQ-Art):** dasselbe Frank–Wolfe auf der unveränderten Zielfunktion, höchstens 100 Schritte; Start im Schwerpunkt `J/n` oder zufällig im Inneren, `0,6·P + 0,4·J/n` mit zufälliger Permutation `P` (SplitMix64); Rundung per Zuordnung. Das ist **eine Näherung in der Art von FAQ (Vogelstein u. a.)**, nicht ihr Programm.
- **Vergleichswert und Lücke:** bis `n = 9` das aufgezählte Optimum, Lücke `(Optimum − Schranke) / Optimum`; ab `n = 10` ersetzt die **beste von 200 steilsten Abstiegen** das Optimum, die Lücken sind dann obere Grenzen.
- **Experimente:** (1) die konvexe Relaxation (Schranke Schritt für Schritt, gegen Gilmore–Lawler und die projizierte Schranke, Handbeispiele); (2) die Lösung `X` und ihre Rundung; (3) die indefinite Relaxation (Schwerpunkt und Zufallsstarts); (4) die Zahl der Starts; (5) jenseits der Aufzählung (`n` = 12, 16). Formeln im 📐-Expander der App.

## Methodik

Experiment 1 bis 4 rechnen **eine** Instanz (`n` = 6 oder 8 sofort mit Aufzählung; `n` = 9 auf Knopfdruck, ebenfalls mit Aufzählung; `n` = 12, 16, 20 ohne Aufzählung und mit 20 statt 50 Starts). Der Vergleich über feste Instanzen rechnet auf Knopfdruck je Familie 20 Instanzen bei `n` = 6 (Seeds 100000 bis 100019) und zehn bei `n` = 8 (Seeds 100000 bis 100009), je 50 Zufallsstarts der FAQ-Art; Experiment 5 rechnet je Familie zehn Instanzen für `n` = 12 und 16 mit 20 Starts gegen die beste von 200 steilsten Abstiegen. Alle Aussagen laufen über feste Seeds, nie über einen einzelnen. Die konvexe Relaxation läuft 300 Frank–Wolfe-Schritte. Zeiten hängen vom Rechner ab und stehen nirgends als Zahl in Tests oder in dieser README.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` (deterministisch; Prozentwerte sind auf die genannte Rundung genau). Reihenfolge der Familien: Netzliste, gleichverteilt, dünn besetzt, zwei Cluster, Stern, wenige starke Flüsse. Die Lücke der konvexen Schranke ist `(Optimum − CQ) / Optimum` und kann über 100 % liegen, wenn `CQ` negativ ist.

| Frage | Befund |
|---|---|
| Handbeispiele | Drei Bauteile auf einer Zeile (Optimum 6): `CQ` knapp unter 5,69 (die Schranke nähert sich in 300 Schritten langsam), Rundung optimal; Raster 2×2 (Optimum 10): `CQ` 9,883, die konvexe Lösung ist der **Schwerpunkt**, die Rundung optimal; vier Bauteile auf einer Zeile (Optimum 6): `CQ` 3,172, die konvexe Lösung ist ebenfalls der **Schwerpunkt**, die Rundung kostet **8 statt 6**. In den beiden letzten Beispielen ist der Schwerpunkt ein **stationärer Punkt**: Frank–Wolfe bewegt sich von dort nicht weg |
| Standardinstanz (Netzliste, `n` = 8, Seed 35) | Optimum 31; Gilmore–Lawler 27; projizierte Schranke 13,7; **konvexe Schranke −22,1** (Verschiebungen 5,105 und 6,828, Konstante 139,45); Rundung der konvexen Lösung **54**; Schwerpunkt als Start: Rundung 36, erstes Optimum beim 3. von 50 Zufallsstarts; 8 % der Starts enden im Optimum (die Tauschsuche aus Stück 1: 26,6 %) |
| Gültigkeit | In allen 180 Instanzen (`n` = 6 und 8) liegt `CQ` nicht über dem aufgezählten Optimum |
| Lücke der konvexen Schranke, `n` = 6 (Median in %, je 20 Instanzen) | 38,0 / 33,4 / 128,2 / 22,2 / 126,0 / 140,0; Gilmore–Lawler zum Vergleich 7,3 / 5,2 / 10,3 / 17,0 / 7,7 / 3,4; über alle 120 Instanzen konvex **74 %**, Gilmore–Lawler **7 %**. `CQ` liegt unter 0 in 0 / 0 / 17 / 0 / 19 / 16 von 20 Instanzen (52 von 120) |
| Lücke, `n` = 8 (Median in %, je 10 Instanzen) | 110,0 / 61,1 / 372,2 / 33,0 / 298,3 / 460,3; Gilmore–Lawler 10,5 / 8,0 / 6,1 / 9,8 / 14,1 / 6,4; über alle 60 Instanzen konvex **226 %**, Gilmore–Lawler **9 %**; `CQ` unter 0 in 5 / 0 / 10 / 0 / 10 / 10 (35 von 60). Grund: die Konstante `½ s_F s_D n` (Median 135 / 415 / 356 / 327 / 440 / 1 136) ist ein Vielfaches des Optimums (41 / 218,5 / 44 / 176,5 / 68 / 116) |
| Die konvexe Lösung liegt innen | größter Abstand eines Eintrags von 0 oder 1 (Median je Familie): `n` = 6: 0,45 / 0,36 / 0,43 / 0,36 / 0,47 / 0,36; `n` = 8: 0,23 / 0,23 / 0,21 / 0,21 / 0,24 / 0,21 |
| Rundung der konvexen Lösung: Kosten über dem Optimum (Median in %) | `n` = 6: 16 / 24 / 40 / 40 / 22 / 42 (insgesamt **29 %**); `n` = 8: 22 / 18 / 53 / 46 / 27 / 44 (insgesamt **33 %**), auf ganze Prozent gerundet. Optimum getroffen in **3 von 120** (`n` = 6) und **1 von 60** (`n` = 8) Instanzen |
| Ein Start der FAQ-Art: Anteil der Zufallsstarts, die im Optimum enden (Median über die Instanzen) | `n` = 6: 0,15 / 0,08 / 0,32 / 0,17 / 0,16 / 0,20; `n` = 8: 0,21 / 0,00 / 0,23 / 0,00 / 0,18 / 0,10. Die Spannen sind groß: bei `n` = 6 reicht der Anteil in jeder Familie von 0,00 bis mindestens 0,86 |
| Zum Vergleich: ein steilster Abstieg (Stück 1, alle Startanordnungen exakt gezählt) | `n` = 6: 0,81 / 0,69 / 0,86 / 0,58 / 0,83 / 0,68; `n` = 8: 0,32 / 0,30 / 0,44 / 0,52 / 0,44 / 0,31. **Ein Start der FAQ-Art ist hier seltener erfolgreich als ein Abstieg**, ein Start ist aber auch nicht gleich teuer wie ein Abstieg (kein Vergleich bei gleicher Rechenzeit) |
| Die Zahl der Starts entscheidet | das beste von 50 Starts enthält das Optimum in **97 von 120** (`n` = 6; je Familie 15 / 15 / 17 / 16 / 18 / 16 von 20) und **41 von 60** (`n` = 8; 7 / 4 / 10 / 4 / 6 / 10 von 10) Instanzen; der Schwerpunkt als einziger Start in 26 von 120 und 14 von 60; **der Schwerpunkt war in keiner dieser 180 Instanzen ein stationärer Punkt** (er ist es nur bei konstanter Zeilensumme von `F` oder `D`, wie in den Handbeispielen) |
| Große Instanzen (Vergleichswert: beste von 200 Abstiegen) | das beste von 20 Starts der FAQ-Art ist besser / gleich / schlechter in 0 / 35 / 25 von 60 Instanzen (`n` = 12) und 1 / 28 / 31 (`n` = 16); Lücke im Median je Familie (%): `n` = 12: 0,0 / 0,0 / 0,0 / 1,3 / 0,0 / 0,5; `n` = 16: 0,0 / 0,1 / 0,0 / 1,5 / 0,0 / 2,3. Bei Cluster- und schiefen Flüssen bleibt die FAQ-Art schlechter |
| Orakel | Schranke höchstens so groß wie der Wert jedes zulässigen Punkts, den `scipy.optimize.minimize` (SLSQP, Zulässigkeit geprüft) für dasselbe konvexe Problem findet; Frank–Wolfe nie um mehr als 0,1 % schlechter als dieser Löser; Zuordnung gleich `scipy.optimize.linear_sum_assignment`; Eigenwerte gleich `scipy.linalg.eigh`; Schranke nie über dem aufgezählten Optimum (36 Instanzen) |

**Was die Zahlen nicht sagen:** Die Familien sind erfunden; andere Parameter können die Rangfolge ändern. Die konvexe Schranke ist eine **einfache Spektralverschiebung**; die stärkere Schranke von Anstreicher und Brixius (und die projizierte Schranke aus Stück 4, die das Spektrum klüger nutzt) wurde nicht gebaut und nicht nachgelesen: ihre Zahlen können nicht aus diesen Messungen abgeleitet werden. Die FAQ-Art ist nicht das Programm der Literatur. Die Vergleiche zur Tauschsuche stellen **einen Start gegen einen Abstieg**. Bei `n` = 8 und bei den großen Instanzen sind es nur zehn Instanzen je Familie; ab `n` = 10 sind die Lücken obere Grenzen.

## Ehrliche Grenzen

- **Einfache konvexe Schranke:** negativ bei vielen Instanzen und weit schwächer als Gilmore–Lawler; sie ist ein Lehrbeispiel für die Konvexifizierung, kein Konkurrent.
- **Eigenfassung der FAQ-Art:** höchstens 100 Frank–Wolfe-Schritte (in der Vormessung, die nicht Teil der Tests ist, änderte sich der Anteil gefundener Optima zwischen 100 und 300 Schritten kaum), feste Startverteilung (`0,6·P + 0,4·J/n`); andere Verteilungen waren in der Vormessung nicht klar besser, wurden aber nur an `n` = 6 verglichen.
- **Rundung hängt von Gleichständen ab:** die konvexe Zielfunktion hat kein eindeutiges Minimum, die Lösung `X` hat viele fast gleiche Einträge, und welche Anordnung die Rundung wählt, entscheidet dann der Zufall des Rechenrauschens (1e-16). Damit die Zahlen auf jedem Rechner gleich sind, werden Gradient und Lösung vor jeder Zuordnung auf acht Stellen gerundet (`tie_quantized`); die Schranke bleibt davon unberührt, sie nutzt die exakte Zuordnung. Mit einer anderen Stellenzahl ändern sich die Kosten der Rundung (bei der Standardinstanz mit sechs, acht und zehn Stellen: 46, 54 und 37), die Schranke nur um 0,01. Die Zahlen zur Rundung gelten für diese Festlegung.
- **Erfundene Familien:** keine echten Platinen, keine QAPLIB-Instanzen.
- **Optimum nur bis `n` = 9:** darüber steht die beste von 200 Abstiegen an seiner Stelle.
- **Wenige Instanzen:** `n` = 6: 20 je Familie; `n` = 8 und große Instanzen: 10. Der Vergleich über feste Instanzen für `n` = 9 fehlt.
- **Quellen:** Vogelstein u. a. (FAQ), Zaslavskiy u. a. (PATH), Anstreicher und Brixius (QP-Schranke) sind aus der Fachliteratur übernommen und **nicht nachgelesen**; Verschiebung, Zerlegung der Kosten, zertifizierte Schranke und Schrittweite sind eigene Herleitungen und werden durch Tests (Aufzählung, Handrechnung, `scipy`) geprüft.

## Verifikation

- **Tests** (`pytest tests/`): **270 Tests**, rund vierzehn Minuten (Windows, Python 3.14). Handwerte (die drei Handbeispiele von Hand, Zuordnung), Eigenschaften (Doppeltstochastik bleibt erhalten, der Zielwert fällt, die Frank–Wolfe-Lücke ist nicht negativ, die untere Schranke ist gültig und steigt monoton, die Zerlegung `g'(P) = cost(P) + ½ s_F s_D n` gilt für alle Permutationen kleiner Instanzen, der Schwerpunkt ist stationär genau bei konstanter Zeilensumme von `F` oder `D`, Gleichheit mit der Handrechnung der drei Beispiele), Orakel (Aufzählung für die Gültigkeit; `scipy` für den konvexen Löser, die Zuordnung und die Eigenwerte; nur mit `requirements-dev.txt`), Randfälle (zwei Bauteile, Nullfluss, gleiche Flüsse, ungültige Eingaben), Verdikte an handgesetzten Werten, Abbildungen, PDF, Rauchtest und Pfade der App (jede Familie, `n` = 9 auf Knopfdruck, beide Vergleiche, große Instanzen), Layout, Footer und Laufzeit-Importe. `test_claims.py` rechnet jede Zahl dieser README nach (rund fünf Minuten).
- **Kopie:** Tests vergleichen die Familien mit denen von Stück 1 bis 5, sofern die Schwester-Repos nebenan liegen.
- **Nicht getestet:** Zeiten; kein Mutationstest; Linux-CI: noch nicht gelaufen (Repo noch nicht veröffentlicht).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: fünf Experimente, Vergleich über feste Instanzen, Grenzen, Formulierung |
| `bkr_rng.py` | SplitMix64 (Kopie aus Stück 1) |
| `bkr_instance.py` | Instanzen: sechs Familien der Flussmatrix, Raster, Handbeispiele |
| `bkr_cost.py` | Kosten, Tausch-`Δ` (Kopie aus Stück 2) |
| `bkr_landscape.py` | Aufzählung für das Optimum (Kopie aus Stück 2) |
| `bkr_assignment.py` | Ungarische Methode (Kopie aus Stück 5, ohne Dualvariablen) |
| `bkr_bound.py` | Gilmore–Lawler-Schranke, Runden, Abstiege und beste gefundene Lösung (Kopie aus Stück 3) |
| `bkr_spectral.py` | Eigenwerte, projizierte Schranke (Kopie aus Stück 4) |
| `bkr_birkhoff.py` | Frank–Wolfe, Spektralverschiebung, konvexe Schranke, Rundung, Läufe der FAQ-Art |
| `bkr_evaluation.py` | Zeilen für die Vergleiche, Zusammenfassungen, Verdikte |
| `bkr_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `bkr_pdf_export.py` | PDF-Export des Ergebnisses |
| `bkr_constants.py` | Größen, Familien, feste Reihen, Farben, Presets |
| `bkr_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Button |
| `bkr_layout.py` | Fester Kopf (Stylesheet) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt; kein `scipy` zur Laufzeit) |
| `tests/` | Zuordnung, Frank–Wolfe und Schranke, Kosten, Instanzen, Landschaft, Auswertung, Randfälle, Verdikte, Abbildungen, Presets, Layout, AppTest, Orakel, Zahlen der README, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Stärkere konvexe Schranke (Anstreicher–Brixius):** Primärlektüre fehlt; die einfache Verschiebung steht für sich.
- **PATH, Sinkhorn, Gromov–Wasserstein:** nur als Hinweis auf die Literatur, nicht gebaut.
- **Semidefinite Relaxation:** Stück 7; Branch and Bound: Stück 8.
- **`n` = 9 und `n` = 20 im Vergleich:** zu langsam; beide nur als Einzelinstanz.
- **Asymmetrische Flüsse und Abstände.**

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python-Version in den App-Einstellungen prüfen (lokal Python 3.14, CI Python 3.12); der Code verwendet nur Merkmale, die in Python 3.12 laufen. <!-- lint-ok: dezimalpunkt -->

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, pandas und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
