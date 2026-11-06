"""Doppeltstochastische Relaxation - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 6 der Reihe „Anordnen: relaxieren und runden“: statt der Zielfunktion (Stück 3 bis 5) wird die Menge der Anordnungen relaxiert, auf die doppeltstochastischen Matrizen (Birkhoff-Polytop). Konvex gibt das eine gültige, grobe Schranke und eine
Lösung im Inneren, die gerundet selten optimal ist; indefinit (FAQ-Art) gibt es Näherungen, bei denen die Startwahl und die Zahl der Starts entscheiden. Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import bkr_constants as C
from bkr_evaluation import (
    Settings,
    de,
    family_summary,
    fam,
    hand_data,
    instance_data,
    landscape_of,
    large_summary,
    median_of,
    num,
    pct,
    sweep,
    verdict_convex,
    verdict_core,
    verdict_faq,
    verdict_large,
    verdict_solution,
    verdict_starts,
    verdict_starts_sweep,
    verdict_sweep,
)
from bkr_layout import dunkles_design, kopf_css
from bkr_pdf_export import build_pdf
from bkr_presets import bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sidebar_presets, sync_query_params
from bkr_visualization import build_best_curve, build_best_hits, build_board, build_bounds_bar, build_convex_trace, build_cost_hist, build_faq_traces, build_large, build_rounding_bars, build_starts_bars, build_x_heatmap

st.set_page_config(page_title="Doppeltstochastische Relaxation – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner="Rechne alle Anordnungen …")
def _landscape(family, n, seed):
    return landscape_of(Settings(family, n, seed))


@st.cache_data(show_spinner=False)
def _sweep(family, n):
    return sweep(family, n)


def verdict(pair):
    kind, text = pair
    getattr(st, kind)(text)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in bkr_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🔀 Doppeltstochastische Relaxation")
st.markdown(
    """
**Was passiert, wenn man die Menge der Anordnungen relaxiert?** Dieselbe Aufgabe wie in Stück 1 bis 5: `n` Bauteile auf `n` Plätze, Kosten gleich der Summe aus **Fluss mal Abstand** über alle Paare. Hier wird nicht die Zielfunktion verändert, sondern die **Menge**: statt einer Permutationsmatrix `X` ist jede
**doppeltstochastische Matrix** erlaubt (Einträge ≥ 0, jede Zeile und jede Spalte summiert zu 1; die Ecken dieser Menge sind genau die Permutationsmatrizen). Gelöst wird mit **Frank–Wolfe** und der **linearen Zuordnung** (Ungarische Methode) als Richtungsschritt. Zwei Gesichter:
**konvex** (die Spektren werden so verschoben, dass die Zielfunktion konvex wird: das gibt eine gültige untere Schranke und eine Lösung, die nicht eindeutig sein muss und gerundet wird) und **indefinit** (die Zielfunktion bleibt, wie sie ist; man nähert sich je nach Start verschiedenen stationären Punkten, in der Art der FAQ-Methode).
Die sechs Familien der Flussmatrix sind **eigene, einfache Erfindungen** (aus Stück 2); beide Relaxationen sind **eigene einfache Fassungen**, nicht die Verfahren der Literatur. Stück 6 der Reihe „Anordnen: relaxieren und runden“.

Weiter unten: **fünf Experimente** (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert es", expanded=True):
    st.markdown(
        """
1. **Die Relaxation:** `X` ist eine doppeltstochastische Matrix; die Zielfunktion `g(X) = ½ tr(F X D Xᵀ)` wird auf dieser Menge minimiert. **Frank–Wolfe:** aus `X` die Richtung bestimmen, die der Gradient `F X D` nahelegt (eine Permutation, aus der linearen Zuordnung), und mit exakter Schrittweite ein Stück dorthin gehen.
2. **Konvex:** `F` und `D` werden um ein Vielfaches der Einheitsmatrix verschoben, bis sie positiv semidefinit sind; auf Permutationen ändert das nichts außer einer Konstanten `½ s_F s_D n`. Der Frank–Wolfe-Lückenwert ist in jedem Schritt eine **gültige untere Schranke**. Die Lösung wird **gerundet** (lineare Zuordnung auf `−X`).
3. **Indefinit (FAQ-Art):** dasselbe Frank–Wolfe auf `g` selbst, bis zu 100 Schritte, dann Rundung. **Start:** der Schwerpunkt `J/n` oder zufällig im Inneren (`0,6 ·` zufällige Permutation `+ 0,4 · J/n`). Je Start ein anderes Ergebnis.
4. **Die Lücke:** (Optimum − Schranke) / Optimum. Bei Instanzen, die sich nicht aufzählen lassen, steht die beste gefundene Lösung (beste von 200 steilsten Abstiegen) an der Stelle des Optimums.
5. **Feste Instanzen:** Vergleiche laufen über feste Seeds (n = 6: 20, n = 8: 10, große Instanzen: 10 je Familie), nie über einen einzelnen.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Die Instanz (Experiment 1 bis 4)**")
    family = st.selectbox("Familie der Flussmatrix", C.FAMILIES, key="family_select", format_func=lambda f: C.FAMILY_NAMES[f], help="Wie der Fluss zwischen den Bauteilen verteilt ist. Die Beschreibung steht unter dem Regler.")
    st.caption(C.FAMILY_HELP[family])
    n = st.select_slider("Bauteile n", options=C.SIZES, key="n_select", help="Bis n = 9 wird alles aufgezählt (n = 9 auf Knopfdruck, rund eine Viertelminute); ab n = 12 steht die beste von 200 Abstiegen an der Stelle des Optimums.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Instanz", width="stretch", on_click=randomize_seed)

sync_query_params({"family_select": family, "n_select": int(n), "seed_input": int(seed)})
settings = Settings(family, int(n), int(seed))

gated = settings.n == 9 and not st.session_state.get("n9_ok")
if gated:
    st.info("Bei n = 9 sind es 362 880 Anordnungen; die vollständige Rechnung dauert rund eine Viertelminute (auf Streamlit Cloud eher länger). Mit dem Knopf starten:")
    if st.button("⏱️ n = 9 rechnen", key="n9_button"):
        st.session_state["n9_ok"] = True
        st.rerun()
else:
    ls = _landscape(settings.family, settings.n, settings.seed) if settings.n in C.EXACT_SIZES else None
    v = instance_data(settings, ls)
    cv = v["convex"]
    ref_label = "Optimum" if v["exact"] else "Beste gefundene Lösung"

    # --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

    st.markdown("## 🎯 Ergebnis")
    m1, m2, m3 = st.columns(3)
    m4, m5 = st.columns(2)
    m1.metric("Konvexe Schranke (CQ)", num(cv.cq), delta=f"Lücke {pct(v['gap_cq'], 1)}", delta_color="off", help="Gültige untere Schranke aus der konvexen Relaxation (Spektralverschiebung); negativ heißt: schwächer als die triviale Schranke 0.")
    m2.metric("Rundung der konvexen Lösung", str(cv.rounded_cost), delta=f"{pct((cv.rounded_cost - v['ref']) / v['ref'], 1)} über dem Vergleichswert", delta_color="off", help="Kosten der Anordnung, die man erhält, wenn man die konvexe Lösung per Zuordnung rundet.")
    m3.metric(f"Bestes von {v['starts']} Starts", str(v["best_cost"]), delta=f"{pct(v['start_success'], 1)} der Starts erreichen den Vergleichswert", delta_color="off", help="FAQ-Art: Frank–Wolfe auf der indefiniten Zielfunktion von zufälligen Starts, gerundet.")
    m4.metric("Gilmore–Lawler", num(v["glb"]), delta=f"Lücke {pct(v['gap_glb'], 1)}", delta_color="off", help="Die Zuordnungsschranke aus Stück 3, zum Vergleich.")
    m5.metric(ref_label, str(v["ref"]), help="Die kleinsten Kosten aller n! Anordnungen (aufgezählt, bewiesen)." if v["exact"] else "Die beste von 200 steilsten Abstiegen aus Zufallsstarts: eine obere Schranke des Optimums, kein Beweis.")
    st.plotly_chart(build_bounds_bar(v), width="stretch", key="bounds_chart")

    if st.button("📄 PDF des aktuellen Ergebnisses erstellen", key="pdf_make"):
        st.session_state["pdf_made"] = (settings, build_pdf(settings, v))
    made = st.session_state.get("pdf_made")
    if made and made[0] == settings:
        st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="birkhoff-relaxation.pdf", mime="application/pdf", key="pdf_download")

    # --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

    st.markdown("## ❓ Kernfrage: Was passiert, wenn man die Menge der Anordnungen relaxiert?")
    verdict(verdict_core(settings, v))

    st.markdown("---")

    # --- Experiment 1 ---------------------------------------------------------------------------------------------------------------------------

    st.subheader("🔬 Experiment 1: Die konvexe Relaxation")
    st.caption("Frank–Wolfe auf der konvexen Zielfunktion: nach jedem Schritt eine gültige untere Schranke (blau) und die Kosten der gerundeten Anordnung (rot). Darunter drei Handbeispiele zum Nachrechnen.")
    st.plotly_chart(build_convex_trace(cv, v["ref"]), width="stretch", key="convex_trace")
    verdict(verdict_convex(settings, v))
    with st.expander("Drei Handbeispiele zum Nachrechnen"):
        for h in hand_data():
            hc, hb = h["convex"], h["bary"]
            start = "der Schwerpunkt ist stationär (Lücke 0, der Lauf bewegt sich nicht)" if hb.gap0 < 1e-9 else f"vom Schwerpunkt aus {hb.steps} Schritte"
            st.markdown(f"**{h['name']}** (Plätze auf einem {h['inst'].rows}×{h['inst'].cols}-Raster): Optimum **{h['opt']}**, Gilmore–Lawler {num(h['glb'])}, konvexe Schranke **{de(hc.cq, 2)}** (Verschiebung s_F = {de(hc.s_f, 2)}, s_D = {de(hc.s_d, 2)}), "
                        f"Rundung der konvexen Lösung **{hc.rounded_cost}**, größter Abstand eines Eintrags von 0 oder 1: {de(hc.interior, 2)}; FAQ-Art vom Schwerpunkt: {start}, gerundet zu Kosten {hb.cost}.")

    st.markdown("---")

    # --- Experiment 2: Lösung -------------------------------------------------------------------------------------------------------------------

    st.subheader("🔬 Experiment 2: Die Lösung der konvexen Relaxation und ihre Rundung")
    st.caption("Links die konvexe Lösung `X` (Bauteil auf Platz): sie ist nicht eindeutig und meist gebrochen. Rechts die Anordnung, die die Rundung daraus macht. Weiter unten der Vergleich über feste Instanzen.")
    s1, s2 = st.columns(2)
    with s1:
        st.plotly_chart(build_x_heatmap(v["inst"], cv.X, "Lösung X der konvexen Relaxation"), width="stretch", key="x_chart")
    with s2:
        st.plotly_chart(build_board(v["inst"], cv.perm, f"Rundung der konvexen Lösung (Kosten {cv.rounded_cost})"), width="stretch", key="board_rounded")
    verdict(verdict_solution(settings, v))

    st.markdown("---")

    # --- Experiment 3 ---------------------------------------------------------------------------------------------------------------------------

    st.subheader("🔬 Experiment 3: Die indefinite Relaxation (FAQ-Art)")
    st.caption("Frank–Wolfe auf der Zielfunktion selbst: der Lauf vom Schwerpunkt (rot) und die ersten Zufallsstarts; rechts die Kosten aller gerundeten Ergebnisse (grün: Vergleichswert, gestrichelt: die gerundete konvexe Lösung).")
    f1, f2 = st.columns(2)
    with f1:
        st.plotly_chart(build_faq_traces(v), width="stretch", key="faq_trace")
    with f2:
        st.plotly_chart(build_cost_hist(v), width="stretch", key="cost_hist")
    verdict(verdict_faq(settings, v))

    st.markdown("---")

    # --- Experiment 4: Zahl der Starts ----------------------------------------------------------------------------------------------------------

    st.subheader("🔬 Experiment 4: Die Zahl der Starts entscheidet")
    st.caption("Die Kosten des besten von K Starts gegen K für die gewählte Instanz; darunter, über feste Instanzen, der Anteil erfolgreicher Starts gegen den exakten Erfolg eines Abstiegs (Experiment 2 im Vergleich unten starten).")
    st.plotly_chart(build_best_curve(v), width="stretch", key="curve_chart")
    verdict(verdict_starts(settings, v))

    st.markdown("---")

# --- Vergleiche über feste Instanzen (Experiment 2 und 4) ---------------------------------------------------------------------------------------

st.subheader("🔬 Vergleich über feste Instanzen (zu Experiment 2 und 4)")
st.caption("Je Familie feste Instanzen, alle Anordnungen aufgezählt: die Lücke der konvexen Schranke, die Rundung der konvexen Lösung und der Anteil erfolgreicher Starts der FAQ-Art (50 Zufallsstarts je Instanz). Die Größe wählen und starten (n = 6: rund eine Minute, n = 8: rund zwei Minuten).")
sweep_n = st.radio("Größe des Vergleichs", C.SWEEP_SIZES, key="sweep_n", horizontal=True, format_func=lambda x: f"n = {x}")
done = st.session_state.setdefault("sweeps_done", {})
if not done.get(sweep_n):
    wait = "rund eine Minute" if sweep_n == 6 else "rund zwei Minuten (auf Streamlit Cloud eher länger)"
    st.info(f"Der Vergleich für n = {sweep_n} rechnet 6 Reihen zu je {len(C.SWEEP_SEEDS_N[sweep_n])} Instanzen (mit Aufzählung, konvexer Relaxation und {C.STARTS_SWEEP} Starts) und dauert {wait}. Mit dem Knopf starten:")
    if st.button(f"⏱️ Vergleich für n = {sweep_n} rechnen", key=f"sweep_button_{sweep_n}"):
        bar = st.progress(0.0, text="Rechne die Familien …")
        for i, f in enumerate(C.FAMILIES):
            _sweep(f, sweep_n)
            bar.progress((i + 1) / len(C.FAMILIES), text=f"{C.FAMILY_NAMES[f]} fertig")
        done[sweep_n] = True
        st.rerun()
else:
    by = {f: _sweep(f, sweep_n) for f in C.FAMILIES}
    st.markdown("**Die konvexe Relaxation** (Experiment 1 und 2): Lücke der Schranke und Kosten der Rundung gegen das Optimum, Median je Familie (logarithmische Achse).")
    st.plotly_chart(build_rounding_bars(sweep_n, by, median_of), width="stretch", key="rounding_chart")
    table = []
    for f in C.FAMILIES:
        s = family_summary(by[f])
        table.append({"Familie": fam(f), "Lücke konvex (Median)": pct(s["gap_cq"]), "konvex unter 0": f"{s['cq_negative']} von {s['count']}", "Lücke Gilmore–Lawler": pct(s["gap_glb"], 1), "Rundung über Optimum (Median)": pct(s["round_over"]),
                      "Rundung optimal": f"{s['round_hit']} von {s['count']}", "größter Abstand zu 0/1 (Median)": de(s["interior"], 2)})
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
    verdict(verdict_sweep(sweep_n, by))

    st.markdown("**Die Zahl der Starts** (Experiment 3 und 4): Anteil der Zufallsstarts, die das Optimum erreichen (Median über die Instanzen), gegen den exakten Erfolg eines steilsten Abstiegs aus Stück 1.")
    st.plotly_chart(build_starts_bars(sweep_n, by, median_of), width="stretch", key="starts_chart")
    table = []
    for f in C.FAMILIES:
        s = family_summary(by[f])
        table.append({"Familie": fam(f), "Start mit Optimum (Median)": de(s["start_success"], 2), "Spanne": f"{de(s['start_min'], 2)} bis {de(s['start_max'], 2)}", "Abstieg (exakt)": de(s["swap_success"], 2),
                      f"Bestes von {C.STARTS_SWEEP} findet Optimum": f"{s['best_hit']} von {s['count']}", "Nur Schwerpunkt findet Optimum": f"{s['bary_hit']} von {s['count']}", "Schwerpunkt stationär": f"{s['bary_stationary']} von {s['count']}"})
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
    by_n = {m: {f: _sweep(f, m) for f in C.FAMILIES} for m in C.SWEEP_SIZES if done.get(m)}
    st.plotly_chart(build_best_hits(by_n), width="stretch", key="hits_chart")
    verdict(verdict_starts_sweep(sweep_n, by))

st.markdown("---")

# --- Experiment 5: große Instanzen --------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: Jenseits der Aufzählung")
st.caption("Für n = 12 und 16: das beste von 20 Zufallsstarts der FAQ-Art gegen die beste von 200 steilsten Abstiegen (Vergleichswert), je Familie zehn feste Instanzen. Eine positive Abweichung heißt: die FAQ-Art ist schlechter.")
if not st.session_state.get("large_ok"):
    st.info("Die Rechnung für n = 12 und 16 dauert rund eine halbe Minute (auf Streamlit Cloud eher länger). Mit dem Knopf starten:")
    if st.button("⏱️ Große Instanzen rechnen (n = 12, 16)", key="large_button"):
        bar = st.progress(0.0, text="Rechne die großen Instanzen …")
        todo = [(f, m) for m in C.LARGE_SWEEP_SIZES for f in C.FAMILIES]
        for i, (f, m) in enumerate(todo):
            _sweep(f, m)
            bar.progress((i + 1) / len(todo), text=f"n = {m}, {C.FAMILY_NAMES[f]} fertig")
        st.session_state["large_ok"] = True
        st.rerun()
else:
    by_big = {m: {f: _sweep(f, m) for f in C.FAMILIES} for m in C.LARGE_SWEEP_SIZES}
    st.plotly_chart(build_large(by_big, median_of), width="stretch", key="large_chart")
    table = []
    for m in C.LARGE_SWEEP_SIZES:
        for f in C.FAMILIES:
            s = large_summary(by_big[m][f])
            table.append({"n": m, "Familie": fam(f), "bestes von 20 (Abweichung, Median)": pct(s["gap_best"], 1), "mittlerer Start": pct(s["gap_median"], 1), "Schwerpunkt": pct(s["gap_bary"], 1), "FAQ besser / gleich / schlechter": f"{s['better']} / {s['equal']} / {s['worse']}"})
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
    verdict(verdict_large(by_big))

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Beide Relaxationen sind einfache Eigenfassungen** | Die konvexe Schranke nutzt nur eine Verschiebung der Spektren; die QP-Schranke der Literatur ist stärker. Die FAQ-Art ist nicht das Programm der FAQ-Autoren. Alle Befunde gelten für diese Fassungen. | – |
| **Die sechs Familien sind erfunden** | Echte Platinen oder QAPLIB-Instanzen haben andere Strukturen; Aussagen wie „bei Cluster-Flüssen mit n = 8 ist ein Start selten erfolgreich“ gelten nur für diese Familien. | – |
| **Das Optimum ist nur bis n = 9 bekannt** | Ab n = 12 steht die beste von 200 steilsten Abstiegen an seiner Stelle; die Vergleiche mit der FAQ-Art sind dann Vergleiche zweier Heuristiken, kein Beweis. | Stück 8 (Branch and Bound) |
| **Frank–Wolfe wird nach höchstens 100 Schritten abgebrochen** | Die Lücke wird auf diesen Instanzen kaum je null; der Anteil gefundener Optima ändert sich zwischen 100 und 300 Schritten bei n = 8 nicht und sinkt bei n = 6 leicht, aber ein Lauf ist nicht „zu Ende“. | – |
| **Ein Start gegen einen Abstieg** | Ein Lauf der FAQ-Art kostet viele Zuordnungen, ein Abstieg Millisekunden; „ein Start ist seltener erfolgreich“ ist kein Vergleich bei gleicher Rechenzeit. | Trajektorien-Metaheuristiken |
| **Wenige Instanzen** | n = 6: 20 Instanzen je Familie, n = 8 und große Instanzen: 10; Unterschiede von wenigen Prozentpunkten zwischen Familien sind nicht belastbar. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Die Aufgabe (wie in Stück 1).** Symmetrischer Fluss $F$, symmetrischer Abstand $D$ der Plätze, beide mit Diagonale 0; für die Permutationsmatrix $X$ ($X_{i,p_i} = 1$) sind die Kosten $\mathrm{cost}(p) = \tfrac12 \operatorname{tr}(F X D X^\top)$.

**Relaxation der Menge.** Das Birkhoff-Polytop $\mathcal{B} = \{X \ge 0 : X \mathbf{1} = \mathbf{1},\ X^\top \mathbf{1} = \mathbf{1}\}$ hat genau die Permutationsmatrizen als Ecken. Man minimiert $g(X) = \tfrac12 \operatorname{tr}(F X D X^\top)$ über $\mathcal{B}$; der Gradient ist $\nabla g(X) = F X D$.

**Frank–Wolfe.** Aus $X \in \mathcal{B}$: $Y \in \arg\min_{P \text{ Permutation}} \langle \nabla g(X), P \rangle$ (lineare Zuordnung), $\Delta = Y - X$. Dann ist $g(X + \alpha \Delta) = g(X) + b\,\alpha + a\,\alpha^2$ mit $b = \langle \nabla g(X), \Delta \rangle \le 0$ und $a = \tfrac12 \langle F \Delta D, \Delta \rangle$;
$\alpha = \min(1, -b/(2a))$ bei $a > 0$, sonst $\alpha = 1$. Die **Frank–Wolfe-Lücke** $-b$ ist genau an stationären Punkten null. Der Schwerpunkt $J/n$ hat den Gradienten $u w^\top / n^2$ mit $u = F e$, $w = D e$; ist $u$ oder $w$ konstant, ist er für alle Permutationen gleich: $J/n$ ist dann stationär.

**Konvexe Relaxation.** Für Permutationsmatrizen $P$ gilt $\operatorname{tr}(P D P^\top) = \operatorname{tr}(D) = 0$ und $\operatorname{tr}(F P P^\top) = \operatorname{tr}(F) = 0$. Mit $F' = F + s_F I$, $D' = D + s_D I$, $s_F = -\lambda_{\min}(F)$, $s_D = -\lambda_{\min}(D)$ (beide positiv semidefinit) ist $g'(X) = \tfrac12 \operatorname{tr}(F' X D' X^\top)$ konvex, und
$$g'(P) = \mathrm{cost}(P) + \tfrac12 s_F s_D\, n .$$
Für konvexes $g'$ liegt jeder Funktionswert über der Tangente: $g'(X) + \min_Y \langle \nabla g'(X), Y - X \rangle \le \min_{\mathcal{B}} g' \le \min_P g'(P)$. Also ist in **jedem** Frank–Wolfe-Schritt
$$\mathrm{CQ} = g'(X) + b - \tfrac12 s_F s_D\, n \;\le\; \mathrm{opt}$$
eine gültige untere Schranke. Die Konstante $\tfrac12 s_F s_D n$ ist meist ein Vielfaches des Optimums; dazu passt, dass CQ oft sehr grob und bei dünn besetzten, sternförmigen und schiefen Flüssen oft negativ ist (eine Deutung, keine gemessene Ursache).

**Rundung.** Die Permutation, die $\langle X, P \rangle$ maximiert (lineare Zuordnung auf $-X$); ihre Kosten sind eine obere Schranke.

**Indefinit (FAQ-Art).** $F$ und $D$ haben Eigenwerte beider Vorzeichen, $g$ ist nicht konvex: Frank–Wolfe nähert sich einem stationären Punkt (nach 100 Schritten haben ihn nur etwa 1 % der Läufe erreicht), das Ergebnis hängt vom Start ab. Start: $J/n$ oder $\lambda P + (1-\lambda) J/n$ mit einer zufälligen Permutationsmatrix $P$ und $\lambda = 0{,}6$.

**Lücke.** $(\mathrm{opt} - B)/\mathrm{opt}$ für eine Schranke $B$; ohne bekanntes Optimum steht die beste gefundene Lösung $U \ge \mathrm{opt}$ an dessen Stelle, und $(U - B)/U \ge (\mathrm{opt} - B)/\mathrm{opt}$ ist eine **obere Grenze** der wahren Lücke, solange die Schranke nichtnegativ ist.

**Literatur.** J. T. Vogelstein u. a. (FAQ, Fast Approximate Quadratic assignment), M. Zaslavskiy, F. Bach, J.-P. Vert (PATH), K. M. Anstreicher, N. W. Brixius (QP-Schranke): Angaben aus der Fachliteratur übernommen, hier **nicht nachgelesen**; Selbstangaben der Autoren zur Güte werden nicht übernommen.
Die Verschiebung, die Zerlegung $g'(P) = \mathrm{cost}(P) + \tfrac12 s_F s_D n$, die zertifizierte Schranke und die Schrittweite sind **eigene** Herleitungen und werden durch Tests (Aufzählung, alle Permutationen kleiner Instanzen, `scipy`) geprüft. Die Zahlen dieser Demo sind eigene Rechnungen an eigenen, erfundenen Familien.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
