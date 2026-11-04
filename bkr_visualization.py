"""Plotly-Abbildungen. Alle 2D-Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import bkr_constants as C

PALETTE = {6: "#4c78a8", 8: "#c0504d", 12: "#2F6B65", 16: "#b08a2e", 20: "#8e6bb0"}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380, legend_below=True):
    legend = dict(orientation="h", yanchor="top", y=-0.22, x=0) if legend_below else dict(orientation="h", yanchor="bottom", y=1.02, x=0)
    fig.update_layout(height=height + (40 if legend_below else 0), margin=dict(l=10, r=10, t=50, b=10), legend=legend, plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_board(inst, p, title: str) -> go.Figure:
    """Plätze als graue Quadrate, Bauteile als Punkte mit Namen auf ihrem Platz; Verbindungen mit Fluss als Linien (Dicke nach dem größten Fluss der Instanz skaliert)."""
    xs_, ys_ = inst.xy[:, 0].astype(float), -inst.xy[:, 1].astype(float)
    pos = [(xs_[int(p[i])], ys_[int(p[i])]) for i in range(inst.n)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs_, y=ys_, mode="markers", marker=dict(symbol="square", size=34 if inst.n > 12 else 40, color="rgba(150,150,150,0.18)", line=dict(width=1, color="rgba(150,150,150,0.5)")), hoverinfo="skip", showlegend=False))
    fmax = max(1, int(inst.F.max()))
    level = lambda a, b: min(4, 1 + int(3 * int(inst.F[a, b]) / fmax))
    levels = sorted({level(a, b) for a in range(inst.n) for b in range(a + 1, inst.n) if inst.F[a, b]})
    for lv in levels:
        xs, ys = [], []
        for a in range(inst.n):
            for b in range(a + 1, inst.n):
                if inst.F[a, b] and level(a, b) == lv:
                    xs += [pos[a][0], pos[b][0], None]
                    ys += [pos[a][1], pos[b][1], None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(width=0.6 + 1.2 * lv, color="rgba(76,120,168,0.35)"), name=("schwacher Fluss", "mittlerer Fluss", "starker Fluss", "stärkster Fluss")[lv - 1], hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[q[0] for q in pos], y=[q[1] for q in pos], mode="markers+text", text=list(inst.names), textposition="middle center", showlegend=False,
                             marker=dict(size=24 if inst.n > 12 else 28, color=C.COLORS["best"]), textfont=dict(color="white", size=10), hovertemplate="%{text}<extra></extra>"))
    fig.update_layout(title=title)
    fig.update_xaxes(visible=False, range=[xs_.min() - 0.7, xs_.max() + 0.7])
    fig.update_yaxes(visible=False, range=[ys_.min() - 0.7, ys_.max() + 0.7])
    return _base(fig, height=300, legend_below=True)


# ----------------------------------------------------------------------------- Experiment 1: konvexe Relaxation

def build_bounds_bar(v: dict) -> go.Figure:
    """Schranken, Rundungen und Vergleichswert einer Instanz auf einer Achse."""
    cv = v["convex"]
    rows = [("Konvexe Schranke (CQ)", cv.cq, C.COLORS["line"]), ("projizierte Schranke", v["pb"], C.COLORS["other"]), ("Gilmore–Lawler", v["glb"], C.COLORS["other"]), ("Optimum" if v["exact"] else "beste gefundene Lösung", v["ref"], C.COLORS["best"]),
            (f"bestes von {v['starts']} Starts (FAQ-Art)", v["best_cost"], C.COLORS["warn"]), ("Rundung der konvexen Lösung", cv.rounded_cost, C.COLORS["local"])]
    fig = go.Figure(go.Bar(x=[r[1] for r in rows], y=[r[0] for r in rows], orientation="h", marker_color=[r[2] for r in rows], hovertemplate="%{y}: %{x:.2f}<extra></extra>", showlegend=False,
                           text=[f"{r[1]:.1f}".replace(".0", "") for r in rows], textposition="auto"))
    fig.update_layout(title="Untere Schranken, Optimum und gerundete Anordnungen")
    fig.update_xaxes(title_text="Kosten", zeroline=True)
    fig.update_yaxes(autorange="reversed")
    return _base(fig, height=340, legend_below=False)


def build_convex_trace(cv, ref: float) -> go.Figure:
    """Frank–Wolfe auf der konvexen Relaxation: die gültige untere Schranke und die Kosten der gerundeten Anordnung nach jedem Schritt."""
    r = list(range(1, len(cv.lowers) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r, y=list(cv.lowers), mode="lines", name="untere Schranke CQ", line=dict(color=C.COLORS["line"], width=3), hovertemplate="Schritt %{x}: %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=r, y=list(cv.rounded), mode="lines", name="Kosten der gerundeten Anordnung", line=dict(color=C.COLORS["local"], width=2), hovertemplate="Schritt %{x}: %{y}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[r[0], r[-1]], y=[ref, ref], mode="lines", name="Vergleichswert", line=dict(color=C.COLORS["best"], width=2, dash="dash"), hoverinfo="skip"))
    fig.update_layout(title="Frank–Wolfe auf der konvexen Relaxation")
    fig.update_xaxes(title_text="Schritt")
    fig.update_yaxes(title_text="Kosten")
    return _base(fig)


# ----------------------------------------------------------------------------- Experiment 2: die konvexe Lösung

def build_x_heatmap(inst, X, title: str = "Lösung X der Relaxation (Bauteil auf Platz)") -> go.Figure:
    """Die doppeltstochastische Matrix: Bauteil (Zeile) auf Platz (Spalte); gebrochene Werte zeigen, dass die Lösung keine Anordnung ist."""
    fig = go.Figure(go.Heatmap(z=X.tolist(), x=[f"Platz {k + 1}" for k in range(inst.n)], y=list(inst.names), zmin=0, zmax=1, colorscale=[[0, "rgba(150,150,150,0.10)"], [1, C.COLORS["line"]]],
                               text=[[f"{x:.2f}" for x in row] for row in X], texttemplate="%{text}" if inst.n <= 9 else None, hovertemplate="%{y} auf %{x}: %{z:.3f}<extra></extra>", colorbar=dict(title="X", thickness=12), xgap=2, ygap=2))
    fig.update_layout(title=title)
    fig.update_yaxes(autorange="reversed")
    return _base(fig, height=max(320, 26 * inst.n + 120), legend_below=False)


def build_rounding_bars(n: int, by_family: dict, median_of) -> go.Figure:
    """Je Familie: Median der Kosten der gerundeten konvexen Lösung über dem Optimum und Median der Lücke der konvexen Schranke."""
    fams = [f for f in C.FAMILIES if by_family.get(f)]
    x = [C.FAMILY_NAMES[f] for f in fams]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=[100 * median_of(by_family[f], "round_over") for f in fams], name="Rundung: Kosten über dem Optimum", marker_color=C.COLORS["local"]))
    fig.add_trace(go.Bar(x=x, y=[100 * median_of(by_family[f], "gap_cq") for f in fams], name="Lücke der konvexen Schranke", marker_color=C.COLORS["line"]))
    fig.add_trace(go.Bar(x=x, y=[100 * median_of(by_family[f], "gap_glb") for f in fams], name="Lücke Gilmore–Lawler", marker_color=C.COLORS["other"]))
    fig.update_layout(title=f"Die konvexe Relaxation (Median, n = {n})", barmode="group")
    fig.update_yaxes(title_text="Prozent", type="log", dtick=1, exponentformat="power", showexponent="all")
    return _base(fig)


# ----------------------------------------------------------------------------- Experiment 3: die indefinite Relaxation

def build_faq_traces(v: dict, shown: int = 8) -> go.Figure:
    """Zielwert g(X) je Frank–Wolfe-Schritt: der Lauf vom Schwerpunkt und die ersten Zufallsstarts."""
    fig = go.Figure()
    bv = v["bary"].values
    fig.add_trace(go.Scatter(x=list(range(len(bv))), y=list(bv), mode="lines", name="Schwerpunkt", line=dict(color=C.COLORS["local"], width=3), hovertemplate="Schritt %{x}: %{y:.2f}<extra></extra>"))
    for k, run in enumerate(v["runs"][:shown]):
        fig.add_trace(go.Scatter(x=list(range(len(run.values))), y=list(run.values), mode="lines", name=f"Start {k + 1}", line=dict(width=1.2), opacity=0.8, hovertemplate="Schritt %{x}: %{y:.2f}<extra>Start " + str(k + 1) + "</extra>"))
    fig.add_trace(go.Scatter(x=[0, max(len(bv), 2) - 1], y=[v["ref"], v["ref"]], mode="lines", name="Vergleichswert", line=dict(color=C.COLORS["best"], width=2, dash="dash"), hoverinfo="skip"))
    fig.update_layout(title="Zielwert der Relaxation je Frank–Wolfe-Schritt")
    fig.update_xaxes(title_text="Schritt")
    fig.update_yaxes(title_text="Zielwert g(X) (relaxierte Kosten)")
    return _base(fig)


def build_cost_hist(v: dict) -> go.Figure:
    """Die Kosten der gerundeten Ergebnisse der Zufallsstarts (Häufigkeit je Kostenwert), mit dem Vergleichswert und der Rundung der konvexen Lösung."""
    from collections import Counter
    cnt = Counter(v["costs"])
    xs = sorted(cnt)
    fig = go.Figure(go.Bar(x=xs, y=[cnt[c] for c in xs], marker_color=[C.COLORS["best"] if c == v["ref"] else C.COLORS["other"] for c in xs], hovertemplate="Kosten %{x}: %{y} Starts<extra></extra>", showlegend=False))
    fig.add_vline(x=v["convex"].rounded_cost, line=dict(color=C.COLORS["local"], dash="dot"), annotation_text="konvexe Lösung gerundet", annotation_position="top")
    fig.update_layout(title=f"Ergebnisse von {v['starts']} Zufallsstarts (grün: Vergleichswert)")
    fig.update_xaxes(title_text="Kosten der gerundeten Anordnung")
    fig.update_yaxes(title_text="Zahl der Starts")
    return _base(fig, legend_below=False)


# ----------------------------------------------------------------------------- Experiment 4: die Zahl der Starts

def build_best_curve(v: dict) -> go.Figure:
    """Die Kosten des besten von K Starts gegen K, mit dem Vergleichswert."""
    k = list(range(1, len(v["curve"]) + 1))
    fig = go.Figure(go.Scatter(x=k, y=list(v["curve"]), mode="lines+markers", name="bestes von K Starts", line=dict(color=C.COLORS["warn"], width=3, shape="hv"), marker=dict(size=5), hovertemplate="K = %{x}: %{y}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[k[0], k[-1]], y=[v["ref"], v["ref"]], mode="lines", name="Optimum" if v["exact"] else "Vergleichswert", line=dict(color=C.COLORS["best"], width=2, dash="dash"), hoverinfo="skip"))
    fig.update_layout(title="Die Zahl der Starts: das beste von K Starts")
    fig.update_xaxes(title_text="Zahl der Starts K")
    fig.update_yaxes(title_text="Kosten")
    return _base(fig)


def build_starts_bars(n: int, by_family: dict, median_of) -> go.Figure:
    """Je Familie: Anteil der Starts mit Optimum (Median über die Instanzen) bei der FAQ-Art gegen den exakten Erfolg eines Abstiegs."""
    fams = [f for f in C.FAMILIES if by_family.get(f)]
    x = [C.FAMILY_NAMES[f] for f in fams]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=[100 * median_of(by_family[f], "swap_success") for f in fams], name="steilster Abstieg (exakt gezählt)", marker_color=C.COLORS["other"]))
    fig.add_trace(go.Bar(x=x, y=[100 * median_of(by_family[f], "start_success") for f in fams], name="ein Start der FAQ-Art", marker_color=C.COLORS["warn"]))
    fig.update_layout(title=f"Anteil erfolgreicher Starts (Median, n = {n})", barmode="group")
    fig.update_yaxes(title_text="Anteil (%)", range=[0, 100])
    return _base(fig)


def build_best_hits(by_n: dict) -> go.Figure:
    """Je Familie und Größe: in wie vielen Instanzen findet das beste von 50 Starts das Optimum, und der Schwerpunkt als einziger Start (Anteil der Instanzen)."""
    fig = go.Figure()
    for n in sorted(by_n):
        fams = [f for f in C.FAMILIES if by_n[n].get(f)]
        fig.add_trace(go.Bar(x=[C.FAMILY_NAMES[f] for f in fams], y=[100 * sum(1 for r in by_n[n][f] if r["best_hit"]) / len(by_n[n][f]) for f in fams], name=f"bestes von {by_n[n][fams[0]][0]['starts']} Starts, n = {n}", marker_color=PALETTE.get(n, "#888")))
        fig.add_trace(go.Bar(x=[C.FAMILY_NAMES[f] for f in fams], y=[100 * sum(1 for r in by_n[n][f] if r["bary_hit"]) / len(by_n[n][f]) for f in fams], name=f"nur der Schwerpunkt, n = {n}", marker_color=PALETTE.get(n, "#888"), opacity=0.45))
    fig.update_layout(title="Anteil der Instanzen, in denen das Optimum gefunden wird", barmode="group")
    fig.update_yaxes(title_text="Instanzen (%)", range=[0, 100])
    return _base(fig)


# ----------------------------------------------------------------------------- Experiment 5: große Instanzen

def build_large(by_n: dict, median_of) -> go.Figure:
    """Je Familie: Median der Abweichung des besten von 20 Starts der FAQ-Art (und des Schwerpunkts) von der besten von 200 Abstiegen; ein Teilbild je Größe."""
    sizes = sorted(by_n)
    fig = make_subplots(rows=1, cols=len(sizes), subplot_titles=[f"n = {n}" for n in sizes], shared_yaxes=True)
    for col, n in enumerate(sizes, start=1):
        fams = [f for f in C.FAMILIES if by_n[n].get(f)]
        x = [C.FAMILY_NAMES[f] for f in fams]
        for key, name, color in (("gap_best", "bestes von 20 Starts", C.COLORS["warn"]), ("gap_median", "mittlerer Start", C.COLORS["other"]), ("gap_bary", "Schwerpunkt", C.COLORS["local"])):
            fig.add_trace(go.Bar(x=x, y=[100 * median_of(by_n[n][f], key) for f in fams], name=name, marker_color=color, showlegend=col == 1), row=1, col=col)
    fig.update_layout(barmode="group", title="FAQ-Art gegen die beste von 200 Abstiegen (Median der Abweichung)")
    fig.update_yaxes(title_text="Abweichung (%, positiv = schlechter)", rangemode="tozero", row=1, col=1)
    fig.update_xaxes(tickangle=-30)
    return _base(fig, legend_below=True)
