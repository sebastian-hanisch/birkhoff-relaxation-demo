"""PDF-Export des aktuellen Ergebnisses (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Sonderzeichen werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import bkr_constants as C
from bkr_evaluation import Settings, de, num

_REPLACE = {"→": "->", "–": "-", "—": "-", "−": "-", "≤": "<=", "≥": ">=", "·": "*", "×": "x", "Δ": "Delta", "↔": "<->", "„": '"', "“": '"', "²": "2", "³": "3", "⁻": "-", "¹": "1", "½": "1/2", "⁴": "4"}


def _s(text) -> str:
    out = str(text).replace("**", "").replace("`", "")
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(s: Settings, v: dict) -> bytes:
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Doppeltstochastische Relaxation"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    inst, cv = v["inst"], v["convex"]
    how = f"alle {v['perms']} Anordnungen aufgezählt" if v["exact"] else f"nicht aufgezählt ({v['perms']} Anordnungen), beste von {C.STARTS} Abstiegen als Vergleich"
    pdf.multi_cell(0, 5, _s(f"Familie {C.FAMILY_NAMES[s.family]}, {s.n} Bauteile, {inst.rows}x{inst.cols}-Raster, Seed {s.seed}; {how}. Eine Instanz, kein Befund über die Familie. Einfache Fassungen (Spektralverschiebung, FAQ-Art). Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    ref_name = "Optimum (aufgezählt)" if v["exact"] else "Beste gefundene Lösung (obere Schranke)"
    rows = [("Konvexe Schranke (CQ)", num(cv.cq)), ("Gilmore-Lawler-Schranke", num(v["glb"])), ("Projizierte Schranke", num(v["pb"])), (ref_name, v["ref"]), ("Lücke der konvexen Schranke", de(100 * v["gap_cq"], 1) + " %"),
            ("Rundung der konvexen Lösung (Kosten)", cv.rounded_cost), (f"Bestes von {v['starts']} Starts der FAQ-Art (Kosten)", v["best_cost"]), ("Lauf vom Schwerpunkt (Kosten)", v["bary"].cost),
            (f"Anteil der {v['starts']} Starts mit Vergleichswert", de(100 * v["start_success"], 1) + " %")]
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(100, 6, "Kennzahl", border=1)
    pdf.cell(85, 6, "Wert", border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for k, val in rows:
        pdf.cell(100, 5.5, _s(k), border=1)
        pdf.cell(85, 5.5, _s(val)[:60], border=1)
        pdf.ln()
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: birkhoff-relaxation-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
