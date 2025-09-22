# app.py — RCC Beam & Column Designer (Streamlit, single file)
# Run:
#   pip install streamlit matplotlib fpdf2
#   streamlit run app.py
# Notes: Educational prototype. Cross-check with IS 456:2000, IS 13920:2016, SP-16.

from dataclasses import dataclass
from typing import List, Tuple
from math import ceil, floor, sqrt, pi
import io
import tempfile

import streamlit as st
import matplotlib.pyplot as plt

# PDF helper (fpdf2)
try:
    from fpdf import FPDF
except Exception:
    from fpdf.fpdf import FPDF

# ---------------------------- Data models ----------------------------
@dataclass
class BeamInputs:
    b: float; d: float; cover: float; fck: float; fy: float
    mu: float; vu: float; tu: float; nu: float
    doubly: bool
    main_dia: int; comp_dia: int

@dataclass
class RowBars:
    label: str; count: int; dia: int; As_row: float

@dataclass
class BarLayout:
    rows: List[RowBars]; As_prov: float

@dataclass
class BeamResult:
    As_req: float; Asc_req: float; layout: BarLayout
    stirrup_dia: int; stirrup_s: float
    torsion_At: float; torsion_Al: float
    phiMn: float
    notes: List[str]; ductile_notes: List[str]

@dataclass
class ColumnInputs:
    b: float; h: float; cover: float; fck: float; fy: float
    pu: float; mux: float; muy: float; tension: bool

@dataclass
class ColumnResult:
    Ast_req: float; bars: int; bar_dia: int
    tie_dia: int; tie_s: float; util: float
    notes: List[str]; ductile_notes: List[str]

# ---------------------------- Core RCC logic ----------------------------

def area_of_bar(dia: int) -> float:
    return pi * dia * dia / 4.0


def solve_xu_for_moment(M, b, d, fck):
    low, high = 1e-9, max(1.0, 0.6 * d)
    for _ in range(60):
        mid = (low + high) * 0.5
        cap = 0.36 * fck * b * mid * (d - 0.42 * mid)
        if cap < M: low = mid
        else: high = mid
    return 0.5 * (low + high)


def tau_c_is456_approx(fck, pt):
    p = max(0.25, min(pt, 3.0))
    return 0.62 * (0.84 + 0.08 * sqrt(p))


def sp16_moment_limit_factor(fck, fy):
    return (0.125 if fy >= 500 else 0.138) * fck


def pack_beam_bars(b, cover, main_dia, comp_dia, As_req, Asc_req) -> BarLayout:
    clear_w = b - 2 * cover
    spacing_min = max(25.0, float(main_dia))
    def pack(As_need, dia):
        area = area_of_bar(dia)
        max_per_row = max(2, int(floor((clear_w + spacing_min) / (dia + spacing_min))))
        bars = max(2, int(ceil(As_need / area)))
        rows = 1
        while bars > max_per_row * rows:
            rows += 1
        return bars, rows
    bot_bars, bot_rows = pack(As_req, main_dia)
    top_bars, top_rows = (pack(Asc_req, comp_dia) if Asc_req > 0 else (0,0))
    rows: List[RowBars] = []
    area_bot = area_of_bar(main_dia)
    per_row_bot = max(1, int(ceil(bot_bars / bot_rows)))
    for i in range(bot_rows):
        rows.append(RowBars(f"Bottom R{i+1}", per_row_bot, main_dia, per_row_bot * area_bot))
    if top_rows > 0:
        area_top = area_of_bar(comp_dia)
        per_row_top = max(1, int(ceil(top_bars / top_rows)))
        for i in range(top_rows):
            rows.append(RowBars(f"Top R{i+1}", per_row_top, comp_dia, per_row_top * area_top))
    As_prov = sum(r.As_row for r in rows)
    return BarLayout(rows, As_prov)


def design_beam(inp: BeamInputs) -> BeamResult:
    b, d, fck, fy = inp.b, inp.d, inp.fck, inp.fy
    Mu, Tu, Vu, Nu = inp.mu * 1e6, inp.tu * 1e6, inp.vu * 1e3, inp.nu * 1e3
    notes = []

    xu_max_by_d = 0.46 if fy >= 500 else (0.48 if fy >= 415 else 0.53)
    xu_lim = xu_max_by_d * d

    def M_for_xu(xu): return 0.36 * fck * b * xu * (d - 0.42 * xu)
    M_lim = M_for_xu(xu_lim)

    overallD = d + inp.cover
    Mt = Tu * (1 + overallD / b)
    Me = Mu + Mt

    As_axial = (Nu / (0.87 * fy * 1e6)) if Nu > 0 else 0.0

    if Me <= M_lim:
        xu = solve_xu_for_moment(Me, b, d, fck)
        z = d - 0.42 * xu
        T = Me / z
        As_req = T / (0.87 * fy)
        Asc_req = 0.0
        notes.append("Singly reinforced governs")
    else:
        z_lim = d - 0.42 * xu_lim
        T_lim = M_lim / z_lim
        As_lim = T_lim / (0.87 * fy)
        dp = inp.cover + inp.comp_dia / 2.0
        lever = d - dp
        sigma_sc = min(0.87 * fy, 0.0035 * 2e5)
        Mr = Me - M_lim
        Asc_req = Mr / (sigma_sc * lever)
        As_add = Asc_req * (sigma_sc / (0.87 * fy))
        As_req = As_lim + As_add
        notes.append("Doubly reinforced (compression steel added)")

    As_req += As_axial
    if As_axial > 0: notes.append(f"Axial tension → +As ≈ {As_axial:.0f} mm²")

    Ve = Vu + 1.6 * Tu / b
    pt = As_req * 100.0 / (b * d)
    tc = tau_c_is456_approx(fck, pt)
    Vc = tc * b * d

    stirrup_s = 300.0; stirrup_dia = 8
    if Ve > Vc:
        Vs = Ve - Vc
        fy_sv = 415.0; legs = 2
        ast_per = legs * area_of_bar(stirrup_dia)
        stirrup_s = min(0.75 * d, max(100.0, 0.87 * fy_sv * ast_per * d / Vs))
        notes.append(f"Shear: 2-ϕ{stirrup_dia} @ {stirrup_s:.0f} mm")
    else:
        notes.append("Concrete carries shear (provide min links)")

    torsion_At = 0.0; torsion_Al = 0.0
    if Tu > 0:
        Ao = 0.85 * (b - 2 * inp.cover) * (overallD - 2 * inp.cover)
        fy_sv = 415.0
        At_over_s = Tu * 1e6 / (0.87 * fy_sv * 2 * Ao)
        torsion_At = At_over_s * stirrup_s
        z = 0.9 * d
        torsion_Al = Tu * 1e6 / (0.87 * fy * z)
        notes.append(f"Torsion: At≈{torsion_At:.0f} mm²/stirrup, ΔAs≈{torsion_Al:.0f} mm²")

    layout = pack_beam_bars(b, inp.cover, inp.main_dia, inp.comp_dia, As_req, Asc_req)

    k = sp16_moment_limit_factor(fck, fy)
    Mlim_est = k * b * d * d
    ductile = []
    if Me > Mlim_est:
        ductile.append("SP‑16 limit estimate exceeded → DR beam acceptable")
    pt_prov = layout.As_prov * 100.0 / (b * d)
    if pt_prov < 0.6: ductile.append("IS 13920: ≥0.6% total tension steel in hinge regions")
    s_max = min(150.0, 8 * float(inp.main_dia))
    if stirrup_s > s_max: ductile.append("IS 13920: stirrup spacing ≤ min(8db,150) near supports")
    ductile.append("Use 135° hooks, crossties within 2d of supports")

    phiMn = (Me / 1e6) / 1.5

    return BeamResult(As_req, Asc_req, layout, stirrup_dia, stirrup_s,
                      torsion_At, torsion_Al, phiMn, notes, ductile)


def design_column(inp: ColumnInputs) -> ColumnResult:
    b, D, fck, fy = inp.b, inp.h, inp.fck, inp.fy
    Pu, Mux, Muy = inp.pu * 1e3, inp.mux * 1e6, inp.muy * 1e6
    dp = inp.cover + 20.0
    notes = []

    def capacities(Ast_try: float):
        Ac = b * D - Ast_try
        Pu0 = 0.4 * fck * Ac + 0.67 * fy * Ast_try
        dx = D - dp; dy = b - dp
        zX = 0.8 * dx; zY = 0.8 * dy
        Mux1 = Pu0 * zX / 2.0; Muy1 = Pu0 * zY / 2.0
        return Pu0, Mux1, Muy1

    Ast = 0.02 * b * D
    alpha = 1.5; util = 2.0
    for _ in range(80):
        Pu0, Mux1, Muy1 = capacities(Ast)
        iax = (abs(Mux) / max(Mux1, 1.0)) ** alpha
        iay = (abs(Muy) / max(Muy1, 1.0)) ** alpha
        bresler = iax + iay
        axial = abs(Pu) / max(Pu0, 1.0)
        util = max(axial, bresler)
        if util <= 1.0: break
        Ast *= 1.05

    Pu0f, Mux1f, Muy1f = capacities(Ast)
    util = max(abs(Pu) / max(Pu0f, 1.0), (abs(Mux) / max(Mux1f, 1.0)) ** alpha + (abs(Muy) / max(Muy1f, 1.0)) ** alpha)

    bar_dia = 20
    area20 = area_of_bar(bar_dia)
    bars = max(8, int(ceil(Ast / area20)))
    minAst = 0.008 * b * D
    if bars * area20 < minAst: bars = int(ceil(minAst / area20))

    tie_dia = 8; tie_s = min(300.0, min(b, D) * 0.25)
    if util > 1.0: notes.append(f"Increase section/steel — utilization = {util:.2f}")
    ex = Mux / max(Pu, 1.0); ey = Muy / max(Pu, 1.0)
    notes.append(f"Eccentricities ex={ex:.1f} mm, ey={ey:.1f} mm")

    ductile = [
        "Hoop spacing ≤ min(8db, 0.25·min(b,D), 100mm) in hinge regions",
        "135° hooks & crossties; lap splices in constant moment zone",
    ]
    if Ast < 0.01 * b * D: ductile.append("Provide ≥1% longitudinal steel for ductility")

    return ColumnResult(bars * area20, bars, bar_dia, tie_dia, tie_s, util, notes, ductile)

# ---------------------------- Plotting ----------------------------

def draw_beam_section(inp: BeamInputs, layout: BarLayout):
    fig, ax = plt.subplots(figsize=(4, 3), dpi=150)
    ax.set_aspect('equal'); ax.axis('off')
    ax.add_patch(plt.Rectangle((0,0), inp.b, inp.d+inp.cover, fill=False, linewidth=2))
    ax.add_patch(plt.Rectangle((6,6), inp.b-12, inp.d+inp.cover-12, fill=False, linestyle='--'))
    y_bot = inp.d - 25; y_top = 25
    bw = inp.b - 40; left = 20
    for idx, r in enumerate([r for r in layout.rows if r.label.startswith('Bottom')]):
        n = r.count; gap = bw/(n+1)
        for i in range(n):
            x = left + (i+1)*gap
            ax.add_patch(plt.Circle((x, y_bot - idx*26), r.dia/2, fill=True))
    for idx, r in enumerate([r for r in layout.rows if r.label.startswith('Top')]):
        n = r.count; gap = bw/(n+1)
        for i in range(n):
            x = left + (i+1)*gap
            ax.add_patch(plt.Circle((x, y_top + idx*26), r.dia/2, fill=True))
    ax.set_xlim(-10, inp.b+10); ax.set_ylim(-10, inp.d+inp.cover+10)
    return fig


def draw_column_section(inp: ColumnInputs, res: ColumnResult):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    ax.set_aspect('equal'); ax.axis('off')
    b = inp.b; D = inp.h
    ax.add_patch(plt.Rectangle((0,0), b, D, fill=False, linewidth=2))
    ax.add_patch(plt.Rectangle((8,8), b-16, D-16, fill=False, linestyle='--'))
    per_side = max(2, res.bars // 4)
    gap = (b - 40) / (per_side - 1) if per_side > 1 else 0
    for i in range(per_side):
        x = 20 + i*gap
        for y in (20, D-20): ax.add_patch(plt.Circle((x, y), 8, fill=True))
    for i in range(1, per_side-1):
        y = 20 + i*gap
        for x in (20, b-20): ax.add_patch(plt.Circle((x, y), 8, fill=True))
    ax.set_xlim(-10, b+10); ax.set_ylim(-10, D+10)
    return fig

# ---------------------------- PDF builders ----------------------------

def make_beam_pdf(inp: BeamInputs, res: BeamResult, png_bytes: bytes) -> bytes:
    pdf = FPDF(unit="mm", format="A4"); pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "RCC Beam Design (prototype)", ln=1)
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 6, f"b={inp.b} mm, d={inp.d} mm, cover={inp.cover} mm, fck={inp.fck} MPa, fy={inp.fy} MPa", ln=1)
    pdf.cell(0, 6, f"Mu={inp.mu} kN·m, Vu={inp.vu} kN, Tu={inp.tu} kN·m, Nu={inp.nu} kN", ln=1)
    pdf.ln(2)
    pdf.cell(0, 6, f"As,req={res.As_req:.0f} mm²; Asc,req={res.Asc_req:.0f} mm²", ln=1)
    pdf.cell(0, 6, f"Stirrups: 2-ϕ{res.stirrup_dia} @ {res.stirrup_s:.0f} mm", ln=1)
    if res.torsion_At>0: pdf.cell(0,6,f"Torsion links≈{res.torsion_At:.0f} mm²/stirrup",ln=1)
    if res.torsion_Al>0: pdf.cell(0,6,f"Longitudinal torsion ΔAs≈{res.torsion_Al:.0f} mm²",ln=1)
    for row in res.layout.rows:
        pdf.cell(0, 6, f"• {row.label}: {row.count}–ϕ{row.dia} (As≈{row.As_row:.0f})", ln=1)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(png_bytes); tmp.flush(); pdf.image(tmp.name, x=130, y=20, w=70)
    return pdf.output(dest='S').encode('latin1')


def make_column_pdf(inp: ColumnInputs, res: ColumnResult, png_bytes: bytes) -> bytes:
    pdf = FPDF(unit="mm", format="A4"); pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "RCC Column Design (prototype)", ln=1)
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 6, f"b={inp.b} mm, D={inp.h} mm, cover={inp.cover} mm, fck={inp.fck} MPa, fy={inp.fy} MPa", ln=1)
    pdf.cell(0, 6, f"Pu={inp.pu} kN, Mux={inp.mux} kN·m, Muy={inp.muy} kN·m", ln=1)
    pdf.ln(2)
    pdf.cell(0, 6, f"Ast,req≈{res.Ast_req:.0f} mm²; Provide {res.bars}–ϕ{res.bar_dia}", ln=1)
    pdf.cell(0, 6, f"Ties ϕ{res.tie_dia} @ {res.tie_s:.0f} mm; Utilization≈{res.util:.2f}", ln=1)
    for n in res.notes: pdf.cell(0, 6, f"• {n}", ln=1)
    pdf.cell(0, 6, "IS 13920:", ln=1)
    for n in res.ductile_notes: pdf.cell(0, 6, f"• {n}", ln=1)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(png_bytes); tmp.flush(); pdf.image(tmp.name, x=130, y=20, w=70)
    return pdf.output(dest='S').encode('latin1')

# ---------------------------- Streamlit UI ----------------------------
#st.set_page_config(page_title="RCC Designer", layout="wide")
st.title("RCC Beam & Column Designer — Streamlit")
st.caption("Units: mm, kN, MPa. Prototype; verify with IS codes.")

tab_beam, tab_col, tab_about = st.tabs(["Beam", "Column", "About"])

with tab_beam:
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        b = st.number_input("b (mm)", 150.0, 2500.0, 300.0, step=10.0)
        d = st.number_input("d effective (mm)", 150.0, 3000.0, 500.0, step=10.0)
        cover = st.number_input("cover (mm)", 20.0, 75.0, 40.0, step=5.0)
        fck = st.number_input("fck (MPa)", 15.0, 80.0, 30.0, step=1.0)
        fy = st.number_input("fy (MPa)", 250.0, 600.0, 500.0, step=10.0)
    with col2:
        mu = st.number_input("Mu (kN·m)", 0.0, 10000.0, 150.0, step=5.0)
        vu = st.number_input("Vu (kN)", 0.0, 10000.0, 180.0, step=5.0)
        tu = st.number_input("Tu (kN·m)", 0.0, 5000.0, 0.0, step=1.0)
        nu = st.number_input("Nu (kN, +tension)", 0.0, 5000.0, 0.0, step=1.0)
        doubly = st.toggle("Doubly reinforced", value=True)
    with col3:
        main_dia = st.slider("Main bar dia (mm)", 12, 32, 20, step=4)
        comp_dia = st.slider("Compression bar dia (mm)", 12, 32, 16, step=4)
        do_calc_beam = st.button("Calculate Beam", use_container_width=True)

    if do_calc_beam:
        binp = BeamInputs(b, d, cover, fck, fy, mu, vu, tu, nu, doubly, main_dia, comp_dia)
        bres = design_beam(binp)
        left, right = st.columns([1,1])
        with left:
            st.subheader("Beam results")
            st.write({
                "As_req (mm²)": round(bres.As_req, 1),
                "Asc_req (mm²)": round(bres.Asc_req, 1),
                "Stirrups": f"2-ϕ{bres.stirrup_dia} @ {bres.stirrup_s:.0f} mm",
                "Torsion At (mm²/stirrup)": round(bres.torsion_At, 1),
                "Longitudinal torsion ΔAs (mm²)": round(bres.torsion_Al, 1),
                "ϕMn (kN·m)": round(bres.phiMn, 1),
                "Total As,prov (mm²)": round(bres.layout.As_prov, 1),
            })
            st.markdown("**Bar schedule**")
            for row in bres.layout.rows:
                st.write(f"• {row.label}: {row.count}–ϕ{row.dia} (As≈{row.As_row:.0f})")
            if bres.notes:
                st.markdown("**Notes**"); [st.write("• ", n) for n in bres.notes]
            if bres.ductile_notes:
                st.markdown("**IS 13920**"); [st.write("• ", n) for n in bres.ductile_notes]
        with right:
            fig = draw_beam_section(binp, bres.layout)
            st.pyplot(fig, clear_figure=True)
            img_buf = io.BytesIO(); fig.savefig(img_buf, format='png', bbox_inches='tight', dpi=200); img_buf.seek(0)
            if st.button("Export Beam PDF", use_container_width=True):
                pdf_bytes = make_beam_pdf(binp, bres, img_buf.getvalue())
                st.download_button("Download Beam_Design.pdf", data=pdf_bytes, file_name="Beam_Design.pdf", mime="application/pdf")

with tab_col:
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        cb = st.number_input("b (mm)", 150.0, 2500.0, 400.0, step=10.0, key='cb')
        ch = st.number_input("D (mm)", 150.0, 2500.0, 400.0, step=10.0, key='ch')
        ccover = st.number_input("cover (mm)", 20.0, 75.0, 40.0, step=5.0, key='ccover')
    with col2:
        cfck = st.number_input("fck (MPa)", 15.0, 80.0, 30.0, step=1.0, key='cfck')
        cfy = st.number_input("fy (MPa)", 250.0, 600.0, 500.0, step=10.0, key='cfy')
        cpu = st.number_input("Pu (kN, +comp)", 0.0, 10000.0, 1200.0, step=10.0, key='cpu')
    with col3:
        cmux = st.number_input("Mux (kN·m)", 0.0, 5000.0, 120.0, step=5.0, key='cmux')
        cmuy = st.number_input("Muy (kN·m)", 0.0, 5000.0, 80.0, step=5.0, key='cmuy')
        do_calc_col = st.button("Calculate Column", use_container_width=True, key='colbtn')

    if do_calc_col:
        cinp = ColumnInputs(cb, ch, ccover, cfck, cfy, cpu, cmux, cmuy, False)
        cres = design_column(cinp)
        left, right = st.columns([1,1])
        with left:
            st.subheader("Column results")
            st.write({
                "Ast_req (mm²)": round(cres.Ast_req, 1),
                "Provide": f"{cres.bars}–ϕ{cres.bar_dia}",
                "Ties": f"ϕ{cres.tie_dia} @ {cres.tie_s:.0f} mm",
                "Utilization": round(cres.util, 2),
            })
            if cres.notes:
                st.markdown("**Notes**"); [st.write("• ", n) for n in cres.notes]
            if cres.ductile_notes:
                st.markdown("**IS 13920**"); [st.write("• ", n) for n in cres.ductile_notes]
        with right:
            fig = draw_column_section(cinp, cres)
            st.pyplot(fig, clear_figure=True)
            img_buf = io.BytesIO(); fig.savefig(img_buf, format='png', bbox_inches='tight', dpi=200); img_buf.seek(0)
            if st.button("Export Column PDF", use_container_width=True):
                pdf_bytes = make_column_pdf(cinp, cres, img_buf.getvalue())
                st.download_button("Download Column_Design.pdf", data=pdf_bytes, file_name="Column_Design.pdf", mime="application/pdf")

with tab_about:
    st.write("This webapp mirrors the desktop prototype: beams (singly/doubly with bending, shear, torsion, axial tension add-on) and columns (Bresler biaxial compression/tension). Bar scheduler auto-packs layers. PDF export embeds the section sketches.")
    st.warning("Approximate SP-16 and IS 13920 checks are included as helpers, NOT as substitutes for code design.")
