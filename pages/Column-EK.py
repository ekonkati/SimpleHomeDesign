# app.py — RCC Column (Biaxial) Design Canvas
# Single-page, narration-rich, Plotly visuals, printable
# Assumptions: IS 456:2000 + IS 13920:2016 style checks (switchable)
# Units: N, mm, MPa, kN·m (displayed clearly where relevant)

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="RCC Column Designer", layout="wide")

# ------------------------- STYLES (Canvas + Print) -------------------------
st.markdown("""
<style>
/* Hide Streamlit chrome when printing */
@media print {
  header, .stToolbar, .stAppDeployButton, .stActionButton, .stDownloadButton, footer, .stSidebar { display: none !important; }
  .block-container { padding: 0.6cm 1.0cm !important; max-width: 100% !important; }
  .stTabs [data-baseweb="tab-list"] { display:none; }
  .print-break { page-break-before: always; }
}

/* Highlighted control wrapper */
.highlight {
  background: #fffbe6;   /* soft yellow */
  border: 1px solid #ffe58f;
  border-radius: 12px;
  padding: 10px 12px;
  margin: 8px 0 14px 0;
}

/* Slight background for inputs within highlights */
.highlight .stSelectbox, .highlight .stNumberInput, .highlight .stSlider, .highlight .stTextInput {
  background: #fffceb !important;
}

/* Tighten Plotly figure margins for printing */
.js-plotly-plot, .plotly, .user-select-none {
  margin: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ------------------------- Helpers & Data Structures ------------------------
@dataclass
class Section:
    b: float   # mm
    D: float   # mm
    cover: float  # mm (to tie)
    bars: List[Tuple[float,float,float]]  # (x, y, dia) bar centers in mm
    Ic_x: float  # mm^4 about x (bending about x => compression top/bottom)
    Ic_y: float  # mm^4 about y

def Ec_from_fck(fck: float) -> float:
    # IS 456: Ec = 5000 * sqrt(fck) (MPa -> N/mm^2)
    return 5000.0 * math.sqrt(fck)

def Ic_rect(b: float, D: float) -> Tuple[float,float]:
    # second moment of area about centroidal axes
    Ic_x = (b * D**3) / 12.0
    Ic_y = (D * b**3) / 12.0
    return Ic_x, Ic_y

def generate_bars_rect(b: float, D: float, cov: float, nx: int, ny: int, dia: float) -> List[Tuple[float,float,float]]:
    """
    Generate a symmetric grid of bars along the perimeter: nx per long side, ny per short side.
    Bars are placed on a rectangle at cover line; min 4 bars ensured.
    """
    bars = []
    # guard
    nx = max(nx, 2); ny = max(ny, 2)
    # perimeter rectangle for bar line:
    x0, x1 = cov, b - cov
    y0, y1 = cov, D - cov
    # top / bottom (nx bars each)
    for i in range(nx):
        x = np.interp(i, [0, nx-1], [x0, x1]) if nx>1 else (x0+x1)/2
        bars.append((x, y1, dia))  # top
        bars.append((x, y0, dia))  # bottom
    # left / right (ny-2 interior to avoid corners duplication)
    for j in range(1, ny-1):
        y = np.interp(j, [0, ny-1], [y0, y1]) if ny>1 else (y0+y1)/2
        bars.append((x0, y, dia))  # left
        bars.append((x1, y, dia))  # right
    # deduplicate very close bars (corners may repeat)
    uniq = []
    eps = 1e-6
    for x,y,d in bars:
        if not any(abs(x-x2)<eps and abs(y-y2)<eps for x2,y2,_ in uniq):
            uniq.append((x,y,d))
    return uniq

def long_bar_diameter_list(bars: List[Tuple[float,float,float]]) -> List[float]:
    return [d for *_ , d in bars]

def min_ecc(storey_clear: float, dim: float) -> float:
    # e_min = max( l/500 + D/30, 20 mm )
    return max(storey_clear/500.0 + dim/30.0, 20.0)

def k_factor_from_restraint(restraint: str) -> float:
    # Simple mapping for effective length factor (indicative)
    return {
        "Fixed-Fixed": 0.65,
        "Fixed-Pinned": 0.80,
        "Pinned-Pinned": 1.00,
        "Fixed-Free (cantilever)": 2.10
    }.get(restraint, 1.0)

def moment_magnifier(Pu: float, le: float, Ec: float, Ic: float, Cm: float=0.85, sway: bool=False) -> float:
    """
    Simple δ magnifier using Euler load; for teaching and quick design.
    For sway frames, reduce Pcr via factor; for non-sway use Cm ~0.85.
    """
    # Euler critical load (N): π^2 E I / le^2
    Pcr = (math.pi**2 * Ec * Ic) / (le**2 + 1e-9)
    # Safety: avoid >0.95 Pcr
    ratio = min(Pu / max(Pcr, 1.0), 0.95)
    if sway:
        # a crude reduction for sway (more sensitive)
        ratio = min(ratio * 1.15, 0.95)
    delta = 1.0 / max(1.0 - ratio, 0.05)
    # apply Cm
    return max(delta * Cm, 1.0)

def bresler_biaxial_interaction(Pu, Mux, Muy, P0x, P0y, Puz):
    """
    Bresler reciprocal interaction: (Mux/Mux0) + (Muy/Muy0) <= 1 for given Pu
    Mux0, Muy0 reduce with Pu by linear rule to zero at squash load Puz.
    P0x, P0y are uniaxial capacities at zero axial (approx).
    """
    # Reduce zero-axial capacities with Pu
    factor = max(0.0001, 1.0 - Pu / max(Puz, 1.0))
    Mux0 = P0x * factor
    Muy0 = P0y * factor
    lhs = (abs(Mux) / max(Mux0, 1e-6)) + (abs(Muy) / max(Muy0, 1e-6))
    return lhs, Mux0, Muy0

def axial_squash_load(fck, b, D, rho_long, fy):
    """
    Very rough squash load: 0.4 fck Ac + 0.67 fy Asc (N) (teaching-friendly)
    Ac in mm², Asc = rho_long * Ac_gross
    """
    Ac = b * D
    Asc = rho_long * Ac
    return 0.4 * fck * Ac + 0.67 * fy * Asc

def approx_uniaxial_M0(fck, fy, b, D, Asc, axis='x'):
    """
    Very rough M capacity at zero axial (kNm) using rectangular stress block idea.
    For teaching / plotting envelopes only.
    """
    # take an equivalent lever arm ≈ 0.9*D (x-axis bend) or 0.9*b (y-axis bend)
    lever = 0.9 * (D if axis=='x' else b)
    # assume tension steel governs at zero axial
    Ts = 0.87 * fy * Asc  # N
    M = Ts * lever  # N·mm
    return M / 1e6    # kN·m

def shear_capacity_check(b, D, Pu, fck, fy, sv, tie_dia):
    """
    Simple IS 456 style: tau_v = Vu/(b*d); tau_c from fck & p_t (simplified); Vus from shear reinforcement
    This is indicative; a proper design should compute tau_c via table (p_t dependent).
    """
    d = 0.9 * D
    # Assume nominal Vu from unbalanced moments (advisory): Vu = 0.1*(Mux/D + Muy/b) converted to N
    # We'll let user set Vu directly later if needed; here keep a placeholder
    Vu = 0.0  # N (let zero unless user supplies)
    tau_v = Vu / (b * d + 1e-9)
    # conservative tau_c as 0.28*sqrt(fck) MPa -> N/mm²
    tau_c = 0.28 * math.sqrt(fck)
    # Shear reinforcement capacity per spacing
    Asv = math.pi * (tie_dia**2) / 4.0  # one leg (mm²); assume 2 legs
    Vus = 0.87 * fy * (2*Asv) * d / max(sv, 1.0)  # N
    return tau_v, tau_c, Vu, Vus

def plotly_section(section: Section, tie_dia: float, tie_spacing: float):
    b, D, cov = section.b, section.D, section.cover
    fig = go.Figure()
    # concrete
    fig.add_shape(type="rect", x0=0, y0=0, x1=b, y1=D,
                  line=dict(width=1), fillcolor="rgba(200,200,200,0.35)")
    # core (to tie)
    fig.add_shape(type="rect", x0=cov, y0=cov, x1=b-cov, y1=D-cov,
                  line=dict(dash="dot"))
    # bars
    xs, ys, txt = [], [], []
    for i,(x,y,d) in enumerate(section.bars, start=1):
        xs.append(x); ys.append(D - y)  # flip for “y up”
        txt.append(f"Bar {i}<br>Ø{int(d)} mm<br>(x={x:.0f}, y={y:.0f})")
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers",
                             marker=dict(size=[max(6, d*0.6) for *_,d in section.bars]),
                             hovertext=txt, hoverinfo="text", name="Bars"))
    # tie label
    fig.add_annotation(x=b-10, y=10, text=f"Ties: Ø{int(tie_dia)} @ {int(tie_spacing)}",
                       showarrow=False, xanchor="right", yanchor="bottom")
    fig.update_yaxes(scaleanchor="x", scaleratio=1, title="y (mm)")
    fig.update_xaxes(title="x (mm)")
    fig.update_layout(height=420, margin=dict(l=10,r=10,t=40,b=10),
                      title="Column Cross-Section (Plotly)",
                      showlegend=False)
    return fig

def plotly_pm(P, M, Pu, Mu, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[p/1e3 for p in P], y=[m/1e6 for m in M], mode="lines", name="Capacity"))
    fig.add_trace(go.Scatter(x=[Pu/1e3], y=[abs(Mu)/1e6], mode="markers", name="Demand"))
    fig.update_layout(title=title, xaxis_title="P (kN)", yaxis_title="M (kN·m)",
                      height=420, margin=dict(l=10,r=10,t=40,b=10))
    return fig

# ------------------------- UI: One-Canvas Tabs ------------------------------
T1, T2, T3, T4, T5, T6, T7 = st.tabs([
    "Inputs", "Slenderness", "Moments", "Interaction", "Shear", "Detailing", "Report"
])

# ------------------------- T1: Inputs --------------------------------------
with T1:
    st.subheader("1) Geometry, Materials, Loads")
    colA, colB, colC, colD = st.columns([1.2,1.2,1.2,1.2])
    with colA:
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        b = st.number_input("Width b (mm)", 200.0, 3000.0, 400.0, 10.0)
        D = st.number_input("Depth D (mm)", 200.0, 3000.0, 600.0, 10.0)
        cover = st.number_input("Clear cover to tie (mm)", 25.0, 80.0, 40.0, 1.0)
        st.markdown('</div>', unsafe_allow_html=True)

    with colB:
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        fck = st.selectbox("Concrete grade fck (MPa)", [20,25,30,35,40,45,50], index=2)
        fy = st.selectbox("Steel fy (MPa)", [415, 500], index=1)
        ductile = st.checkbox("Ductile detailing (IS 13920)")
        st.markdown('</div>', unsafe_allow_html=True)

    with colC:
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        Pu = st.number_input("Factored axial load Pu (kN, +compression)", -5000.0, 15000.0, 2000.0, 10.0) * 1e3
        Mux = st.number_input("Factored Mux (kN·m) about x", -5000.0, 5000.0, 120.0, 1.0) * 1e6
        Muy = st.number_input("Factored Muy (kN·m) about y", -5000.0, 5000.0, 80.0, 1.0) * 1e6
        st.markdown('</div>', unsafe_allow_html=True)

    with colD:
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        Lx = st.number_input("Clear height in x-plane lx (mm)", 1000.0, 6000.0, 3000.0, 10.0)
        Ly = st.number_input("Clear height in y-plane ly (mm)", 1000.0, 6000.0, 3000.0, 10.0)
        restraint = st.selectbox("End restraint (k-factor)", ["Fixed-Fixed","Fixed-Pinned","Pinned-Pinned","Fixed-Free (cantilever)"], index=0)
        sway_frame = st.selectbox("Frame type", ["Non-sway","Sway"], index=0) == "Sway"
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("—")
    st.subheader("2) Longitudinal Bars (Perimeter Grid)")
    c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1])
    with c1:
        nx = st.number_input("Bars per long side (nx)", 2, 8, 3, 1)
    with c2:
        ny = st.number_input("Bars per short side (ny)", 2, 8, 3, 1)
    with c3:
        dia_bar = st.selectbox("Bar dia (mm)", [12,16,20,25,28,32], index=1)
    with c4:
        tie_dia = st.selectbox("Tie dia (mm)", [6,8,10,12], index=1)
    with c5:
        tie_spacing = st.number_input("Tie spacing sv (mm)", 50.0, 300.0, 150.0, 5.0)

    bars = generate_bars_rect(b, D, cover, nx, ny, float(dia_bar))
    Ic_x, Ic_y = Ic_rect(b, D)
    section = Section(b=b, D=D, cover=cover, bars=bars, Ic_x=Ic_x, Ic_y=Ic_y)

    st.plotly_chart(plotly_section(section, float(tie_dia), float(tie_spacing)), use_container_width=True)

# ------------------------- T2: Slenderness ---------------------------------
with T2:
    st.subheader("Slenderness & Effective Length")
    st.markdown('<div class="highlight">', unsafe_allow_html=True)
    k = k_factor_from_restraint(restraint)
    le_x = k * Lx
    le_y = k * Ly
    st.write(f"**Effective length factors** k = {k:.2f} → le,x = {le_x:.0f} mm, le,y = {le_y:.0f} mm.")
    short_x = (le_x / (D)) <= 12.0
    short_y = (le_y / (b)) <= 12.0
    st.write(f"Short/Slender check: le/D = {le_x/D:.1f} → **{'Short' if short_x else 'Slender'}** about x; "
             f"le/b = {le_y/b:.1f} → **{'Short' if short_y else 'Slender'}** about y.")
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("Show calculation details (Slenderness)", expanded=False):
        st.latex(r"l_e = k \cdot l_{clear};\quad \text{compare } \frac{l_e}{D}\text{ and }\frac{l_e}{b} \text{ with 12 (indicative).}")

# ------------------------- T3: Moments (min e + magnifier) -----------------
with T3:
    st.subheader("Minimum Eccentricity & Magnified Moments")

    st.markdown('<div class="highlight">', unsafe_allow_html=True)
    Ec = Ec_from_fck(float(fck))
    emin_x = min_ecc(Lx, D)
    emin_y = min_ecc(Ly, b)
    st.write(f"Minimum eccentricity (IS 456): e_min,x = {emin_x:.0f} mm; e_min,y = {emin_y:.0f} mm.")

    # bump base moments with e_min * Pu
    Mux_base = max(abs(Mux), abs(Pu * emin_x))
    Muy_base = max(abs(Muy), abs(Pu * emin_y))

    # magnifiers
    delta_x = 1.0 if short_x else moment_magnifier(Pu, le_x, Ec, section.Ic_x, Cm=0.85, sway=sway_frame)
    delta_y = 1.0 if short_y else moment_magnifier(Pu, le_y, Ec, section.Ic_y, Cm=0.85, sway=sway_frame)

    Mux_eff = Mux_base * delta_x
    Muy_eff = Muy_base * delta_y

    st.write(f"Magnifiers: δx = {delta_x:.2f}, δy = {delta_y:.2f} → **Mux,eff** = {Mux_eff/1e6:.1f} kN·m, "
             f"**Muy,eff** = {Muy_eff/1e6:.1f} kN·m.")
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("Show calculation details (δ)", expanded=False):
        st.latex(r"""
        E_c = 5000\sqrt{f_{ck}};\quad
        P_{cr} = \frac{\pi^2 E_c I_c}{l_e^2};\quad
        \delta = \frac{1}{1 - P_u/P_{cr}} \times C_m
        """)

# ------------------------- T4: Interaction (Bresler) -----------------------
with T4:
    st.subheader("Biaxial Interaction (Bresler-style)")

    # steel ratio (approx): based on total bar area over gross
    Asc = sum([math.pi*(d**2)/4 for *_,d in section.bars])
    Acg = b * D
    rho = Asc / max(Acg, 1.0)

    Puz = axial_squash_load(float(fck), b, D, rho, float(fy))  # N
    # zero-axial M capacities (very rough, kNm)
    M0x_kNm = approx_uniaxial_M0(float(fck), float(fy), b, D, Asc, axis='x')
    M0y_kNm = approx_uniaxial_M0(float(fck), float(fy), b, D, Asc, axis='y')

    lhs, Mux0, Muy0 = bresler_biaxial_interaction(Pu, Mux_eff, Muy_eff,
                                                  P0x=M0x_kNm, P0y=M0y_kNm, Puz=Puz)
    st.write(f"Axial squash (approx): **Puz** ≈ {Puz/1e3:.0f} kN.")
    st.write(f"Zero-axial uniaxial capacities: **M0x**≈{M0x_kNm:.0f} kN·m, **M0y**≈{M0y_kNm:.0f} kN·m.")
    st.write(f"Bresler check: (Mux/Mux0) + (Muy/Muy0) = **{lhs:.2f}** → "
             f"{'✅ Safe (≤1.0)' if lhs<=1.0 else '❌ NG (>1.0)'} "
             f"with Mux0≈{Mux0:.0f} kN·m, Muy0≈{Muy0:.0f} kN·m.")

    # quick P–M curves (advisory)
    Px = np.linspace(1e3, max(Puz*0.99, 1e3), 120)  # N
    # assume Mx capacity reduces linearly with Pu (teaching curve)
    Mx = (M0x_kNm * (1 - Px / Puz)).clip(min=0) * 1e6  # Nmm
    Py = Px.copy()
    My = (M0y_kNm * (1 - Py / Puz)).clip(min=0) * 1e6

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plotly_pm(Px, Mx, Pu, Mux_eff, "P–Mx (advisory envelope)"), use_container_width=True)
    with c2:
        st.plotly_chart(plotly_pm(Py, My, Pu, Muy_eff, "P–My (advisory envelope)"), use_container_width=True)

    with st.expander("Notes on interaction model", expanded=False):
        st.write(
            "This uses a reciprocal (Bresler-like) approximation for quick design and visualization. "
            "For final design, a strain-compatibility PM surface should be computed with cracked section properties "
            "and detailed neutral-axis search (not included here for brevity)."
        )

# ------------------------- T5: Shear (advisory) ----------------------------
with T5:
    st.subheader("Shear Check (Advisory)")
    st.markdown('<div class="highlight">', unsafe_allow_html=True)
    user_Vu_kN = st.number_input("Optional: Input factored shear Vu (kN) for evaluation", 0.0, 5000.0, 0.0, 1.0)
    st.markdown('</div>', unsafe_allow_html=True)
    # override placeholder Vu if user provides
    def shear_eval(Vu_kN):
        d = 0.9 * D
        tau_v = (Vu_kN*1e3) / (b * d + 1e-9)
        tau_c = 0.28 * math.sqrt(float(fck))
        Asv = math.pi * (float(tie_dia)**2) / 4.0
        Vus = 0.87 * float(fy) * (2*Asv) * d / max(float(tie_spacing), 1.0) / 1e3  # kN
        return tau_v, tau_c, Vus, d
    tau_v, tau_c, Vus_kN, d = shear_eval(user_Vu_kN)

    st.write(f"Depth d ≈ {d:.0f} mm; τ_v = {tau_v:.3f} N/mm² vs τ_c ≈ {tau_c:.3f} N/mm² "
             f"(very conservative). Shear steel contribution Vus ≈ {Vus_kN:.1f} kN.")
    st.info("For code-grade column shear in ductile frames, capacity-compatible shear "
            "based on probable end moments is required; ensure end regions have closed hoops with 135° hooks.")

# ------------------------- T6: Detailing (IS 13920 aide) -------------------
with T6:
    st.subheader("Detailing Aide (Ties, Confinement, Splices)")

    st.markdown('<div class="highlight">', unsafe_allow_html=True)
    db_list = long_bar_diameter_list(section.bars)
    db_long = max(db_list) if db_list else float(dia_bar)
    lo = max(D, b, Lx/6.0)   # plastic hinge length proxy
    s_end = min(6*db_long, 100.0)
    s_rest = min(8*db_long, 150.0)
    st.write(f"Suggest: Confining length l₀ ≈ max(D, b, L_clear/6) ≈ **{lo:.0f} mm** at both ends.")
    st.write(f"Hoop spacing: **≤ {s_end:.0f} mm** within l₀; elsewhere **≤ {s_rest:.0f} mm**. "
             f"Use 135° hooks, crossties to capture all core corners.")
    st.write(f"Lap splices (if any) in middle half of member height; avoid splices in l₀; "
             f"stagger laps; densify hoops over lap length.")
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("Bar schedule (quick)", expanded=False):
        total_bars = len(section.bars)
        st.write(f"Bars: {total_bars} nos Ø{int(dia_bar)}. Tie: Ø{int(tie_dia)} @ {int(tie_spacing)} mm.")
        st.write("Provide corner bars on all four corners; ensure tie legs/crossties anchor corner & side bars.")

# ------------------------- T7: Report (Printable) --------------------------
with T7:
    st.subheader("Submission-Ready Report")
    st.caption("Use the button below, or Ctrl/Cmd + P. This page prints as a clean multi-page PDF.")
    st.button("🖨️ Print / Save as PDF")

    st.markdown("---")
    st.write("### A. Inputs")
    i1, i2, i3 = st.columns([1.2,1.2,1.2])
    with i1:
        st.write(f"**b × D:** {b:.0f} × {D:.0f} mm  \n**Cover:** {cover:.0f} mm")
        st.write(f"**fck / fy:** {int(fck)} / {int(fy)} MPa  \n**Ductile:** {'Yes' if ductile else 'No'}")
    with i2:
        st.write(f"**Pu:** {Pu/1e3:.0f} kN  \n**Mux:** {Mux/1e6:.1f} kN·m  \n**Muy:** {Muy/1e6:.1f} kN·m")
        st.write(f"**Bars:** {len(bars)} nos Ø{int(dia_bar)}")
    with i3:
        st.write(f"**l_clear x/y:** {Lx:.0f} / {Ly:.0f} mm  \n**k:** {k:.2f} ({restraint})  \n**Frame:** {'Sway' if sway_frame else 'Non-sway'}")

    st.markdown("---")
    st.write("### B. Slenderness & Effective Length")
    st.write(f"le,x = {le_x:.0f} mm (→ {'Short' if short_x else 'Slender'}), le,y = {le_y:.0f} mm (→ {'Short' if short_y else 'Slender'}).")

    st.write("### C. Minimum Eccentricity & Magnified Moments")
    st.write(f"e_min,x = {emin_x:.0f} mm; e_min,y = {emin_y:.0f} mm. "
             f"δx = {delta_x:.2f}; δy = {delta_y:.2f}.")
    st.write(f"**Mux,eff** = {Mux_eff/1e6:.1f} kN·m; **Muy,eff** = {Muy_eff/1e6:.1f} kN·m.")

    st.write("### D. Biaxial Interaction (Advisory)")
    st.write(f"Puz ≈ {Puz/1e3:.0f} kN; M0x≈{M0x_kNm:.0f} kN·m; M0y≈{M0y_kNm:.0f} kN·m.")
    st.write(f"(Mux/Mux0) + (Muy/Muy0) = **{lhs:.2f}** → "
             f"{'✅ Safe (≤1.0)' if lhs<=1.0 else '❌ Not OK (>1.0) — increase section/rebars or reduce demand'}.")

    st.write("### E. Shear (Advisory)")
    st.write(f"d ≈ {d:.0f} mm; τ_v = {tau_v:.3f} N/mm²; τ_c ≈ {tau_c:.3f} N/mm²; Vus ≈ {Vus_kN:.1f} kN "
             f"(Ø{int(tie_dia)} @ {int(tie_spacing)}).")

    st.write("### F. Detailing Notes (IS 13920 oriented)")
    st.write(f"Confinement over l₀≈{lo:.0f} mm at ≤{s_end:.0f} mm; elsewhere ≤{s_rest:.0f} mm; "
             f"135° hooks; crossties to core corners; splice in middle half only (not in l₀).")

    st.markdown("### G. Figures")
    st.plotly_chart(plotly_section(section, float(tie_dia), float(tie_spacing)), use_container_width=True)
    st.plotly_chart(plotly_pm(Px, Mx, Pu, Mux_eff, "P–Mx (advisory)"), use_container_width=True)
    st.plotly_chart(plotly_pm(Py, My, Pu, Muy_eff, "P–My (advisory)"), use_container_width=True)
    st.markdown('<div class="print-break"></div>', unsafe_allow_html=True)

    st.caption(
        "Notes: This app shows a clear, narrated workflow with teaching-grade approximations for PM interaction "
        "and shear. For final submissions, adopt a strain-compatibility PM surface and capacity-compatible shear "
        "per code, with detailed clause citations in your calculation sheets."
    )
