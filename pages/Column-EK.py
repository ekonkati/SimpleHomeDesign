import math
import json
import base64
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------
# 1. CONSTANTS, UTILITIES, and SECTION DATACLASS
# -----------------------------
ES = 200000.0  # MPa (N/mm2)
EPS_CU = 0.0035  # Ultimate concrete compressive strain
FY_DEFAULT = 500.0  # MPa (from original code)

def Ec_from_fck(fck: float) -> float:
    """IS 456: Ec = 5000 * sqrt(fck) (MPa)."""
    [cite_start]return 5000.0 * math.sqrt(max(fck, 1e-6)) [cite: 2]

def bar_area(dia_mm: float) -> float:
    """Area of a single circular bar (mm^2)."""
    [cite_start]return math.pi * (dia_mm ** 2) / 4.0 [cite: 4]

def kN(value: float) -> str:
    """Formats N to kN (1 decimal place)."""
    return f"{value / 1e3:.1f}"

def kNm(value: float) -> str:
    """Formats N-mm to kN-m (1 decimal place)."""
    return f"{value / 1e6:.1f}"

@dataclass
class Section:
    b: float  # mm (width along local x-axis)
    D: float  # mm (depth along local y-axis)
    cover: float  # mm (clear cover to main bars)
    bars: List[Tuple[float, float, float]]  # list of (x, y, dia_mm)
    tie_dia: float = 8.0 # Added default

    @property
    def Ag(self) -> float:
        [cite_start]return self.b * self.D [cite: 2]

    @property
    def Ic_x(self) -> float:
        # About local x-axis (neutral axis // to x, bending about x means compression at top/bottom along y)
        [cite_start]return self.b * self.D**3 / 12.0 [cite: 3]

    @property
    def Ic_y(self) -> float:
        # About local y-axis (neutral axis // to y, bending about y means compression at left/right along x)
        [cite_start]return self.D * self.b**3 / 12.0 [cite: 3]

    @property
    def rx(self) -> float:
        [cite_start]return math.sqrt(self.Ic_x / self.Ag) [cite: 4]

    @property
    def ry(self) -> float:
        [cite_start]return math.sqrt(self.Ic_y / self.Ag) [cite: 4]
    
    @property
    def As_long(self) -> float:
        """Total longitudinal steel area (mm^2)."""
        return sum(bar_area(dia) for _, _, dia in self.bars)

# -----------------------------
# 2. CORE ENGINEERING LOGIC (FROM column.txt)
# -----------------------------

def generate_rectangular_bar_layout(b: float, D: float, cover: float, 
                                    n_top: int, n_bot: int, n_left: int, n_right: int, 
                                    dia_top: float, dia_bot: float, dia_side: float) -> List[Tuple[float, float, float]]:
    """
    Returns list of (x, y, dia) for a rectangular tied column. 
    [cite_start]Coordinate system: origin at top-left corner (x right, y down)[cite: 6].
    """
    bars = []
    # helper for linear spacing
    [cite_start]def linspace(a, c, n): [cite: 7]
        if n == 1:
            return [0.5 * (a + c)]
        return [a + i * (c - a) / (n - 1) for i in range(n)]

    # Top row
    [cite_start]y_top = cover + dia_top / 2.0 [cite: 7]
    [cite_start]x_span_left = cover + dia_side / 2.0 [cite: 7]
    [cite_start]x_span_right = b - (cover + dia_side / 2.0) [cite: 8]
    if n_top > 0:
        for x in linspace(x_span_left, x_span_right, n_top):
            [cite_start]bars.append((x, y_top, dia_top)) [cite: 8]

    # Bottom row
    [cite_start]y_bot = D - (cover + dia_bot / 2.0) [cite: 8]
    if n_bot > 0:
        for x in linspace(x_span_left, x_span_right, n_bot):
            [cite_start]bars.append((x, y_bot, dia_bot)) [cite: 8]

    # Left column (excluding top/bottom corners to avoid duplicates)
    [cite_start]x_left = cover + dia_side / 2.0 [cite: 9]
    [cite_start]y_span_top = cover + dia_top / 2.0 [cite: 9]
    [cite_start]y_span_bot = D - (cover + dia_bot / 2.0) [cite: 9]
    if n_left > 2:
        for y in linspace(y_span_top, y_span_bot, n_left)[1:-1]:
            [cite_start]bars.append((x_left, y, dia_side)) [cite: 9]

    # Right column
    [cite_start]x_right = b - (cover + dia_side / 2.0) [cite: 9]
    if n_right > 2:
        for y in linspace(y_span_top, y_span_bot, n_right)[1:-1]:
            [cite_start]bars.append((x_right, y, dia_side)) [cite: 10]

    return bars

def uniaxial_capacity_Mu_for_Pu(section: Section, fck: float, fy: float, Pu: float, axis: str) -> float:
    """
    Compute ultimate uniaxial moment capacity Mu,lim for a given factored axial Pu (N) and bending axis 
    [cite_start]using binary search on the neutral axis depth 'c'[cite: 25, 26].
    """
    b, D = section.b, section.D
    
    # Iterate neutral axis depth c (mm) to satisfy axial equilibrium for given Pu.
    [cite_start]c_min = 0.05 * (D if axis == 'x' else b) [cite: 18]
    [cite_start]c_max = 1.50 * (D if axis == 'x' else b) [cite: 18]

    def forces_and_moment(c: float):
        # Concrete block contribution
        if axis == 'x':
            [cite_start]xu = min(c, D)  # limit to section depth [cite: 19]
            [cite_start]Cc = 0.36 * fck * b * xu  # N/mm2 * mm * mm = N [cite: 19]
            [cite_start]arm_Cc = (D / 2.0) - (0.42 * xu)  # lever arm to section centroid [cite: 19]
            Mc = Cc * arm_Cc
        else:  # axis == 'y'
            [cite_start]xu = min(c, b) [cite: 20]
            [cite_start]Cc = 0.36 * fck * D * xu [cite: 20]
            [cite_start]arm_Cc = (b / 2.0) - (0.42 * xu) [cite: 20]
            Mc = Cc * arm_Cc
        
        # [cite_start]Steel bars [cite: 21]
        Fs = 0.0
        Ms = 0.0
        for (x, y, dia) in section.bars:
            As = bar_area(dia)
            if axis == 'x':
                # compression at top y=0, neutral axis at y=c
                [cite_start]strain = EPS_CU * (1.0 - (y / max(c, 1e-6))) [cite: 21]
                [cite_start]stress = np.clip(ES * strain, -0.87 * fy, 0.87 * fy) [cite: 23]
                force = stress * As
                [cite_start]z = (D / 2.0) - y # lever arm to centroid at D/2 [cite: 23]
            else:
                # compression at left x=0
                [cite_start]strain = EPS_CU * (1.0 - (x / max(c, 1e-6))) [cite: 24]
                [cite_start]stress = np.clip(ES * strain, -0.87 * fy, 0.87 * fy) [cite: 24]
                force = stress * As
                [cite_start]z = (b / 2.0) - x # lever arm to centroid at b/2 [cite: 25]

            Fs += force
            Ms += force * z

        [cite_start]N_res = Cc + Fs  # resultant axial (compression +) [cite: 25]
        [cite_start]M_res = Mc + Ms  # about centroidal axis [cite: 25]
        return N_res, M_res

    # [cite_start]Binary search on c to match Pu (target axial) [cite: 26, 27]
    target = Pu
    cL, cR = c_min, c_max
    NL, ML = forces_and_moment(cL)
    NR, MR = forces_and_moment(cR)

    # [cite_start]If target outside bracket, clamp [cite: 27]
    if (NL - target) * (NR - target) > 0:
        candidates = [(abs(NL - target), ML), (abs(NR - target), MR)]
        Mu = min(candidates, key=lambda t: t[0])[1]
        return float(Mu)

    [cite_start]for _ in range(60): [cite: 28]
        cm = 0.5 * (cL + cR)
        Nm, Mm = forces_and_moment(cm)
        if (Nm - target) == 0:
            return float(Mm)
        if (NL - target) * (Nm - target) <= 0:
            [cite_start]cR, NR, MR = cm, Nm, Mm [cite: 28]
        else:
            [cite_start]cL, NL, ML = cm, Nm, Mm [cite: 29]
    
    Mu = 0.5 * (ML + MR)
    return float(Mu)

def pm_curve(section: Section, fck: float, fy: float, axis: str, n: int = 80):
    [cite_start]"""Generate (P, M) points by sweeping neutral axis depth c[cite: 30]."""
    b, D = section.b, section.D

    def forces_and_moment(c: float):
        # [cite_start]Concrete block [cite: 30]
        if axis == 'x':
            xu = min(c, D)
            Cc = 0.36 * fck * b * xu
            [cite_start]arm_Cc = (D / 2.0) - (0.42 * xu) [cite: 31]
            Mc = Cc * arm_Cc
        else:  # 'y'
            xu = min(c, b)
            Cc = 0.36 * fck * D * xu
            [cite_start]arm_Cc = (b / 2.0) - (0.42 * xu) [cite: 31]
            Mc = Cc * arm_Cc
        # [cite_start]Steel [cite: 32]
        Fs = 0.0
        Ms = 0.0
        for (x, y, dia) in section.bars:
            As = bar_area(dia)
            if axis == 'x':
                [cite_start]strain = EPS_CU * (1.0 - (y / max(c, 1e-6))) [cite: 32, 33]
                stress = np.clip(ES * strain, -0.87 * fy, 0.87 * fy)
                force = stress * As
                z = (D / 2.0) - y
            else:
                [cite_start]strain = EPS_CU * (1.0 - (x / max(c, 1e-6))) [cite: 34]
                stress = np.clip(ES * strain, -0.87 * fy, 0.87 * fy)
                force = stress * As
                z = (b / 2.0) - x
            Fs += force
            [cite_start]Ms += force * z [cite: 35]
        N = Cc + Fs
        M = Mc + Ms
        return N, M

    cmin = 0.01 * (D if axis == 'x' else b)
    cmax = 1.50 * (D if axis == 'x' else b)
    cs = np.linspace(cmin, cmax, n)
    P_list, M_list = [], []
    for c in cs:
        [cite_start]N, M = forces_and_moment(c) [cite: 36]
        P_list.append(N)
        [cite_start]M_list.append(abs(M))  # envelope magnitude [cite: 36]
    return P_list, M_list

def moment_magnifier(Pu: float, le_mm: float, Ec: float, Ic: float, Cm: float = 0.85, sway: bool = False) -> float:
    [cite_start]"""Compute simple moment magnifier δ[cite: 11]."""
    [cite_start]Pcr = (math.pi ** 2) * Ec * Ic / (le_mm ** 2 + 1e-9) [cite: 13]
    if Pcr <= 0:
        return 1.0
    [cite_start]delta = 1.0 / max(1e-6, (1.0 - Pu / Pcr)) [cite: 11]
    
    if not sway:
        [cite_start]delta = max(1.0, Cm * delta) [cite: 13]
    [cite_start]delta = float(np.clip(delta, 1.0, 2.5)) [cite: 13]
    return delta

def scale_section_bars(section: Section, s_area: float) -> Section:
    [cite_start]"""Return a new Section with bar diameters scaled so bar areas multiply by s_area[cite: 36, 37]."""
    k = math.sqrt(max(s_area, 1e-6))
    new_bars = [(x, y, d * k) for (x, y, d) in section.bars]
    return Section(b=section.b, D=section.D, cover=section.cover, bars=new_bars, tie_dia=section.tie_dia)

def biaxial_utilization(section: Section, fck: float, fy: float, Pu: float, Mux_eff: float, Muy_eff: float, alpha: float) -> Tuple[float, float, float]:
    [cite_start]"""Return (util, Mux_lim, Muy_lim) for given demand using current section bars[cite: 38]."""
    Mux_lim = uniaxial_capacity_Mu_for_Pu(section, fck, fy, Pu, axis='x')
    Muy_lim = uniaxial_capacity_Mu_for_Pu(section, fck, fy, Pu, axis='y')
    [cite_start]Rx = (Mux_eff / max(Mux_lim, 1e-9)) ** alpha [cite: 38]
    [cite_start]Ry = (Muy_eff / max(Muy_lim, 1e-9)) ** alpha [cite: 38]
    util = Rx + Ry
    return util, Mux_lim, Muy_lim

def required_As_biaxial(section: Section, fck: float, fy: float, Pu: float, Mux_eff: float, Muy_eff: float, alpha: float, As_min: float, s_max: float = 5.0) -> Tuple[float, float, float]:
    [cite_start]"""Find required longitudinal steel area (mm²) to satisfy biaxial utilization ≤ 1 by uniformly scaling all bar areas (keeping geometry)[cite: 39]."""
    As_prov = section.As_long
    util_now, _, _ = biaxial_utilization(section, fck, fy, Pu, Mux_eff, Muy_eff, alpha)

    if util_now <= 1.0 and As_prov >= As_min:
        return As_prov, 1.0, util_now

    [cite_start]s_lo = max(1.0, As_min / max(As_prov, 1e-6)) [cite: 40]
    s_hi = max(s_lo + 1e-6, s_max)

    # [cite_start]Binary search for the scale factor s_area (s_hi) [cite: 41]
    for _ in range(50):
        s_mid = 0.5 * (s_lo + s_hi)
        sec_mid = scale_section_bars(section, s_mid)
        util_mid, _, _ = biaxial_utilization(sec_mid, fck, fy, Pu, Mux_eff, Muy_eff, alpha)
        if util_mid <= 1.0:
            s_hi = s_mid
        else:
            [cite_start]s_lo = s_mid [cite: 42]
            
    As_req = As_prov * s_hi
    return As_req, s_hi, util_now

def svg_cross_section(section: Section, tie_dia: float, tie_spacing: float) -> str:
    [cite_start]"""Return an SVG string showing the column, bars, and a sample tie rectangle with leader labels[cite: 43]."""
    b, D = section.b, section.D
    scale = 1.0  # 1 mm = 1 px (adjust if large)
    pad = 40  # px around
    [cite_start]W = int(b * scale + 2 * pad) [cite: 43]
    [cite_start]H = int(D * scale + 2 * pad) [cite: 43]

    def px(val_mm):
        return val_mm * scale + pad

    # SVG header
    parts = [
        f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">',
        [cite_start]'<style> .txt{font: 12px sans-serif;} .lbl{font: 11px sans-serif;}</style>' [cite: 43]
    ]

    # Concrete rectangle
    [cite_start]parts.append(f'<rect x="{px(0)}" y="{px(0)}" width="{b*scale}" height="{D*scale}" fill="#f2f2f2" stroke="#555" stroke-width="1"/>') [cite: 44]

    # Tie (one pitch shown as a rectangle inset by cover) – simplified visual
    cov = section.cover
    parts.append(
        [cite_start]f'<rect x="{px(cov)}" y="{px(cov)}" width="{(b-2*cov)*scale}" height="{(D-2*cov)*scale}" fill="none" stroke="#888" stroke-dasharray="4,3" stroke-width="1"/>' [cite: 44]
    )

    # Bars
    for i, (x, y, dia) in enumerate(section.bars, start=1):
        cx, cy, r = px(x), px(y), (dia * scale) / 2.0
        [cite_start]parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#0a66c2" stroke="#043f77" stroke-width="1"/>') [cite: 44]
        # leader
        [cite_start]lx, ly = cx + 30, cy - 10 [cite: 45]
        [cite_start]parts.append(f'<line x1="{cx}" y1="{cy}" x2="{lx}" y2="{ly}" stroke="#333" stroke-width="1"/>') [cite: 45]
        [cite_start]parts.append(f'<text x="{lx+4}" y="{ly-4}" class="lbl">Bar {i}: {int(dia)}mm</text>') [cite: 45]

    # Labels
    [cite_start]parts.append(f'<text x="{px(b/2)}" y="{px(-10)}" class="txt" text-anchor="middle">b = {b:.0f} mm</text>') [cite: 45]
    [cite_start]parts.append(f'<text x="{px(-10)}" y="{px(D/2)}" class="txt" text-anchor="end" transform="rotate(-90 {px(-10)} {px(D/2)})">D = {D:.0f} mm</text>') [cite: 45]
    [cite_start]parts.append(f'<text x="{px(b+5)}" y="{px(15)}" class="lbl">Tie: {int(tie_dia)} mm @ {int(tie_spacing)} mm c/c</text>') [cite: 45]

    parts.append('</svg>')
    return "\n".join(parts)


# ----------------------------
# 3. STREAMLIT UI and DATA I/O (New Structure)
# ----------------------------

def to_json_serializable(state: dict) -> dict:
    """Converts the session state to a JSON-safe dictionary."""
    safe_state = {}
    for key, value in state.items():
        if isinstance(value, float):
            safe_state[key] = round(value, 6)
        elif isinstance(value, np.float64):
            safe_state[key] = round(float(value), 6)
        elif isinstance(value, (list, tuple)) and all(isinstance(v, tuple) for v in value):
            # Convert list of tuples (x, y, dia) to list of lists for JSON
            safe_state[key] = [list(item) for item in value]
        else:
            safe_state[key] = value
    return safe_state

def get_json_download_link(data_dict: dict, filename: str) -> str:
    """Generates a downloadable link for a JSON dictionary."""
    json_str = json.dumps(data_dict, indent=4)
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'<a href="data:file/json;base64,{b64}" download="{filename}">💾 Download State as JSON</a>'
    return href

def plot_pm_curve_plotly(P_list, M_list, Pu, Mu_eff, axis):
    """Generates an interactive Plotly chart for the P-M curve."""
    df = pd.DataFrame({'P (kN)': np.array(P_list)/1e3, f'M_{axis} (kNm)': np.array(M_list)/1e6})

    fig = go.Figure()
    # P-M Capacity Curve
    fig.add_trace(go.Scatter(
        x=df['P (kN)'], y=df[f'M_{axis} (kNm)'], mode='lines', name='Capacity Envelope (M_lim)',
        line=dict(color='#0a66c2', width=3)
    ))
    # Design Point (Demand)
    fig.add_trace(go.Scatter(
        x=[Pu/1e3], y=[abs(Mu_eff)/1e6], mode='markers', name='Demand (Pu, Mu_eff)',
        marker=dict(symbol='x', size=12, color='red', line=dict(width=2, color='red'))
    ))
    
    fig.update_layout(
        title=f'P–M Capacity Envelope (Axis: {axis.upper()})',
        xaxis_title='Axial Load P (kN)',
        yaxis_title=f'Moment M_{axis} (kNm)',
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='white'
    )
    return fig


# ----------------------------
# MAIN APP BODY
# ----------------------------

st.set_page_config(page_title="RCC Column (Biaxial) Designer", layout="wide")
st.title("🧱 RCC Column Design — Biaxial Moments ± Axial ± Shear (IS 456/13920)")
st.markdown("---")

# --- Custom CSS for Input Appearance ---
st.markdown("""
<style>
/* Highlight background of input widgets */
.stNumberInput, .stSelectbox, .stCheckbox, .stSlider {
    background-color: #f0f8ff; /* Light blue/AliceBlue */
    padding: 10px;
    border-radius: 5px;
    border: 1px solid #ccc;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)


if "state" not in st.session_state:
    st.session_state.state = {
        # [cite_start]Defaults matching original file [cite: 48, 49]
        "b": 450.0, "D": 600.0, "cover": 40.0, "fck": 30.0, "fy": 500.0,
        "Pu": 1200e3, "Mux": 120e6, "Muy": 80e6, "Vu": 150e3,
        "storey_clear": 3200.0, "kx": 1.0, "ky": 1.0, "restraint": "Pinned-Pinned", "sway": False,
        # [cite_start]Bar defaults [cite: 50, 51]
        "n_top": 3, "n_bot": 3, "n_left": 2, "n_right": 2, 
        "dia_top": 16.0, "dia_bot": 16.0, "dia_side": 12.0, "tie_dia": 8.0, "tie_spacing": 150.0,
        "alpha": 1.0, "bars": []
    }
state = st.session_state.state

# -----------------------------
# 0. Data Management (JSON Import/Export)
# -----------------------------
st.header("🗄️ Data Management (Import / Export)")
c_up, c_down = st.columns(2)

with c_up:
    uploaded_file = st.file_uploader("Upload State (JSON)", type="json")
    if uploaded_file is not None:
        try:
            data = json.load(uploaded_file)
            # Basic conversion for lists of lists back to list of tuples
            if 'bars' in data:
                data['bars'] = [tuple(item) for item in data['bars']]
            st.session_state.state.update(data)
            st.success("Configuration loaded successfully! Scroll down to see updated inputs.")
        except Exception as e:
            st.error(f"Error loading JSON: {e}")

with c_down:
    if state["bars"]: # Only allow export if bars are generated
        st.markdown("**Export Current Design State**")
        safe_state = to_json_serializable(state)
        st.markdown(get_json_download_link(safe_state, "column_design_state.json"), unsafe_allow_html=True)
    else:
        st.info("Define inputs first to enable data export.")

# -----------------------------
# 1. Inputs & Section Geometry (T1)
# -----------------------------
with st.container():
    st.markdown("---")
    st.header("1️⃣ Column Geometry, Materials, and Loads")
    
    # [cite_start]--- Geometry and Materials --- [cite: 48, 49]
    st.markdown("### Geometry and Materials")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        state["b"] = st.number_input("Width $b$ (mm)", 200.0, 2000.0, state["b"], 25.0, key='in_b')
        state["fck"] = st.number_input("$f_{ck}$ (MPa)", 20.0, 80.0, state["fck"], 1.0, key='in_fck')
        state["sway"] = st.checkbox("Sway Frame?", value=state["sway"], key='in_sway')
    with c2:
        state["D"] = st.number_input("Depth $D$ (mm)", 200.0, 3000.0, state["D"], 25.0, key='in_D')
        state["fy"] = st.number_input("$f_{y}$ (MPa)", 415.0, 600.0, state["fy"], 5.0, key='in_fy')
        [cite_start]restraint_options = ["Fixed-Fixed", "Fixed-Pinned", "Pinned-Pinned", "Fixed-Free (cantilever)"] [cite: 10]
        state["restraint"] = st.selectbox("End Restraint", restraint_options, index=restraint_options.index(state["restraint"]), key='in_restraint')
    with c3:
        state["cover"] = st.number_input("Clear Cover (mm)", 20.0, 75.0, state["cover"], 5.0, key='in_cover')
        state["storey_clear"] = st.number_input("Clear Storey Height $l_0$ (mm)", 2000.0, 6000.0, state["storey_clear"], 50.0, key='in_l0')
        [cite_start]k_factor = effective_length_factor(state["restraint"]) [cite: 10]
        state["kx"], state["ky"] = k_factor, k_factor
        st.markdown(f"**$k_x = k_y$**: **{k_factor:.2f}**")
    
    # [cite_start]--- Factored Loads (Demand) --- [cite: 49]
    with c4:
        state["Pu"] = st.number_input("Axial Load $P_u$ (kN, +comp)", -3000.0, 6000.0, state["Pu"] / 1e3, 10.0, key='in_Pu') * 1e3
        state["Mux"] = st.number_input("Moment $M_{ux}$ (kNm)", -2000.0, 2000.0, state["Mux"] / 1e6, 5.0, key='in_Mux') * 1e6
        state["Muy"] = st.number_input("Moment $M_{uy}$ (kNm)", -2000.0, 2000.0, state["Muy"] / 1e6, 5.0, key='in_Muy') * 1e6
        state["Vu"] = st.number_input("Shear $V_u$ (kN)", 0.0, 5000.0, state["Vu"] / 1e3, 5.0, key='in_Vu') * 1e3

    # [cite_start]--- Longitudinal Bar Layout --- [cite: 50, 51]
    st.markdown("### Longitudinal Bar Layout")
    cL1, cL2, cL3, cL4, cL5 = st.columns(5)
    bar_options = [12, 16, 20, 25, 28, 32]
    
    with cL1:
        state["n_top"] = st.number_input("Top row bars ($n_{top}$)", 0, 10, state["n_top"], 1, key='in_ntop')
        state["dia_top"] = float(st.selectbox("Top bar $\\phi$ (mm)", bar_options, index=bar_options.index(int(state["dia_top"])), key='in_dtop'))
    with cL2:
        state["n_bot"] = st.number_input("Bottom row bars ($n_{bot}$)", 0, 10, state["n_bot"], 1, key='in_nbot')
        state["dia_bot"] = float(st.selectbox("Bottom bar $\\phi$ (mm)", bar_options, index=bar_options.index(int(state["dia_bot"])), key='in_dbot'))
    with cL3:
        state["n_left"] = st.number_input("Left column bars ($n_{left}$)", 0, 10, state["n_left"], 1, key='in_nleft')
        state["dia_side"] = float(st.selectbox("Side bar $\\phi$ (mm)", bar_options[:-1], index=bar_options[:-1].index(int(state["dia_side"])), key='in_dside'))
    with cL4:
        state["n_right"] = st.number_input("Right column bars ($n_{right}$)", 0, 10, state["n_right"], 1, key='in_nright')
        tie_options = [6, 8, 10, 12]
        state["tie_dia"] = float(st.selectbox("Tie $\\phi$ (mm)", tie_options, index=tie_options.index(int(state["tie_dia"])), key='in_tdia'))
    with cL5:
        state["tie_spacing"] = st.number_input("Tie Spacing $s$ (mm)", 50.0, 300.0, state["tie_spacing"], 5.0, key='in_ts')
        
    # Build bars and Section object
    state["bars"] = generate_rectangular_bar_layout(state["b"], state["D"], state["cover"], 
                                                    state["n_top"], state["n_bot"], state["n_left"], state["n_right"], 
                                                    [cite_start]state["dia_top"], state["dia_bot"], state["dia_side"]) [cite: 52]
    section = Section(b=state["b"], D=state["D"], cover=state["cover"], bars=state["bars"], tie_dia=state["tie_dia"])
    state["As_long"] = section.As_long

    st.markdown("#### Cross-section Preview")
    [cite_start]svg = svg_cross_section(section, tie_dia=state["tie_dia"], tie_spacing=state["tie_spacing"]) [cite: 53]
    st.components.v1.html(svg, height=min(600, int(section.D + 100)))

# -----------------------------
# 2. Slenderness Check (T2)
# -----------------------------
with st.container():
    st.markdown("---")
    st.header("2️⃣ Slenderness Check (IS 456: Cl. 25.1)")
    
    st.markdown(f"""
    The column's effective lengths are $l_{{e,x}} = k_x l_0 = {state["kx"]} \\times {state["storey_clear"]:.0f}$ mm and $l_{{e,y}} = k_y l_0 = {state["ky"]} \\times {state["storey_clear"]:.0f}$ mm.
    The slenderness ratio is calculated as: $\\lambda = \\frac{{l_e}}{{r}}$ (where $r = \\sqrt{{I_c/A_g}}$).
    """)

    # [cite_start]Calculations [cite: 55]
    le_x = state["kx"] * state["storey_clear"]
    le_y = state["ky"] * state["storey_clear"]
    lam_x = le_x / max(section.rx, 1e-6)
    lam_y = le_y / max(section.ry, 1e-6)
    short_x = lam_x <= 12.0
    short_y = lam_y <= 12.0
    state.update(dict(le_x=le_x, le_y=le_y, lam_x=lam_x, lam_y=lam_y, short_x=short_x, short_y=short_y))

    col_x, col_y = st.columns(2)
    with col_x:
        st.write(f"Radius of gyration $r_x$ = **{section.rx:.1f} mm**")
        st.metric("Slenderness $\lambda_x$", f"{lam_x:.1f}")
    with col_y:
        st.write(f"Radius of gyration $r_y$ = **{section.ry:.1f} mm**")
        st.metric("Slenderness $\lambda_y$", f"{lam_y:.1f}")
    
    [cite_start]st.info(f"Classification → About x: {'**Short**' if short_x else '**Slender**'}, About y: {'**Short**' if short_y else '**Slender**'} [cite: 56]")

# -----------------------------
# 3. Moments (2nd-order magnification) (T3)
# -----------------------------
with st.container():
    st.markdown("---")
    st.header("3️⃣ Second-Order Moment Magnification")
    
    st.markdown(f"""
    For slender axes, the design moment $M_u$ is magnified by $\\delta$ (IS 456 Cl. 39.7.1).
    We use the simplified effective stiffness $EI_{{eff}} \\approx 0.4 E_c I_c$.
    """)

    # [cite_start]Calculations [cite: 58]
    Ec = Ec_from_fck(state["fck"])
    
    delta_x = moment_magnifier(state["Pu"], state["le_x"], Ec, section.Ic_x, Cm=0.85, sway=state["sway"]) if not state["short_x"] else 1.0
    delta_y = moment_magnifier(state["Pu"], state["le_y"], Ec, section.Ic_y, Cm=0.85, sway=state["sway"]) if not state["short_y"] else 1.0
    
    Mux_eff = state["Mux"] * delta_x
    Muy_eff = state["Muy"] * delta_y
    state.update(dict(Mux_eff=Mux_eff, Muy_eff=Muy_eff, delta_x=delta_x, delta_y=delta_y))

    c1, c2 = st.columns(2)
    with c1:
        st.write(f"Magnifier $\\delta_x$ = **{delta_x:.2f}**")
        st.metric("Magnified Moment $M_{ux}'$ (kNm)", f"{kNm(Mux_eff)}")
    with c2:
        st.write(f"Magnifier $\\delta_y$ = **{delta_y:.2f}**")
        st.metric("Magnified Moment $M_{uy}'$ (kNm)", f"{kNm(Muy_eff)}")

# -----------------------------
# 4. Interaction (Biaxial) (T4)
# -----------------------------
with st.container():
    st.markdown("---")
    st.header("4️⃣ Biaxial Interaction and Capacity Check")
    
    st.markdown("""
    The column capacity is checked against the magnified moments using Bresler's interaction equation:
    $$\\left(\\frac{|M_{ux}'|}{M_{ux,lim}}\\right)^{\\alpha} + \\left(\\frac{|M_{uy}'|}{M_{uy,lim}}\\right)^{\\alpha} \le 1.0$$
    $M_{u,lim}$ are the uniaxial moment capacities found by **Strain Compatibility** for $P_u$.
    """)
    
    # Contextual Input: Interaction Exponent
    state["alpha"] = st.slider("Interaction Exponent $\\alpha$ (IS 456 suggests 1.0-2.0)", 0.8, 2.0, state["alpha"], 0.1, key='in_alpha')

    # [cite_start]Compute uniaxial Mu,lim for the given Pu [cite: 60]
    with st.spinner("Computing uniaxial capacities via strain compatibility..."):
        Mux_lim = uniaxial_capacity_Mu_for_Pu(section, state["fck"], state["fy"], state["Pu"], axis='x')
        Muy_lim = uniaxial_capacity_Mu_for_Pu(section, state["fck"], state["fy"], state["Pu"], axis='y')

    # [cite_start]Utilization Check [cite: 61]
    util, Mux_lim, Muy_lim = biaxial_utilization(section, state["fck"], state["fy"], state["Pu"], Mux_eff, Muy_eff, state["alpha"])
    state.update(dict(Mux_lim=Mux_lim, Muy_lim=Muy_lim, util=util))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Uniaxial $M_{ux,lim}$ (kNm)", f"{kNm(Mux_lim)}")
    with c2:
        st.metric("Uniaxial $M_{uy,lim}$ (kNm)", f"{kNm(Muy_lim)}")
    with c3:
        st.metric("Utilization (Σ ≤ 1)", f"**{util:.2f}**")
        
    if util <= 1.0:
        [cite_start]st.success("✅ Biaxial interaction PASS. [cite: 62]")
    else:
        [cite_start]st.error("❌ Biaxial interaction FAIL — increase section/steel or revise layout. [cite: 62]")

    # --- P-M Interaction Curves (Plotly) ---
    st.markdown("#### Interactive P–M Interaction Curves (Capacity vs. Demand)")
    colx, coly = st.columns(2)
    
    with colx:
        [cite_start]Px, Mx = pm_curve(section, state["fck"], state["fy"], axis='x', n=120) [cite: 62]
        figx = plot_pm_curve_plotly(Px, Mx, state["Pu"], Mux_eff, 'x')
        st.plotly_chart(figx, use_container_width=True)
    
    with coly:
        [cite_start]Py, My = pm_curve(section, state["fck"], state["fy"], axis='y', n=120) [cite: 63]
        figy = plot_pm_curve_plotly(Py, My, state["Pu"], Muy_eff, 'y')
        st.plotly_chart(figy, use_container_width=True)

# -----------------------------
# 5. Shear Design (T5)
# -----------------------------
with st.container():
    st.markdown("---")
    st.header("5️⃣ Shear Design (IS 456 Cl. 40)")
    
    st.markdown(f"""
    Concrete shear capacity $V_c$ is enhanced by axial compression via the factor $\\phi_N$ (IS 456 Cl. 40.2.2):
    $$\\phi_N = 1.0 + \\frac{{P_u}}{{0.25 f_{{ck}} A_g}} \quad (0.5 \\le \\phi_N \\le 1.5)$$
    Excess shear $V_{{us}} = V_u - V_c$ determines the required tie spacing.
    """)

    # [cite_start]Calculations [cite: 65]
    max_long_dia = max([bar[2] for bar in state["bars"]], default=16.0)
    d_eff = state["D"] - (state["cover"] + 0.5 * max_long_dia) # Simplified effective depth
    
    Ag = section.Ag
    phiN_calc = 1.0 + (state["Pu"] / max(1.0, (0.25 * state["fck"] * Ag)))
    [cite_start]phiN = float(np.clip(phiN_calc, 0.5, 1.5)) [cite: 65]
    [cite_start]tau_c = 0.62 * math.sqrt(state["fck"]) / 1.0  # MPa (teaching simplification) [cite: 65]
    [cite_start]Vc = tau_c * state["b"] * d_eff * phiN # N [cite: 66]
    
    Vus = max(0.0, state["Vu"] - Vc)
    state.update(dict(Vc=Vc, Vus=Vus, phiN=phiN, d_eff=d_eff))
    
    # Tie inputs and calculations
    st.markdown("### Tie Configuration")
    col_l, col_d = st.columns(2)
    with col_l:
        leg_options_map = {"2-legged": 2, "4-legged": 4, "6-legged": 6}
        current_legs_str = f"{state.get('legs', 2)}-legged"
        current_legs_str = current_legs_str if current_legs_str in leg_options_map else "2-legged"
        selected_legs_str = st.selectbox("Tie Legs ($A_{sv}$)", list(leg_options_map.keys()), index=list(leg_options_map.keys()).index(current_legs_str), key='in_legs')
        state["legs"] = leg_options_map[selected_legs_str]
    with col_d:
        tie_options = [6.0, 8.0, 10.0, 12.0]
        state["tie_dia"] = st.selectbox("Tie $\\phi$ (mm)", tie_options, index=tie_options.index(state["tie_dia"]), key='in_tdia_shear')

    [cite_start]Asv = state["legs"] * bar_area(state["tie_dia"]) [cite: 67]
    
    c_met1, c_met2 = st.columns(2)
    with c_met1:
        st.metric("Concrete Capacity $V_c$ (kN)", f"{kN(Vc)}")
    with c_met2:
        st.metric("Excess Shear $V_{us}$ (kN)", f"{kN(Vus)}")
        
    if Vus > 0:
        [cite_start]s_required = (0.87 * state["fy"] * Asv * d_eff) / max(Vus, 1e-6) [cite: 67]
        state["s_required"] = s_required
        [cite_start]st.warning(f"$V_{{us}}$ of {kN(Vus)} kN requires $s_{{req}}$ $\\le$ **{s_required:.0f} mm**. [cite: 69]")
    else:
        s_required = 300.0 # Minimum shear spacing required for all columns
        state["s_required"] = s_required
        [cite_start]st.info("Vu $\\le V_c \\rightarrow$ Provide minimum transverse reinforcement. [cite: 66]")
    
    st.write(f"Provided Ties: **{state['legs']} legs $\\times$ {int(state['tie_dia'])} mm** @ **{state['tie_spacing']:.0f} mm** c/c.")

# -----------------------------
# 6. Detailing and Final Check (T6)
# -----------------------------
with st.container():
    st.markdown("---")
    st.header("6️⃣ Detailing Checks and Final Summary")
    
    # [cite_start]Longitudinal Steel Checks [cite: 70]
    As_long = section.As_long
    rho_long = 100.0 * As_long / section.Ag
    As_min = 0.008 * section.Ag 
    
    st.write(f"Total $A_{{st}}$ provided = **{As_long:.0f} mm²** ($\\rho_l$ = **{rho_long:.2f}%**)")
    
    # [cite_start]Required steel by biaxial capacity (≥ As_min) [cite: 72]
    As_req_cap, s_req_scale, util_now = required_As_biaxial(section, state["fck"], state["fy"], state["Pu"], Mux_eff, Muy_eff, state["alpha"], As_min)
    As_governing = max(As_req_cap, As_min)
    [cite_start]delta_As = max(0.0, As_governing - As_long) [cite: 73]

    c_l1, c_l2 = st.columns(2)
    with c_l1:
        st.metric("$A_{st}$ Minimum (0.8%)", f"{As_min:.0f} mm²")
    with c_l2:
        st.metric("Governing $A_{st}$ Required (mm²)", f"{As_governing:.0f}")

    if util_now <= 1.0 and As_long >= As_min:
        [cite_start]st.success("🎉 **Design PASS** for Strength and Minimum Steel. [cite: 74]")
    else:
        [cite_start]st.error(f"⚠️ **Design FAIL**. Need an additional **{delta_As:.0f} mm²** of steel. [cite: 74]")

    # [cite_start]Tie Spacing Detailing Checks [cite: 70, 71]
    s_lim1 = 16.0 * max_long_dia
    s_lim2 = min(state["b"], state["D"]) 
    s_lim3 = 300.0
    s_cap = min(s_lim1, s_lim2, s_lim3)
    s_governing_tie = min(s_cap, state["s_required"])

    st.markdown("### Transverse Reinforcement Spacing Check")
    [cite_start]st.write(f"Tie Spacing Max Cap (IS 456) $\\le$ min($16\\phi_{{long}}$, Least Dim, 300) = **{s_cap:.0f} mm**. [cite: 71]")
    st.write(f"Required for Shear $s_{{req,V}}$ = **{state['s_required']:.0f} mm**.")
    
    if state["tie_spacing"] <= s_governing_tie:
        st.success(f"Tie Spacing PASS: {state['tie_spacing']:.0f} mm $\\le$ {s_governing_tie:.0f} mm.")
    else:
        st.error(f"Tie Spacing FAIL: Must provide spacing $\\le$ **{s_governing_tie:.0f} mm**.")


# -----------------------------
# 7. Output Summary (Printable Table) (T7)
# -----------------------------
with st.container():
    st.markdown("---")
    st.header("7️⃣ Printable Output Summary")
    
    out = {
        "b (mm)": state["b"],
        "D (mm)": state["D"],
        "fck (MPa)": state["fck"],
        "fy (MPa)": state["fy"],
        [cite_start]"Pu (kN)": state["Pu"] / 1e3, [cite: 75]
        "Mux′ (kNm)": kNm(Mux_eff),
        "Muy′ (kNm)": kNm(Muy_eff),
        [cite_start]"Vu (kN)": state["Vu"] / 1e3, [cite: 75]
        "--- ENGINEERING RESULTS ---": "---",
        [cite_start]"Slenderness λx": f"{state['lam_x']:.1f}", [cite: 76]
        [cite_start]"Slenderness λy": f"{state['lam_y']:.1f}", [cite: 76]
        [cite_start]"Moment Magnifier δx": f"{state['delta_x']:.2f}", [cite: 76]
        [cite_start]"Uniaxial Capacity Mux,lim (kNm)": kNm(Mux_lim), [cite: 76]
        [cite_start]"Biaxial Utilization (≤1)": f"{state['util']:.2f}", [cite: 76]
        [cite_start]"Concrete Shear Vc (kN)": state["Vc"] / 1e3, [cite: 76]
        [cite_start]"Vus (kN)": state["Vus"] / 1e3, [cite: 77]
        "--- DETAILING ---": "---",
        "Ast Provided (mm²)": f"{As_long:.0f}",
        "Ast Governing Required (mm²)": f"{As_governing:.0f}",
        "Tie Spacing Provided (mm)": f"{state['tie_spacing']:.0f}",
        "Tie Spacing Governing Req (mm)": f"{s_governing_tie:.0f}",
    }

    df_out = pd.DataFrame({"Parameter": list(out.keys()), "Value": list(out.values())})
    st.dataframe(df_out, use_container_width=True)
    
    [cite_start]st.caption("Notes: Uniaxial capacities are computed via full strain compatibility. Verify with SP-16/design charts and apply full IS 456/13920 clauses in production. [cite: 77, 78]")
