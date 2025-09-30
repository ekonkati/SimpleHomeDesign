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
FY_DEFAULT = 500.0  # MPa 

def Ec_from_fck(fck: float) -> float:
    """IS 456: Ec = 5000 * sqrt(fck) (MPa)."""
    return 5000.0 * math.sqrt(max(fck, 1e-6))

def bar_area(dia_mm: float) -> float:
    """Area of a single circular bar (mm^2)."""
    return math.pi * (dia_mm ** 2) / 4.0

def kN(value: float) -> str:
    """Formats N to kN (1 decimal place)."""
    return f"{value / 1e3:.1f}"

def kNm(value: float) -> str:
    """Formats N-mm to kN-m (1 decimal place)."""
    return f"{value / 1e6:.1f}"

def effective_length_factor(restraint: str) -> float:
    """Returns k factor based on simplified end restraint (IS 456: Table 28)."""
    if restraint == "Fixed-Fixed": return 0.65
    if restraint == "Fixed-Pinned": return 0.8
    if restraint == "Pinned-Pinned": return 1.0
    if restraint == "Fixed-Free (cantilever)": return 2.0
    return 1.0

@dataclass
class Section:
    b: float  # mm (width along local x-axis)
    D: float  # mm (depth along local y-axis)
    cover: float  # mm (clear cover to main bars)
    bars: List[Tuple[float, float, float]]  # list of (x, y, dia_mm)
    tie_dia: float = 8.0 

    @property
    def Ag(self) -> float:
        return self.b * self.D

    @property
    def Ic_x(self) -> float:
        """Moment of inertia about local x-axis (for bending about x)."""
        return self.b * self.D**3 / 12.0

    @property
    def Ic_y(self) -> float:
        """Moment of inertia about local y-axis (for bending about y)."""
        return self.D * self.b**3 / 12.0

    @property
    def rx(self) -> float:
        """Radius of gyration about local x-axis."""
        return math.sqrt(self.Ic_x / self.Ag)

    @property
    def ry(self) -> float:
        """Radius of gyration about local y-axis."""
        return math.sqrt(self.Ic_y / self.Ag)
    
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
    """Generates bar list (x, y, dia) for a rectangular tied column. Origin at (0, 0)."""
    bars = []
    
    # Helper for linear spacing
    def linspace(a, c, n):
        if n == 1: return [a + (c - a) / 2.0]
        return [a + i * (c - a) / (n - 1) for i in range(n)]

    # Corner and side offsets
    d_x = cover + dia_side / 2.0
    d_y_top = cover + dia_top / 2.0
    d_y_bot = D - (cover + dia_bot / 2.0)
    
    x_span = linspace(d_x, b - d_x, max(n_top, n_bot, 2))
    y_span = linspace(d_y_top, d_y_bot, max(n_left, n_right, 2))

    # Top row
    if n_top > 0:
        for x in x_span:
            bars.append((x, d_y_top, dia_top))

    # Bottom row
    if n_bot > 0:
        for x in x_span:
            bars.append((x, d_y_bot, dia_bot))

    # Left and Right columns (excluding corners handled by top/bot)
    x_left = d_x
    x_right = b - d_x
    
    for y in y_span:
        if y != d_y_top and y != d_y_bot:
            if n_left > 0: bars.append((x_left, y, dia_side))
            if n_right > 0: bars.append((x_right, y, dia_side))

    # Remove near-duplicate corner bars introduced by side spacing
    unique_bars = []
    for x, y, dia in bars:
        is_unique = True
        for ux, uy, udia in unique_bars:
            if abs(x - ux) < 1.0 and abs(y - uy) < 1.0:
                is_unique = False
                break
        if is_unique:
            unique_bars.append((x, y, dia))
            
    return unique_bars


def uniaxial_capacity_Mu_for_Pu(section: Section, fck: float, fy: float, Pu: float, axis: str) -> float:
    """
    Compute ultimate uniaxial moment capacity Mu,lim for a given factored axial Pu (N) and bending axis 
    using binary search on the neutral axis depth 'c' to satisfy Pu.
    """
    dimension = section.D if axis == 'x' else section.b
    
    def forces_and_moment(c: float):
        # Concrete block contribution
        Cc = 0.36 * fck * dimension * min(c, dimension)
        arm_Cc = 0.5 * dimension - 0.42 * min(c, dimension)
        Mc = Cc * arm_Cc
        
        # Steel bars (y-coordinates are referenced from the compression face)
        Fs = 0.0
        Ms = 0.0
        for (x, y_abs, dia) in section.bars:
            As = bar_area(dia)
            y = y_abs if axis == 'y' else dimension - y_abs # y is distance from compression face (x=0 or y=0)
            
            strain = EPS_CU * (1.0 - (y / max(c, 1e-6)))
            stress = np.clip(ES * strain, -0.87 * fy, 0.87 * fy)
            force = stress * As
            
            z = 0.5 * dimension - y # lever arm to centroid
            
            Fs += force
            Ms += force * z

        N_res = Cc + Fs  # resultant axial (compression +)
        M_res = Mc + Ms  # about centroidal axis
        return N_res, M_res

    # Binary search on c to match Pu
    c_min = 0.05 * dimension
    c_max = 1.50 * dimension

    target = Pu
    cL, cR = c_min, c_max
    NL, ML = forces_and_moment(cL)
    NR, MR = forces_and_moment(cR)

    # Clamping for extreme cases
    if (NL - target) * (NR - target) > 0:
        candidates = [(abs(NL - target), ML), (abs(NR - target), MR)]
        return min(candidates, key=lambda t: t[0])[1]

    # Binary search loop
    for _ in range(60):
        cm = 0.5 * (cL + cR)
        Nm, Mm = forces_and_moment(cm)
        if abs(Nm - target) < 1.0: return float(Mm)
        
        if (NL - target) * (Nm - target) <= 0:
            cR, NR, MR = cm, Nm, Mm
        else:
            cL, NL, ML = cm, Nm, Mm
    
    return float(0.5 * (ML + MR))


def pm_curve(section: Section, fck: float, fy: float, axis: str, n: int = 80):
    """Generate (P, M) points by sweeping neutral axis depth c."""
    dimension = section.D if axis == 'x' else section.b

    def forces_and_moment(c: float):
        # Concrete block contribution
        Cc = 0.36 * fck * dimension * min(c, dimension)
        arm_Cc = 0.5 * dimension - 0.42 * min(c, dimension)
        Mc = Cc * arm_Cc
        
        # Steel bars
        Fs = 0.0
        Ms = 0.0
        for (x, y_abs, dia) in section.bars:
            As = bar_area(dia)
            y = y_abs if axis == 'y' else dimension - y_abs
            
            strain = EPS_CU * (1.0 - (y / max(c, 1e-6)))
            stress = np.clip(ES * strain, -0.87 * fy, 0.87 * fy)
            force = stress * As
            
            z = 0.5 * dimension - y # lever arm to centroid
            
            Fs += force
            Ms += force * z

        N = Cc + Fs
        M = Mc + Ms
        return N, M

    # Sweep c from 0.01D (tension failure) to 1.5D (compression failure)
    cmin = 0.01 * dimension
    cmax = 1.50 * dimension
    cs = np.linspace(cmin, cmax, n)
    P_list, M_list = [], []
    for c in cs:
        N, M = forces_and_moment(c)
        P_list.append(N)
        M_list.append(abs(M))  # envelope magnitude
    return P_list, M_list

def moment_magnifier(Pu: float, le_mm: float, Ec: float, Ic: float, Cm: float = 0.85, sway: bool = False) -> float:
    """Compute simple moment magnifier δ."""
    Pcr = (math.pi ** 2) * 0.4 * Ec * Ic / (le_mm ** 2 + 1e-9) # 0.4 Ec Ic is a simplification
    
    # Check for short column (handled upstream by calling function)
    if Pcr <= Pu:
        return 10.0 # Effectively infinity, leads to fail
        
    delta = 1.0 / max(1e-6, (1.0 - Pu / Pcr))
    
    if not sway:
        delta = max(1.0, Cm * delta)
    delta = float(np.clip(delta, 1.0, 2.5))
    return delta

def scale_section_bars(section: Section, s_area: float) -> Section:
    """Return a new Section with bar areas scaled by s_area."""
    k = math.sqrt(max(s_area, 1e-6))
    new_bars = [(x, y, d * k) for (x, y, d) in section.bars]
    return Section(b=section.b, D=section.D, cover=section.cover, bars=new_bars, tie_dia=section.tie_dia)

def biaxial_utilization(section: Section, fck: float, fy: float, Pu: float, Mux_eff: float, Muy_eff: float, alpha: float) -> Tuple[float, float, float]:
    """Return (util, Mux_lim, Muy_lim) for given demand using current section bars."""
    Mux_lim = uniaxial_capacity_Mu_for_Pu(section, fck, fy, Pu, axis='x')
    Muy_lim = uniaxial_capacity_Mu_for_Pu(section, fck, fy, Pu, axis='y')
    Rx = (abs(Mux_eff) / max(Mux_lim, 1e-9)) ** alpha
    Ry = (abs(Muy_eff) / max(Muy_lim, 1e-9)) ** alpha
    util = Rx + Ry
    return util, Mux_lim, Muy_lim

def required_As_biaxial(section: Section, fck: float, fy: float, Pu: float, Mux_eff: float, Muy_eff: float, alpha: float, As_min: float, s_max: float = 5.0) -> Tuple[float, float, float]:
    """Find required longitudinal steel area (mm²) to satisfy biaxial utilization ≤ 1."""
    As_prov = section.As_long
    util_now, _, _ = biaxial_utilization(section, fck, fy, Pu, Mux_eff, Muy_eff, alpha)

    if util_now <= 1.0 and As_prov >= As_min:
        return As_prov, 1.0, util_now

    s_lo = max(1.0, As_min / max(As_prov, 1e-6))
    s_hi = max(s_lo + 0.1, s_max) # Ensure s_hi is larger than s_lo

    # Binary search for the scale factor s_area (s_hi)
    for _ in range(50):
        s_mid = 0.5 * (s_lo + s_hi)
        sec_mid = scale_section_bars(section, s_mid)
        util_mid, _, _ = biaxial_utilization(sec_mid, fck, fy, Pu, Mux_eff, Muy_eff, alpha)
        if util_mid <= 1.0:
            s_hi = s_mid
        else:
            s_lo = s_mid
            
    As_req = As_prov * s_hi
    return As_req, s_hi, util_now


def svg_cross_section(section: Section, tie_dia: float, tie_spacing: float) -> str:
    """Return an SVG string showing the column, bars, and a sample tie rectangle."""
    # ... (SVG function remains the same as previous step for brevity, assuming it works)
    b, D = section.b, section.D
    scale = 0.35 # mm per px for viewing
    pad = 40 
    W = int(b / scale + 2 * pad)
    H = int(D / scale + 2 * pad)

    def px(val_mm):
        return val_mm / scale + pad

    parts = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">',
             '<style> .txt{font: 12px sans-serif;} .lbl{font: 11px sans-serif;}</style>']

    # Concrete
    parts.append(f'<rect x="{px(0)}" y="{px(0)}" width="{b/scale}" height="{D/scale}" fill="#f2f2f2" stroke="#555" stroke-width="1"/>')

    # Tie (simplified)
    cov = section.cover
    parts.append(f'<rect x="{px(cov)}" y="{px(cov)}" width="{(b-2*cov)/scale}" height="{(D-2*cov)/scale}" fill="none" stroke="#888" stroke-dasharray="4,3" stroke-width="1"/>')

    # Bars
    for i, (x, y, dia) in enumerate(section.bars, start=1):
        cx, cy, r = px(x), px(y), (dia / scale) / 2.0
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#0a66c2" stroke="#043f77" stroke-width="1"/>')

    # Labels
    parts.append(f'<text x="{px(b/2)}" y="{px(-10)}" class="txt" text-anchor="middle">b = {b:.0f} mm</text>')
    parts.append(f'<text x="{px(-10)}" y="{px(D/2)}" class="txt" text-anchor="end" transform="rotate(-90 {px(-10)} {px(D/2)})">D = {D:.0f} mm</text>')
    parts.append(f'<text x="{px(b+5)}" y="{px(15)}" class="lbl">Tie: {int(tie_dia)} mm @ {int(tie_spacing)} mm c/c</text>')

    parts.append('</svg>')
    return "\n".join(parts)

# ----------------------------
# 3. STREAMLIT UI and DATA I/O (New Structure)
# ----------------------------

# (to_json_serializable, get_json_download_link, plot_pm_curve_plotly remain the same)
def to_json_serializable(state: dict) -> dict:
    safe_state = {}
    for key, value in state.items():
        if isinstance(value, float): safe_state[key] = round(value, 6)
        elif isinstance(value, np.float64): safe_state[key] = round(float(value), 6)
        elif isinstance(value, (list, tuple)) and all(isinstance(v, tuple) for v in value):
            safe_state[key] = [list(item) for item in value]
        else: safe_state[key] = value
    return safe_state

def get_json_download_link(data_dict: dict, filename: str) -> str:
    json_str = json.dumps(data_dict, indent=4)
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'<a href="data:file/json;base64,{b64}" download="{filename}">💾 Download State as JSON</a>'
    return href

def plot_pm_curve_plotly(P_list, M_list, Pu, Mu_eff, axis):
    df = pd.DataFrame({'P (kN)': np.array(P_list)/1e3, f'M_{axis} (kNm)': np.array(M_list)/1e6})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['P (kN)'], y=df[f'M_{axis} (kNm)'], mode='lines', name='Capacity Envelope (M_lim)', line=dict(color='#0a66c2', width=3)))
    fig.add_trace(go.Scatter(x=[Pu/1e3], y=[abs(Mu_eff)/1e6], mode='markers', name='Demand (Pu, Mu_eff)', marker=dict(symbol='x', size=12, color='red', line=dict(width=2, color='red'))))
    fig.update_layout(title=f'P–M Capacity Envelope (Axis: {axis.upper()})', xaxis_title='Axial Load P (kN)', yaxis_title=f'Moment M_{axis} (kNm)', hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor='white')
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
.stNumberInput, .stSelectbox, .stCheckbox, .stSlider {
    background-color: #f0f8ff; 
    padding: 10px;
    border-radius: 5px;
    border: 1px solid #ccc;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)


if "state" not in st.session_state:
    # Initialize state with default values
    st.session_state.state = {
        "b": 450.0, "D": 600.0, "cover": 40.0, "fck": 30.0, "fy": 500.0,
        "Pu": 1200e3, "Mux": 120e6, "Muy": 80e6, "Vu": 150e3,
        "storey_clear": 3200.0, "kx": 1.0, "ky": 1.0, "restraint": "Pinned-Pinned", "sway": False,
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
            if 'bars' in data:
                data['bars'] = [tuple(item) for item in data['bars']]
            st.session_state.state.update(data)
            st.success("Configuration loaded successfully! Scroll down to see updated inputs.")
        except Exception as e:
            st.error(f"Error loading JSON: {e}")

with c_down:
    if state["bars"]: 
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
    
    # --- Geometry and Materials ---
    st.markdown("### Geometry and Materials")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        state["b"] = st.number_input("Width $b$ (mm)", 200.0, 2000.0, state["b"], 25.0, key='in_b')
        state["fck"] = st.number_input("$f_{ck}$ (MPa)", 20.0, 80.0, state["fck"], 1.0, key='in_fck')
        state["sway"] = st.checkbox("Sway Frame?", value=state["sway"], key='in_sway')
    with c2:
        state["D"] = st.number_input("Depth $D$ (mm)", 200.0, 3000.0, state["D"], 25.0, key='in_D')
        state["fy"] = st.number_input("$f_{y}$ (MPa)", 415.0, 600.0, state["fy"], 5.0, key='in_fy')
        restraint_options = ["Fixed-Fixed", "Fixed-Pinned", "Pinned-Pinned", "Fixed-Free (cantilever)"]
        state["restraint"] = st.selectbox("End Restraint", restraint_options, index=restraint_options.index(state["restraint"]), key='in_restraint')
    with c3:
        state["cover"] = st.number_input("Clear Cover (mm)", 20.0, 75.0, state["cover"], 5.0, key='in_cover')
        state["storey_clear"] = st.number_input("Clear Storey Height $l_0$ (mm)", 2000.0, 6000.0, state["storey_clear"], 50.0, key='in_l0')
        k_factor = effective_length_factor(state["restraint"])
        state["kx"], state["ky"] = k_factor, k_factor
        st.markdown(f"**$k_x = k_y$**: **{k_factor:.2f}**")
    
    # --- Factored Loads (Demand) ---
    with c4:
        state["Pu"] = st.number_input("Axial Load $P_u$ (kN, +comp)", -3000.0, 6000.0, state["Pu"] / 1e3, 10.0, key='in_Pu') * 1e3
        state["Mux"] = st.number_input("Moment $M_{ux}$ (kNm)", -2000.0, 2000.0, state["Mux"] / 1e6, 5.0, key='in_Mux') * 1e6
        state["Muy"] = st.number_input("Moment $M_{uy}$ (kNm)", -2000.0, 2000.0, state["Muy"] / 1e6, 5.0, key='in_Muy') * 1e6
        state["Vu"] = st.number_input("Shear $V_u$ (kN)", 0.0, 5000.0, state["Vu"] / 1e3, 5.0, key='in_Vu') * 1e3

    # --- Longitudinal Bar Layout ---
    st.markdown("### Longitudinal Bar Layout")
    cL1, cL2, cL3, cL4, cL5 = st.columns(5)
    bar_options = [12.0, 16.0, 20.0, 25.0, 28.0, 32.0]
    
    with cL1:
        state["n_top"] = st.number_input("Top row bars ($n_{top}$)", 0, 10, state["n_top"], 1, key='in_ntop')
        state["dia_top"] = st.selectbox("Top bar $\\phi$ (mm)", bar_options, index=bar_options.index(state["dia_top"]), key='in_dtop')
    with cL2:
        state["n_bot"] = st.number_input("Bottom row bars ($n_{bot}$)", 0, 10, state["n_bot"], 1, key='in_nbot')
        state["dia_bot"] = st.selectbox("Bottom bar $\\phi$ (mm)", bar_options, index=bar_options.index(state["dia_bot"]), key='in_dbot')
    with cL3:
        state["n_left"] = st.number_input("Left column bars ($n_{left}$)", 0, 10, state["n_left"], 1, key='in_nleft')
        state["dia_side"] = st.selectbox("Side bar $\\phi$ (mm)", bar_options[:-1], index=bar_options[:-1].index(state["dia_side"]), key='in_dside')
    with cL4:
        state["n_right"] = st.number_input("Right column bars ($n_{right}$)", 0, 10, state["n_right"], 1, key='in_nright')
        tie_options = [6.0, 8.0, 10.0, 12.0]
        state["tie_dia"] = st.selectbox("Tie $\\phi$ (mm)", tie_options, index=tie_options.index(state["tie_dia"]), key='in_tdia')
    with cL5:
        state["tie_spacing"] = st.number_input("Tie Spacing $s$ (mm)", 50.0, 300.0, state["tie_spacing"], 5.0, key='in_ts')
        
    # Build bars and Section object
    state["bars"] = generate_rectangular_bar_layout(state["b"], state["D"], state["cover"], 
                                                    state["n_top"], state["n_bot"], state["n_left"], state["n_right"], 
                                                    state["dia_top"], state["dia_bot"], state["dia_side"])
    section = Section(b=state["b"], D=state["D"], cover=state["cover"], bars=state["bars"], tie_dia=state["tie_dia"])
    state["As_long"] = section.As_long

    st.markdown("#### Cross-section Preview")
    svg = svg_cross_section(section, tie_dia=state["tie_dia"], tie_spacing=state["tie_spacing"])
    st.components.v1.html(svg, height=min(600, int(section.D / 0.35 + 80)))

# -----------------------------
# 2. Slenderness Check (T2)
# -----------------------------
with st.container():
    st.markdown("---")
    st.header("2️⃣ Slenderness Check (IS 456: Cl. 25.1)")
    
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
    
    st.info(f"Classification → About x: {'**Short**' if short_x else '**Slender**'}, About y: {'**Short**' if short_y else '**Slender**'}")

# -----------------------------
# 3. Moments (2nd-order magnification) (T3)
# -----------------------------
with st.container():
    st.markdown("---")
    st.header("3️⃣ Second-Order Moment Magnification")
    
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
    $$\\left(\\frac{|M_{ux}'|}{M_{ux,lim}}\\right)^{\\alpha} + \\left(\\frac{|M_{uy}'|}{M_{uy,lim}}\\right)^{\\alpha} \le 1.0$$
    """)
    
    state["alpha"] = st.slider("Interaction Exponent $\\alpha$", 0.8, 2.0, state["alpha"], 0.1, key='in_alpha')

    with st.spinner("Computing uniaxial capacities via strain compatibility..."):
        util, Mux_lim, Muy_lim = biaxial_utilization(section, state["fck"], state["fy"], state["Pu"], Mux_eff, Muy_eff, state["alpha"])
    state.update(dict(Mux_lim=Mux_lim, Muy_lim=Muy_lim, util=util))

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Uniaxial $M_{ux,lim}$ (kNm)", f"{kNm(Mux_lim)}")
    with c2: st.metric("Uniaxial $M_{uy,lim}$ (kNm)", f"{kNm(Muy_lim)}")
    with c3: st.metric("Utilization (Σ ≤ 1)", f"**{util:.2f}**")
        
    if util <= 1.0: st.success("✅ Biaxial interaction PASS.")
    else: st.error("❌ Biaxial interaction FAIL.")

    st.markdown("#### Interactive P–M Interaction Curves (Capacity vs. Demand)")
    colx, coly = st.columns(2)
    with colx:
        Px, Mx = pm_curve(section, state["fck"], state["fy"], axis='x', n=120)
        figx = plot_pm_curve_plotly(Px, Mx, state["Pu"], Mux_eff, 'x')
        st.plotly_chart(figx, use_container_width=True)
    with coly:
        Py, My = pm_curve(section, state["fck"], state["fy"], axis='y', n=120)
        figy = plot_pm_curve_plotly(Py, My, state["Pu"], Muy_eff, 'y')
        st.plotly_chart(figy, use_container_width=True)

# -----------------------------
# 5. Shear Design (T5)
# -----------------------------
with st.container():
    st.markdown("---")
    st.header("5️⃣ Shear Design (IS 456 Cl. 40)")
    
    max_long_dia = max([bar[2] for bar in state["bars"]], default=16.0)
    d_eff = state["D"] - (state["cover"] + 0.5 * max_long_dia) 
    
    Ag = section.Ag
    phiN_calc = 1.0 + (state["Pu"] / max(1.0, (0.25 * state["fck"] * Ag)))
    phiN = float(np.clip(phiN_calc, 0.5, 1.5))
    tau_c = 0.62 * math.sqrt(state["fck"]) / 1.0  
    Vc = tau_c * state["b"] * d_eff * phiN 
    Vus = max(0.0, state["Vu"] - Vc)
    state.update(dict(Vc=Vc, Vus=Vus, phiN=phiN, d_eff=d_eff))
    
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

    Asv = state["legs"] * bar_area(state["tie_dia"])
    
    c_met1, c_met2 = st.columns(2)
    with c_met1: st.metric("Concrete Capacity $V_c$ (kN)", f"{kN(Vc)}")
    with c_met2: st.metric("Excess Shear $V_{us}$ (kN)", f"{kN(Vus)}")
        
    if Vus > 0:
        s_required = (0.87 * state["fy"] * Asv * d_eff) / max(Vus, 1e-6)
        state["s_required"] = s_required
        st.warning(f"$V_{{us}}$ of {kN(Vus)} kN requires $s_{{req}}$ $\\le$ **{s_required:.0f} mm**.")
    else:
        s_required = 300.0 
        state["s_required"] = s_required
        st.info("Vu $\\le V_c \\rightarrow$ Provide minimum transverse reinforcement.")
    
    st.write(f"Provided Ties: **{state['legs']} legs $\\times$ {int(state['tie_dia'])} mm** @ **{state['tie_spacing']:.0f} mm** c/c.")

# -----------------------------
# 6. Detailing and Final Check (T6)
# -----------------------------
with st.container():
    st.markdown("---")
    st.header("6️⃣ Detailing Checks and Final Summary")
    
    As_long = section.As_long
    rho_long = 100.0 * As_long / section.Ag
    As_min = 0.008 * section.Ag 
    
    st.write(f"Total $A_{{st}}$ provided = **{As_long:.0f} mm²** ($\\rho_l$ = **{rho_long:.2f}%**)")
    
    As_req_cap, s_req_scale, util_now = required_As_biaxial(section, state["fck"], state["fy"], state["Pu"], Mux_eff, Muy_eff, state["alpha"], As_min)
    As_governing = max(As_req_cap, As_min)
    delta_As = max(0.0, As_governing - As_long)

    c_l1, c_l2 = st.columns(2)
    with c_l1: st.metric("$A_{st}$ Minimum (0.8%)", f"{As_min:.0f} mm²")
    with c_l2: st.metric("Governing $A_{st}$ Required (mm²)", f"{As_governing:.0f}")

    if util_now <= 1.0 and As_long >= As_min:
        st.success("🎉 **Design PASS** for Strength and Minimum Steel.")
    else:
        st.error(f"⚠️ **Design FAIL**. Need an additional **{delta_As:.0f} mm²** of steel.")

    # Tie Spacing Detailing Checks
    s_lim1 = 16.0 * max_long_dia
    s_lim2 = min(state["b"], state["D"]) 
    s_lim3 = 300.0
    s_cap = min(s_lim1, s_lim2, s_lim3)
    s_governing_tie = min(s_cap, state["s_required"])

    st.markdown("### Transverse Reinforcement Spacing Check")
    st.write(f"Tie Spacing Max Cap (IS 456) $\\le$ min($16\\phi_{{long}}$, Least Dim, 300) = **{s_cap:.0f} mm**.")
    
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
        "b (mm)": state["b"], "D (mm)": state["D"], "fck (MPa)": state["fck"], "fy (MPa)": state["fy"],
        "Pu (kN)": state["Pu"] / 1e3, "Mux′ (kNm)": kNm(Mux_eff), "Muy′ (kNm)": kNm(Muy_eff), "Vu (kN)": state["Vu"] / 1e3,
        "--- ENGINEERING RESULTS ---": "---",
        "Slenderness λx": f"{state['lam_x']:.1f}", "Slenderness λy": f"{state['lam_y']:.1f}",
        "Moment Magnifier δx": f"{state['delta_x']:.2f}", 
        "Uniaxial Capacity Mux,lim (kNm)": kNm(Mux_lim),
        "Biaxial Utilization (≤1)": f"{state['util']:.2f}",
        "Concrete Shear Vc (kN)": state["Vc"] / 1e3, "Vus (kN)": state["Vus"] / 1e3,
        "--- DETAILING ---": "---",
        "Ast Provided (mm²)": f"{As_long:.0f}",
        "Ast Governing Required (mm²)": f"{As_governing:.0f}",
        "Tie Spacing Provided (mm)": f"{state['tie_spacing']:.0f}",
        "Tie Spacing Governing Req (mm)": f"{s_governing_tie:.0f}",
    }

    df_out = pd.DataFrame({"Parameter": list(out.keys()), "Value": list(out.values())})
    st.dataframe(df_out, use_container_width=True)
    
    st.caption("Notes: Uniaxial capacities are computed via full strain compatibility. Verify with SP-16/design charts and apply full IS 456/13920 clauses in production.")
