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
# 2. CORE ENGINEERING LOGIC 
# -----------------------------

def generate_rectangular_bar_layout(b: float, D: float, cover: float, 
                                    n_top: int, n_bot: int, n_left: int, n_right: int, 
                                    dia_top: float, dia_bot: float, dia_side: float) -> List[Tuple[float, float, float]]:
    """Generates bar list (x, y, dia) for a rectangular tied column. Origin at (0, 0, bottom-left).
    Simplified logic to avoid complex conditional branching.
    """
    bars = []
    
    def linspace(a, c, n):
        if n <= 1: return [a + (c - a) / 2.0]
        return [a + i * (c - a) / (n - 1) for i in range(n)]

    # Distances from edge to bar centerline
    dx_bar = cover + dia_side / 2.0
    dy_bar_top = D - (cover + dia_top / 2.0) # Measured from (0,0) bottom-left
    dy_bar_bot = cover + dia_bot / 2.0      # Measured from (0,0) bottom-left
    
    # 1. Top Row
    if n_top > 0:
        x_span = linspace(dx_bar, b - dx_bar, n_top)
        for x in x_span:
            bars.append((x, dy_bar_top, dia_top))

    # 2. Bottom Row
    if n_bot > 0:
        x_span = linspace(dx_bar, b - dx_bar, n_bot)
        for x in x_span:
            bars.append((x, dy_bar_bot, dia_bot))

    # 3. Left & Right Columns (must align with corners and use side bar properties)
    n_y_span = max(n_left, n_right, 1)
    y_span = linspace(dy_bar_bot, dy_bar_top, n_y_span) # span from bottom to top

    # Left Column
    if n_left > 0:
        for i in range(n_left):
            bars.append((dx_bar, y_span[i], dia_side))

    # Right Column
    if n_right > 0:
        for i in range(n_right):
            bars.append((b - dx_bar, y_span[i], dia_side))

    # Final cleanup for unique bars (essential step for corner bars)
    unique_bars = []
    for x, y, dia in bars:
        is_unique = True
        for i, (ux, uy, udia) in enumerate(unique_bars):
            # Check for near-duplicate position
            if abs(x - ux) < 1e-3 and abs(y - uy) < 1e-3:
                # If a duplicate is found (e.g., at a corner), use the larger bar
                if dia > udia:
                    unique_bars[i] = (x, y, dia) 
                is_unique = False
                break
        if is_unique:
            unique_bars.append((x, y, dia))
            
    return unique_bars

def uniaxial_capacity_Mu_for_Pu(section: Section, fck: float, fy: float, Pu: float, axis: str) -> float:
    """
    Computes ultimate uniaxial moment capacity Mu,lim for a given factored axial Pu (N).
    Bending about x means compression face is on top (y=D). Bending about y means compression face is on left (x=0).
    """
    dimension = section.D if axis == 'x' else section.b
    
    def forces_and_moment(c: float):
        # Concrete block contribution (0.36 fck b x_u * (D/2 - 0.42 x_u))
        Cc = 0.36 * fck * dimension * min(c, dimension)
        arm_Cc = 0.5 * dimension - 0.42 * min(c, dimension)
        Mc = Cc * arm_Cc
        
        Fs, Ms = 0.0, 0.0
        
        for (x_abs, y_abs, dia) in section.bars:
            As = bar_area(dia)
            # y is distance from compression face 
            if axis == 'x': # Bending about x-axis (D is depth), Compression face at y=D
                y = section.D - y_abs
            else: # Bending about y-axis (b is depth), Compression face at x=0
                y = x_abs 
            
            strain = EPS_CU * (1.0 - (y / max(c, 1e-6)))
            stress = np.clip(ES * strain, -0.87 * fy, 0.87 * fy)
            force = stress * As
            
            z = 0.5 * dimension - y # lever arm to centroid
            
            Fs += force
            Ms += force * z

        N_res = Cc + Fs  # resultant axial (compression +)
        M_res = Mc + Ms  # about centroidal axis
        return N_res, M_res

    # Binary search on c to match Pu (Pu must be positive/compression for search to work well)
    target = Pu
    c_min = 0.05 * dimension
    c_max = 1.50 * dimension

    cL, cR = c_min, c_max
    NL, ML = forces_and_moment(cL)
    NR, MR = forces_and_moment(cR)

    # Simplified checks for out-of-bounds Pu
    if target <= NL: return ML
    if target >= NR: return MR

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
        Cc = 0.36 * fck * dimension * min(c, dimension)
        arm_Cc = 0.5 * dimension - 0.42 * min(c, dimension)
        Mc = Cc * arm_Cc
        Fs, Ms = 0.0, 0.0
        
        for (x_abs, y_abs, dia) in section.bars:
            As = bar_area(dia)
            if axis == 'x': 
                y = section.D - y_abs
            else: 
                y = x_abs 
            
            strain = EPS_CU * (1.0 - (y / max(c, 1e-6)))
            stress = np.clip(ES * strain, -0.87 * fy, 0.87 * fy)
            force = stress * As
            z = 0.5 * dimension - y 
            
            Fs += force
            Ms += force * z

        N = Cc + Fs
        M = Mc + Ms
        return N, M

    cmin = 0.01 * dimension
    cmax = 1.50 * dimension
    cs = np.linspace(cmin, cmax, n)
    P_list, M_list = [], []
    for c in cs:
        N, M = forces_and_moment(c)
        P_list.append(N)
        M_list.append(abs(M))
    return P_list, M_list

def moment_magnifier(Pu: float, le_mm: float, Ec: float, Ic: float, Cm: float = 0.85, sway: bool = False) -> float:
    """Compute simple moment magnifier δ."""
    Pcr = (math.pi ** 2) * 0.4 * Ec * Ic / (le_mm ** 2 + 1e-9)
    if Pcr <= Pu: return 10.0
    delta = 1.0 / max(1e-6, (1.0 - Pu / Pcr))
    if not sway: delta = max(1.0, Cm * delta)
    return float(np.clip(delta, 1.0, 2.5))

def biaxial_utilization(section: Section, fck: float, fy: float, Pu: float, Mux_eff: float, Muy_eff: float, alpha: float) -> Tuple[float, float, float]:
    Mux_lim = uniaxial_capacity_Mu_for_Pu(section, fck, fy, Pu, axis='x')
    Muy_lim = uniaxial_capacity_Mu_for_Pu(section, fck, fy, Pu, axis='y')
    
    # Handle Mux_lim or Muy_lim being zero for extreme tension cases (approximate for utilization)
    if Mux_lim < 1e-3 and abs(Mux_eff) > 1e-3: Mux_lim = 1e-3
    if Muy_lim < 1e-3 and abs(Muy_eff) > 1e-3: Muy_lim = 1e-3
        
    Rx = (abs(Mux_eff) / Mux_lim) ** alpha
    Ry = (abs(Muy_eff) / Muy_lim) ** alpha
    util = Rx + Ry
    return util, Mux_lim, Muy_lim

# ----------------------------
# 3. PLOTLY VISUALIZATIONS
# ----------------------------

def plotly_cross_section(section: Section) -> go.Figure:
    """Generates a scaled Plotly figure of the column cross-section."""
    b, D = section.b, section.D
    cover = section.cover
    
    fig = go.Figure()

    # 1. Concrete Cross-Section (Outer box)
    fig.add_shape(type="rect", x0=0, y0=0, x1=b, y1=D, line=dict(color="black", width=2), fillcolor="rgba(240, 240, 240, 0.8)")

    # 2. Clear Cover Boundary
    fig.add_shape(type="rect", x0=cover, y0=cover, x1=b - cover, y1=D - cover, line=dict(color="gray", width=1, dash="dot"), fillcolor="rgba(0,0,0,0)")

    # 3. Longitudinal Bars
    bar_x, bar_y, bar_size, bar_text = [], [], [], []
    for x_geom, y_geom, dia in section.bars:
        bar_x.append(x_geom)
        bar_y.append(y_geom) 
        bar_size.append(dia * 2) 
        bar_text.append(f"Ø{dia:.0f} mm ({bar_area(dia):.0f} mm²)")

    fig.add_trace(go.Scatter(
        x=bar_x, y=bar_y, mode='markers', name='Rebars',
        marker=dict(size=bar_size, sizemode='diameter', color='#0a66c2', line=dict(width=1, color='DarkSlateGrey')),
        hovertemplate='Size: %{text}<extra></extra>', text=bar_text
    ))

    # Layout and Aspect Ratio
    max_dim = max(b, D)
    fig.update_layout(
        title=f"Cross Section (b={b:.0f}, D={D:.0f} mm)",
        xaxis=dict(range=[-0.1 * max_dim, b + 0.1 * max_dim], showgrid=False, zeroline=False, title="Width (mm)"),
        yaxis=dict(range=[-0.1 * max_dim, D + 0.1 * max_dim], showgrid=False, zeroline=False, title="Depth (mm)"),
        width=400, height=400 * D / b if b != 0 else 400,
        showlegend=False, hovermode="closest", plot_bgcolor='white',
        yaxis_scaleanchor="x", yaxis_scaleratio=1 # Force 1:1 aspect ratio
    )
    return fig

def plotly_elevation(state: dict, section: Section) -> go.Figure:
    """Generates an elevation view of the column showing ties and effective length."""
    b = section.b
    lo = state["storey_clear"]
    le = state["kx"] * lo
    
    fig = go.Figure()

    # 1. Column body
    fig.add_shape(type="rect", x0=-b/2, y0=0, x1=b/2, y1=lo, line=dict(color="black", width=2), fillcolor="rgba(240, 240, 240, 0.8)"))

    # 2. Longitudinal Rebars (Simplified 4 face bars)
    x_rebars = [-b/2 + section.cover, b/2 - section.cover] # Edge bars
    for x_r in x_rebars:
        fig.add_trace(go.Scatter(x=[x_r, x_r], y=[0, lo], mode='lines', name='Rebar', line=dict(color='#0a66c2', width=2), showlegend=False))

    # 3. Ties (Horizontal lines)
    s_prov = state["tie_spacing"]
    n_ties = int(lo / s_prov)
    if n_ties > 0:
        tie_y = np.linspace(s_prov / 2, lo - s_prov / 2, n_ties)
        
        for y_t in tie_y:
            fig.add_trace(go.Scatter(x=[-b/2, b/2], y=[y_t, y_t], mode='lines', name='Tie', line=dict(color='#888', width=1), showlegend=False))
    
    # 4. Effective Length (Annotation)
    fig.add_shape(type="line", x0=b/2 + 20, y0=lo/2 - le/2, x1=b/2 + 20, y1=lo/2 + le/2, line=dict(color="red", width=3, dash="dash"))
    fig.add_annotation(x=b/2 + 20, y=lo/2 + le/2 + 50, text=f"$l_e = {le:.0f} mm$", showarrow=False, font=dict(color="red", size=14))
    
    # 5. Tie Spacing Annotation
    if n_ties > 0:
        fig.add_annotation(
            x=-b/2 - 20, y=s_prov / 2 + 50,
            text=f"Tie Spacing $s = {s_prov:.0f} mm$", showarrow=True, arrowhead=2, ax=-80, ay=50,
            bgcolor="white", font=dict(size=12)
        )

    # Layout
    max_x = b/2 + 100
    fig.update_layout(
        title=f"Column Elevation ($l_0 = {lo:.0f} mm$)",
        xaxis=dict(range=[-max_x, max_x], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-0.1 * lo, lo + 0.1 * lo], showgrid=False, zeroline=False, title="Height (mm)"),
        width=450, height=600, showlegend=False, plot_bgcolor='white',
    )
    return fig


# ----------------------------
# 4. STREAMLIT UI and DATA I/O
# ----------------------------
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
    st.session_state.state = {
        "b": 450.0, "D": 600.0, "cover": 40.0, "fck": 30.0, "fy": 500.0,
        "Pu": 1200e3, "Mux": 120e6, "Muy": 80e6, "Vu": 150e3,
        "storey_clear": 3200.0, "kx": 1.0, "ky": 1.0, "restraint": "Pinned-Pinned", "sway": False,
        "n_top": 3, "n_bot": 3, "n_left": 2, "n_right": 2, 
        "dia_top": 16.0, "dia_bot": 16.0, "dia_side": 12.0, "tie_dia": 8.0, "tie_spacing": 150.0,
        "alpha": 1.0, "bars": [], "legs": 2
    }
state = st.session_state.state

# Initial Section definition for metadata and first computations
bars_temp = generate_rectangular_bar_layout(state["b"], state["D"], state["cover"], 
                                                state["n_top"], state["n_bot"], state["n_left"], state["n_right"], 
                                                state["dia_top"], state["dia_bot"], state["dia_side"])
section = Section(b=state["b"], D=state["D"], cover=state["cover"], bars=bars_temp, tie_dia=state["tie_dia"])
state["As_long"] = section.As_long

# -----------------------------
# 0. Data Management
# -----------------------------
st.header("🗄️ Data Management (Import / Export)")
c_up, c_down = st.columns(2)

with c_up:
    uploaded_file = st.file_uploader("Upload State (JSON)", type="json")
    if uploaded_file is not None:
        try:
            data = json.load(uploaded_file)
            if 'bars' in data: data['bars'] = [tuple(item) for item in data['bars']]
            st.session_state.state.update(data)
            st.rerun() 
        except Exception as e:
            st.error(f"Error loading JSON: {e}")

with c_down:
    if state["bars"]: 
        st.markdown("**Export Current Design State**")
        safe_state = to_json_serializable(state)
        st.markdown(get_json_download_link(safe_state, "column_design_state.json"), unsafe_allow_html=True)
    else:
        st.info("Define inputs first to enable data export.")

st.markdown("---")

# -----------------------------
# 1. Geometry, Materials, and Loads
# -----------------------------
st.header("1️⃣ Column Geometry, Materials, and Loads")
st.markdown("Define the **physical dimensions, material grades, and factored loads** for the column.")

# --- Geometry and Materials ---
st.markdown("### Geometry and Materials")
c1, c2, c3, c4 = st.columns(4)
with c1:
    state["b"] = st.number_input("Width $b$ (mm)", 200.0, 2000.0, state["b"], 25.0, key='in_b')
    state["fck"] = st.number_input("$f_{ck}$ (MPa, M-Grade)", 20.0, 80.0, state["fck"], 1.0, key='in_fck')
    state["sway"] = st.checkbox("Sway Frame? (Yes/No)", value=state["sway"], key='in_sway')
with c2:
    state["D"] = st.number_input("Depth $D$ (mm)", 200.0, 3000.0, state["D"], 25.0, key='in_D')
    state["fy"] = st.number_input("$f_{y}$ (MPa, Fe-Grade)", 415.0, 600.0, state["fy"], 5.0, key='in_fy')
    restraint_options = ["Fixed-Fixed", "Fixed-Pinned", "Pinned-Pinned", "Fixed-Free (cantilever)"]
    state["restraint"] = st.selectbox("End Restraint", restraint_options, index=restraint_options.index(state["restraint"]), key='in_restraint')
with c3:
    state["cover"] = st.number_input("Clear Cover (mm)", 20.0, 75.0, state["cover"], 5.0, key='in_cover')
    state["storey_clear"] = st.number_input("Clear Storey Height $l_0$ (mm)", 2000.0, 6000.0, state["storey_clear"], 50.0, key='in_l0')
    k_factor = effective_length_factor(state["restraint"])
    state["kx"], state["ky"] = k_factor, k_factor
    st.markdown(f"Effective Length Factor **$k_x = k_y$**: **{k_factor:.2f}**")

# --- Factored Loads (Demand) ---
with c4:
    state["Pu"] = st.number_input("Axial Load $P_u$ (kN, +comp)", -3000.0, 6000.0, state["Pu"] / 1e3, 10.0, key='in_Pu') * 1e3
    state["Mux"] = st.number_input("Moment $M_{ux}$ (kNm)", -2000.0, 2000.0, state["Mux"] / 1e6, 5.0, key='in_Mux') * 1e6
    state["Muy"] = st.number_input("Moment $M_{uy}$ (kNm)", -2000.0, 2000.0, state["Muy"] / 1e6, 5.0, key='in_Muy') * 1e6
    state["Vu"] = st.number_input("Shear $V_u$ (kN)", 0.0, 5000.0, state["Vu"] / 1e3, 5.0, key='in_Vu') * 1e3

st.markdown("---")

# -----------------------------
# 2. Slenderness Check
# -----------------------------
st.header("2️⃣ Slenderness Check (IS 456: Cl. 25.1)")
st.markdown("Determining if the column is **short** ($\lambda \le 12$) or **slender** ($\lambda > 12$) based on the effective length.")

# Calculations
le_x = state["kx"] * state["storey_clear"]
le_y = state["ky"] * state["storey_clear"]
lam_x = le_x / max(section.rx, 1e-6)
lam_y = le_y / max(section.ry, 1e-6)
short_x = lam_x <= 12.0
short_y = lam_y <= 12.0
state.update(dict(le_x=le_x, le_y=le_y, lam_x=lam_x, lam_y=lam_y, short_x=short_x, short_y=short_y))

st.latex(r"\text{Slenderness Ratio: } \lambda = \frac{l_{e}}{r} \quad \text{where } r = \sqrt{I_{c}/A_g}")
st.markdown(f"**$A_g$**: {section.Ag:.0f} mm², **$I_{c,x}$**: {section.Ic_x/1e6:.0f} $\\times 10^6$ mm⁴, **$r_x$**: {section.rx:.1f} mm")

col_x, col_y = st.columns(2)
with col_x:
    st.metric(f"Effective Length $l_{{e,x}}$", f"{le_x:.0f} mm")
    st.metric("Slenderness $\lambda_x$", f"{lam_x:.1f}")
with col_y:
    st.metric(f"Effective Length $l_{{e,y}}$", f"{le_y:.0f} mm")
    st.metric("Slenderness $\lambda_y$", f"{lam_y:.1f}")

st.info(f"Classification → About x: {'**Short**' if short_x else '**Slender**'}, About y: {'**Short**' if short_y else '**Slender**'}")
st.markdown("---")

# -----------------------------
# 3. Moments (2nd-order magnification)
# -----------------------------
st.header("3️⃣ Second-Order Moment Magnification")
st.markdown("For slender columns, the design moment $M_u$ is magnified to account for P-$\delta$ effects using the formula below.")

# Calculations
Ec = Ec_from_fck(state["fck"])
st.latex(r"E_c = 5000 \sqrt{f_{ck}} = " + f"{Ec:.0f} \text{ MPa}")
st.latex(r"\text{Critical Load: } P_{cr} = \frac{\pi^2 (0.4 E_c I_c)}{l_e^2}")
st.latex(r"\text{Magnifier: } \delta = \frac{C_m}{1 - P_u/P_{cr}} \quad (\text{Max } 2.5, \text{ Min } 1.0 \text{ if non-sway})")

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
st.markdown("---")

# -----------------------------
# 6. Detailing and Final Check (Rebar Adjustment) -> PERFORMED HERE FIRST FOR ITERATION
# -----------------------------
with st.container():
    st.header("6️⃣ Longitudinal Rebar Adjustment & Detailing Checks")
    st.markdown("Adjust the bar layout here to meet the required capacity from **Section 4**. The change instantly triggers analysis re-runs.")
    
    # --- Longitudinal Bar Layout (Moved from Section 1) ---
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
        leg_options_map = {"2-legged": 2, "4-legged": 4, "6-legged": 6}
        current_legs_str = f"{state.get('legs', 2)}-legged"
        selected_legs_str = st.selectbox("Tie Legs ($A_{sv}$)", list(leg_options_map.keys()), index=list(leg_options_map.keys()).index(current_legs_str), key='in_legs')
        state["legs"] = leg_options_map[selected_legs_str]
    with cL5:
        tie_options = [6.0, 8.0, 10.0, 12.0]
        state["tie_dia"] = st.selectbox("Tie $\\phi$ (mm)", tie_options, index=tie_options.index(state["tie_dia"]), key='in_tdia_shear')
        state["tie_spacing"] = st.number_input("Tie Spacing $s$ (mm)", 50.0, 300.0, state["tie_spacing"], 5.0, key='in_ts')

    # --- Bar Generation and Re-definition of Section ---
    state["bars"] = generate_rectangular_bar_layout(state["b"], state["D"], state["cover"], 
                                                state["n_top"], state["n_bot"], state["n_left"], state["n_right"], 
                                                state["dia_top"], state["dia_bot"], state["dia_side"])
    section = Section(b=state["b"], D=state["D"], cover=state["cover"], bars=state["bars"], tie_dia=state["tie_dia"])
    state["As_long"] = section.As_long

    st.markdown("### Cross-Section and Elevation Visualization")
    c_plot_cs, c_plot_elev = st.columns(2)
    with c_plot_cs:
        st.plotly_chart(plotly_cross_section(section), use_container_width=True)
    with c_plot_elev:
        st.plotly_chart(plotly_elevation(state, section), use_container_width=True)
    
    # Detailing Checks (Minimum/Maximum Steel)
    As_long = section.As_long
    rho_long = 100.0 * As_long / section.Ag
    As_min = 0.008 * section.Ag 
    As_max = 0.06 * section.Ag

    c_l1, c_l2, c_l3 = st.columns(3)
    with c_l1: st.metric("$A_{st}$ Provided (mm²)", f"{As_long:.0f} ($\\rho_l$ = {rho_long:.2f}%)")
    with c_l2: st.metric("$A_{st}$ Minimum (0.8%)", f"{As_min:.0f} mm²")
    with c_l3: st.metric("$A_{st}$ Maximum (6.0%)", f"{As_max:.0f} mm²")
    
    if As_long < As_min:
        st.error("⚠️ **FAIL**: Provided steel is less than the 0.8% minimum.")
    elif As_long > As_max:
        st.error("⚠️ **FAIL**: Provided steel exceeds the 6.0% maximum.")
    else:
        st.success("✅ Minimum/Maximum steel limits are met.")

st.markdown("---")

# -----------------------------
# 4. Interaction (Biaxial)
# -----------------------------
st.header("4️⃣ Biaxial Interaction and Capacity Check (Strength)")
st.markdown("The uniaxial moment capacity $M_{u,lim}$ is determined using the **Strain Compatibility Solver** for the given axial load $P_u$.")

# Contextual Input: Interaction Exponent
st.markdown("### Bresler's Interaction Formula (IS 456)")
state["alpha"] = st.slider("Interaction Exponent $\\alpha$ (IS 456 suggests 1.0-2.0)", 0.8, 2.0, state["alpha"], 0.1, key='in_alpha_check')
st.latex(r"\left(\frac{|M_{ux}'|}{M_{ux,lim}}\right)^{\alpha} + \left(\frac{|M_{uy}'|}{M_{uy,lim}}\right)^{\alpha} \le 1.0")


with st.spinner("Computing uniaxial capacities via strain compatibility..."):
    util, Mux_lim, Muy_lim = biaxial_utilization(section, state["fck"], state["fy"], state["Pu"], Mux_eff, Muy_eff, state["alpha"])
state.update(dict(Mux_lim=Mux_lim, Muy_lim=Muy_lim, util=util))

c1, c2, c3 = st.columns(3)
with c1: st.metric("Uniaxial $M_{ux,lim}$ (kNm)", f"{kNm(Mux_lim)}")
with c2: st.metric("Uniaxial $M_{uy,lim}$ (kNm)", f"{kNm(Muy_lim)}")
with c3: st.metric("Utilization (Σ ≤ 1)", f"**{util:.2f}**")
    
if util <= 1.0: st.success("✅ Biaxial interaction PASS.")
else: st.error("❌ Biaxial interaction FAIL — **Increase steel area** in Section 6.")

# P-M Interaction Curves (Plotly)
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
st.markdown("---")


# -----------------------------
# 5. Shear Design
# -----------------------------
with st.container():
    st.header("5️⃣ Shear Design (IS 456 Cl. 40)")
    st.markdown("Shear capacity ($V_c$) is enhanced by axial compression ($\\phi_N$).")

    # Calculations
    max_long_dia = max([bar[2] for bar in state["bars"]], default=16.0)
    d_eff = state["D"] - (state["cover"] + 0.5 * max_long_dia) 
    
    Ag = section.Ag
    st.latex(r"\phi_N = 1.0 + \frac{P_u}{0.25 f_{ck} A_g} \quad (0.5 \le \phi_N \le 1.5)")
    phiN_calc = 1.0 + (state["Pu"] / max(1.0, (0.25 * state["fck"] * Ag)))
    phiN = float(np.clip(phiN_calc, 0.5, 1.5))
    
    # Using simplified minimum tau_c for safety
    tau_c = 0.62 * math.sqrt(state["fck"]) / 1.0 
    Vc = tau_c * state["b"] * d_eff * phiN
    st.latex(r"V_c = \tau_c' \cdot b \cdot d_{eff} = " + f"{kN(Vc)} \text{ kN} \quad (\tau_c' = \phi_N \tau_c)")
    
    Vus = max(0.0, state["Vu"] - Vc)
    state.update(dict(Vc=Vc, Vus=Vus, phiN=phiN, d_eff=d_eff))
    
    c_met1, c_met2 = st.columns(2)
    with c_met1: st.metric("Concrete Capacity $V_c$ (kN)", f"{kN(Vc)}")
    with c_met2: st.metric("Excess Shear $V_{us}$ (kN)", f"{kN(Vus)}")
        
    # Shear reinforcement calculation
    Asv = state["legs"] * bar_area(state["tie_dia"])
    s_required = (0.87 * state["fy"] * Asv * d_eff) / max(Vus, 1e-6) if Vus > 0 else 300.0
    state["s_required"] = s_required

    if Vus > 0:
        st.latex(r"s_{req, V} = \frac{0.87 f_y A_{sv} d_{eff}}{V_{us}}")
        st.warning(f"$V_{{us}}$ of {kN(Vus)} kN requires $s_{{req}}$ $\\le$ **{s_required:.0f} mm**.")
    else:
        st.info("Vu $\\le V_c \\rightarrow$ Provide minimum transverse reinforcement.")
    
    # Tie Spacing Detailing Checks (Summary)
    s_lim1 = 16.0 * max_long_dia
    s_lim2 = min(state["b"], state["D"]) 
    s_lim3 = 300.0
    s_cap = min(s_lim1, s_lim2, s_lim3)
    s_governing_tie = min(s_cap, state["s_required"])
    state["s_governing_tie"] = s_governing_tie
    
    st.markdown("### Transverse Reinforcement Spacing Check (IS 456 & Shear)")
    st.latex(r"s_{max} \le \min(16 \phi_{long}, \text{Least Dimension}, 300 \text{ mm}, s_{req, V})")
    
    st.write(f"Tie Spacing Max (IS 456 Detailing) $\\le$ **{s_cap:.0f} mm**.")
    st.write(f"Required for Shear $s_{{req,V}}$ = **{state['s_required']:.0f} mm**.")
    
    if state["tie_spacing"] <= s_governing_tie:
        st.success(f"Tie Spacing PASS: {state['tie_spacing']:.0f} mm $\\le$ {s_governing_tie:.0f} mm.")
    else:
        st.error(f"Tie Spacing FAIL: Must provide spacing $\\le$ **{s_governing_tie:.0f} mm**.")

st.markdown("---")

# -----------------------------
# 7. Output Summary (Printable Table)
# -----------------------------
with st.container():
    st.header("7️⃣ Printable Output Summary")
    
    # Simplified estimation for required As based on last util for output table
    As_governing = max(As_min, As_long * (state.get("util", 1.0) ** (1/state.get("alpha", 1.0))))
    
    out = {
        "b (mm)": state["b"], "D (mm)": state["D"], "fck (MPa)": state["fck"], "fy (MPa)": state["fy"],
        "Pu (kN)": state["Pu"] / 1e3, "Mux′ (kNm)": kNm(Mux_eff), "Muy′ (kNm)": kNm(Muy_eff), "Vu (kN)": state["Vu"] / 1e3,
        "--- STRENGTH RESULTS ---": "---",
        "Slenderness λx": f"{state['lam_x']:.1f}", "Slenderness λy": f"{state['lam_y']:.1f}",
        "Moment Magnifier δx": f"{state['delta_x']:.2f}", 
        "Uniaxial Capacity Mux,lim (kNm)": kNm(Mux_lim),
        "Biaxial Utilization (≤1)": f"{state['util']:.2f}",
        "Concrete Shear Vc (kN)": state["Vc"] / 1e3, "Vus (kN)": state["Vus"] / 1e3,
        "--- DETAILING RESULTS ---": "---",
        "Ast Provided (mm²)": f"{As_long:.0f}",
        "Ast Governing Required (mm²)": f"{As_governing:.0f}",
        "Tie Spacing Provided (mm)": f"{state['tie_spacing']:.0f}",
        "Tie Spacing Governing Req (mm)": f"{s_governing_tie:.0f}",
    }

    df_out = pd.DataFrame({"Parameter": list(out.keys()), "Value": list(out.values())})
    st.dataframe(df_out, use_container_width=True)
    
    st.caption("Notes: Uniaxial capacities are computed via full strain compatibility. All calculations follow IS 456 (2000) and IS 13920 (2016) principles.")
