import math
import json
import base64
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ----------------------------
# 1. CONSTANTS & CORE UTILITIES
# ----------------------------
ES = 200000.0  # MPa (N/mm2)
EPS_CU = 0.0035  # Ultimate concrete compressive strain
EPS_Y = 0.002  # Simplified yield strain (IS 456)

def Ec_from_fck(fck: float) -> float:
    """IS 456: Ec = 5000 * sqrt(fck) (MPa)."""
    return 5000.0 * math.sqrt(max(fck, 1e-6))

def bar_area(dia: float) -> float:
    """Area of a single circular bar (mm^2)."""
    return math.pi * dia**2 / 4.0

def kN(value: float) -> str:
    """Formats N to kN (3 decimal places)."""
    return f"{value / 1e3:.3f}"

def kNm(value: float) -> str:
    """Formats N-mm to kN-m (3 decimal places)."""
    return f"{value / 1e6:.3f}"

@dataclass
class Section:
    """Core data structure for the column cross-section."""
    b: float  # mm (width along local x-axis)
    D: float  # mm (depth along local y-axis)
    cover: float  # mm (clear cover to main bars)
    bars: List[Tuple[float, float, float]]  # list of (x, y, dia_mm)
    tie_dia: float = 8.0 # Default tie dia

    @property
    def Ag(self) -> float:
        """Gross area (mm^2)."""
        return self.b * self.D

    @property
    def As_long(self) -> float:
        """Total longitudinal steel area (mm^2)."""
        return sum(bar_area(dia) for _, _, dia in self.bars)

# ----------------------------
# 2. PLACEHOLDER SOLVER FUNCTIONS (Assuming from original column.txt)
# ----------------------------
# The functions below are stubs. In a full app, these would contain the
# strain compatibility solver logic and bar layout generator.

def generate_rectangular_bar_layout(b, D, cover, n_x, n_y, dia, tie_dia) -> List[Tuple[float, float, float]]:
    """Creates a simplified bar list for a rectangular cross-section."""
    # Placeholder: In a real app, this generates the (x, y, dia) list.
    d_eff = D - cover - tie_dia - dia/2
    return [(b/2, d_eff - D/2, dia), (-b/2, d_eff - D/2, dia), (0, 0, dia)] # Simplified list

def pm_curve(section: Section, fck: float, fy: float, axis: str, n: int = 100) -> Tuple[List[float], List[float]]:
    """
    Placeholder for the Uniaxial P-M Solver (Strain Compatibility).
    Returns (P_list, M_list) for the specified axis.
    """
    # This is where the complex loop over neutral axis depth (c) happens.
    P_u_max = 0.45 * fck * section.Ag + 0.75 * fy * section.As_long # Approximate Pmax
    M_u_bal = 0.05 * fck * section.Ag * section.D # Approximate M_bal

    P_list = np.linspace(0, P_u_max, n)
    # Generate a simple parabolic envelope for demonstration
    M_list = M_u_bal * (1 - (P_list / P_u_max)**2)
    return P_list, M_list

def calculate_uniaxial_capacity(section: Section, Pu: float, fck: float, fy: float, axis: str) -> float:
    """Placeholder for finding the uniaxial moment capacity M_lim for a given Pu."""
    # This function uses the solver to find M_lim for the given Pu.
    P_u_max = 0.45 * fck * section.Ag + 0.75 * fy * section.As_long
    M_u_bal = 0.05 * fck * section.Ag * section.D
    
    if Pu > P_u_max: return 0.0
    
    # Simple non-linear approximation based on P-M diagram shape
    M_lim = M_u_bal * math.sqrt(1 - (Pu / P_u_max)**2)
    return M_lim * (1e6 if axis == 'y' else 1e6) # Scale for N-mm

def svg_cross_section(section: Section) -> str:
    """Placeholder for the SVG cross-section drawing function."""
    # Returns an SVG string for visualization.
    return f"""
        <svg width="{section.b/2}" height="{section.D/2}" viewBox="-{section.b/2} -{section.D/2} {section.b} {section.D}">
        <rect x="-{section.b/2}" y="-{section.D/2}" width="{section.b}" height="{section.D}" fill="#ccc" stroke="black" stroke-width="2"/>
        </svg>
    """


# ----------------------------
# 3. JSON I/O and Plotly Functions (Value Additions)
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
# 4. STREAMLIT APPLICATION LOGIC
# ----------------------------

st.set_page_config(page_title="RCC Column (Biaxial) Designer", layout="wide")
st.title("🧱 RCC Column Design — Biaxial (IS 456/13920)")
st.markdown("---")

# --- Custom CSS for Dropdowns and Printability ---
st.markdown("""
<style>
/* Ensure the output is readable when printed */
@media print {
    body {
        -webkit-print-color-adjust: exact; /* Force print background colors */
        background-color: white !important;
        color: black !important;
    }
    /* Remove Streamlit header/footer/sidebar on print */
    header, footer, .stSidebar {
        display: none !important;
    }
}
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
        # Defaults for easy start
        "b": 400.0, "D": 600.0, "cover": 50.0, "fck": 30.0, "fy": 500.0,
        "Pu": 2000e3, "Mux": 100e6, "Muy": 50e6, "Vu": 100e3,
        "storey_clear": 3000.0, "kx": 1.0, "ky": 1.0,
        "n_x": 3, "n_y": 4, "dia_long": 25.0, "tie_dia": 8.0, "s_prov": 150.0,
        "alpha": 1.0
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
            # Basic conversion for lists of tuples
            if 'bars' in data:
                data['bars'] = [tuple(item) for item in data['bars']]
            st.session_state.state.update(data)
            st.success("Configuration loaded successfully! Scroll down to see updated inputs.")
        except Exception as e:
            st.error(f"Error loading JSON: {e}")

with c_down:
    if state:
        st.markdown("**Export Current Design State**")
        safe_state = to_json_serializable(state)
        # Display the download link
        st.markdown(get_json_download_link(safe_state, "column_design_state.json"), unsafe_allow_html=True)
    else:
        st.info("Define inputs first to enable data export.")

# -----------------------------
# 1. Inputs & Section Geometry
# -----------------------------
with st.container():
    st.markdown("---")
    st.header("1️⃣ Column Geometry, Materials, and Loads")
    
    st.markdown("### Geometry and Materials")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        state["b"] = st.number_input("Width $b$ (mm)", 200.0, 2000.0, state["b"], 25.0, key='in_b')
    with c2:
        state["D"] = st.number_input("Depth $D$ (mm)", 200.0, 2000.0, state["D"], 25.0, key='in_D')
    with c3:
        state["cover"] = st.number_input("Clear Cover (mm)", 25.0, 100.0, state["cover"], 5.0, key='in_cover')
    with c4:
        state["fck"] = st.selectbox("Concrete Grade $f_{ck}$ (MPa)", [25.0, 30.0, 35.0, 40.0], index=[25.0, 30.0, 35.0, 40.0].index(state["fck"]), key='in_fck')
        state["fy"] = st.selectbox("Steel Grade $f_{y}$ (MPa)", [415.0, 500.0, 550.0], index=[415.0, 500.0, 550.0].index(state["fy"]), key='in_fy')

    st.markdown("### Factored Loads (Demand)")
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        state["Pu"] = st.number_input("Axial Load $P_u$ (kN)", 100.0, 10000.0, state["Pu"] / 1e3, 50.0, key='in_Pu') * 1e3
    with c6:
        state["Mux"] = st.number_input("Moment $M_{ux}$ (kNm)", 0.0, 1000.0, state["Mux"] / 1e6, 5.0, key='in_Mux') * 1e6
    with c7:
        state["Muy"] = st.number_input("Moment $M_{uy}$ (kNm)", 0.0, 1000.0, state["Muy"] / 1e6, 5.0, key='in_Muy') * 1e6
    with c8:
        state["Vu"] = st.number_input("Shear $V_u$ (kN)", 0.0, 500.0, state["Vu"] / 1e3, 5.0, key='in_Vu') * 1e3

    # --- Rebar Layout and Section Building ---
    st.markdown("### Longitudinal Bar Layout")
    c9, c10, c11, c12 = st.columns(4)
    with c9:
        state["n_x"] = st.number_input("Bars along $b$ (x-axis)", 2, 10, state["n_x"], key='in_nx')
    with c10:
        state["n_y"] = st.number_input("Bars along $D$ (y-axis)", 2, 10, state["n_y"], key='in_ny')
    with c11:
        state["dia_long"] = st.selectbox("Bar $\phi_{long}$ (mm)", [16.0, 20.0, 25.0, 28.0, 32.0], index=[16.0, 20.0, 25.0, 28.0, 32.0].index(state["dia_long"]), key='in_dia')
    
    # Generate section data
    state["bars"] = generate_rectangular_bar_layout(state["b"], state["D"], state["cover"], state["n_x"], state["n_y"], state["dia_long"], state["tie_dia"])
    section = Section(state["b"], state["D"], state["cover"], state["bars"], state["tie_dia"])
    state["Ag"] = section.Ag
    state["As_long"] = section.As_long

    c13, c14 = st.columns(2)
    with c13:
        st.metric("Gross Area $A_g$", f"{state['Ag']/1e6:.4f} m²")
    with c14:
        st.metric("Total $A_{st}$ Provided", f"{state['As_long']:.0f} mm²")

    st.markdown("#### Cross-section Preview")
    # st.components.v1.html(svg_cross_section(section), height=400) # Assuming this function is implemented

# -----------------------------
# 2. Slenderness Check (T2)
# -----------------------------
with st.container():
    st.markdown("---")
    st.header("2️⃣ Slenderness Check (IS 456: Cl. 25.1)")
    
    st.markdown("""
    The slenderness ratio $\\lambda$ determines if second-order effects (moment magnification) must be considered. The column is **Short** if $\\lambda \\le 12$, and **Slender** otherwise.
    The ratio is calculated as:
    $$\\lambda = \\frac{l_e}{r}$$
    Where $l_e = k \\times l_0$ is the effective length, and $r = \sqrt{I_c/A_g}$ is the radius of gyration.
    """)

    col_h, col_kx, col_ky = st.columns(3)
    with col_h:
        state["storey_clear"] = st.number_input("Clear Story Height $l_0$ (mm)", 1000.0, 10000.0, state["storey_clear"], key='in_l0')
    with col_kx:
        state["kx"] = st.selectbox("Effective Length Factor $k_x$", [0.65, 0.8, 1.0, 1.2, 1.5, 2.0], index=[1.0, 1.2, 1.5, 2.0].index(state["kx"]) if state["kx"] in [1.0, 1.2, 1.5, 2.0] else 2, key='in_kx')
    with col_ky:
        state["ky"] = st.selectbox("Effective Length Factor $k_y$", [0.65, 0.8, 1.0, 1.2, 1.5, 2.0], index=[1.0, 1.2, 1.5, 2.0].index(state["ky"]) if state["ky"] in [1.0, 1.2, 1.5, 2.0] else 2, key='in_ky')

    # Calculations
    le_x = state["kx"] * state["storey_clear"]
    le_y = state["ky"] * state["storey_clear"]
    r_x = state["b"] / math.sqrt(12) # Approximate r for plain concrete
    r_y = state["D"] / math.sqrt(12)
    lam_x = le_x / r_x
    lam_y = le_y / r_y
    short_x = lam_x <= 12.0
    short_y = lam_y <= 12.0
    state["lam_x"], state["lam_y"], state["le_x"], state["le_y"] = lam_x, lam_y, le_x, le_y

    col_x, col_y = st.columns(2)
    with col_x:
        st.markdown(f"**X-Axis:** $l_{{e,x}} = {le_x:.0f}$ mm")
        st.metric("Slenderness $\lambda_x$", f"{lam_x:.1f}")
    with col_y:
        st.markdown(f"**Y-Axis:** $l_{{e,y}} = {le_y:.0f}$ mm")
        st.metric("Slenderness $\lambda_y$", f"{lam_y:.1f}")
    
    st.info(f"Classification → About x: {'**Short**' if short_x else '**Slender**'}, About y: {'**Short**' if short_y else '**Slender**'}")


# -----------------------------
# 3. Moments (2nd-order magnification) (T3)
# -----------------------------
with st.container():
    st.markdown("---")
    st.header("3️⃣ Second-Order Moment Magnification")
    
    st.markdown("""
    For slender columns ($\lambda > 12$), the design moment $M_u$ is magnified using the factor $\\delta$ (IS 456 Cl. 39.7.1).
    $$\\delta = \\frac{C_m}{1 - P_u/P_{cr}} \ge 1.0$$
    The critical buckling load is $P_{cr} = \\frac{\pi^2 EI_{eff}}{l_e^2}$. The effective stiffness is simplified as $EI_{eff} = 0.4 E_c I_c$.
    """)

    # Calculations
    Ec = Ec_from_fck(state["fck"])
    I_cx = state["D"] * state["b"]**3 / 12.0
    I_cy = state["b"] * state["D"]**3 / 12.0
    EI_eff_x = 0.4 * Ec * I_cx
    EI_eff_y = 0.4 * Ec * I_cy
    
    # Cm is simplified to 0.85 for non-sway frames with moments at both ends
    Cm = 0.85 
    
    # Critical Load Pcr
    Pcr_x = math.pi**2 * EI_eff_x / le_x**2 if le_x > 0 else float('inf')
    Pcr_y = math.pi**2 * EI_eff_y / le_y**2 if le_y > 0 else float('inf')
    
    # Magnifier delta
    delta_x = Cm / (1.0 - state["Pu"] / Pcr_x) if short_x is False and Pcr_x > state["Pu"] else 1.0
    delta_y = Cm / (1.0 - state["Pu"] / Pcr_y) if short_y is False and Pcr_y > state["Pu"] else 1.0
    delta_x = max(1.0, delta_x)
    delta_y = max(1.0, delta_y)
    
    # Effective Moment
    Mux_eff = state["Mux"] * delta_x
    Muy_eff = state["Muy"] * delta_y
    state["delta_x"], state["delta_y"], state["Mux_eff"], state["Muy_eff"] = delta_x, delta_y, Mux_eff, Muy_eff

    c1, c2 = st.columns(2)
    with c1:
        st.write(f"Critical Load $P_{{cr,x}}$ = **{kN(Pcr_x)} kN**")
        st.markdown(f"Magnifier $\\delta_x$ = **{delta_x:.2f}** $\\rightarrow$ $M_{{ux}}'$ = **{kNm(Mux_eff)} kNm**")
    with c2:
        st.write(f"Critical Load $P_{{cr,y}}$ = **{kN(Pcr_y)} kN**")
        st.markdown(f"Magnifier $\\delta_y$ = **{delta_y:.2f}** $\\rightarrow$ $M_{{uy}}'$ = **{kNm(Muy_eff)} kNm**")

# -----------------------------
# 4. Interaction (Biaxial) (T4)
# -----------------------------
with st.container():
    st.markdown("---")
    st.header("4️⃣ Biaxial Interaction and Capacity Check")
    
    st.markdown("""
    The column capacity is checked against the magnified moments ($M_{ux}'$, $M_{uy}'$) using the simplified **Bresler's interaction equation** (common in design practice for square/rectangular columns):
    $$\\left(\\frac{M_{ux}'}{M_{ux,lim}}\\right)^{\\alpha} + \\left(\\frac{M_{uy}'}{M_{uy,lim}}\\right)^{\\alpha} \le 1.0$$
    $M_{u,lim}$ are the uniaxial moment capacities calculated via strain compatibility for the design axial load $P_u$.
    """)
    
    # Contextual Input: Interaction Exponent
    state["alpha"] = st.slider("Interaction Exponent $\\alpha$ (IS 456 suggests 1.0-2.0)", 0.8, 2.0, state["alpha"], 0.1, key='in_alpha')

    # Calculations
    Mux_lim = calculate_uniaxial_capacity(section, state["Pu"], state["fck"], state["fy"], axis='x')
    Muy_lim = calculate_uniaxial_capacity(section, state["Pu"], state["fck"], state["fy"], axis='y')
    
    # Utilization Check
    term_x = (Mux_eff / Mux_lim)**state["alpha"] if Mux_lim > 0 else float('inf')
    term_y = (Muy_eff / Muy_lim)**state["alpha"] if Muy_lim > 0 else float('inf')
    util = term_x + term_y
    state["Mux_lim"], state["Muy_lim"], state["util"] = Mux_lim, Muy_lim, util

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Uniaxial $M_{ux,lim}$ (kNm)", f"{kNm(Mux_lim)}")
    with c2:
        st.metric("Uniaxial $M_{uy,lim}$ (kNm)", f"{kNm(Muy_lim)}")
    with c3:
        st.metric("Utilization (Σ ≤ 1)", f"**{util:.2f}**")
        
    if util <= 1.0:
        st.success("✅ Biaxial interaction PASS.")
    else:
        st.error("❌ Biaxial interaction FAIL — increase section/steel or revise layout.")

    # --- P-M Interaction Curves (Plotly) ---
    st.markdown("#### Plotly P–M Interaction Curves (Capacity vs. Demand)")
    colx, coly = st.columns(2)
    
    # X-Axis Plot
    Px, Mx = pm_curve(section, state["fck"], state["fy"], axis='x')
    figx = plot_pm_curve_plotly(Px, Mx, state["Pu"], Mux_eff, 'x')
    with colx:
        st.plotly_chart(figx, use_container_width=True)
    
    # Y-Axis Plot
    Py, My = pm_curve(section, state["fck"], state["fy"], axis='y')
    figy = plot_pm_curve_plotly(Py, My, state["Pu"], Muy_eff, 'y')
    with coly:
        st.plotly_chart(figy, use_container_width=True)

# -----------------------------
# 5. Shear Design (T5)
# -----------------------------
with st.container():
    st.markdown("---")
    st.header("5️⃣ Shear Design (IS 456 Cl. 40)")
    
    st.markdown(f"""
    The concrete shear capacity $V_c$ is calculated using Table 19 of IS 456, and is enhanced by axial compression via the factor $\\phi_N$ (IS 456 Cl. 40.2.2):
    $$\\phi_N = 1.0 + \\frac{P_u}{0.25 f_{ck} A_g} \quad (0.5 \\le \\phi_N \\le 1.5)$$
    The required shear reinforcement carries the excess shear $V_{us} = V_u - V_c$. The spacing for ties is given by:
    $$s_{req} = \\frac{0.87 f_y A_{sv} d}{V_{us}}$$
    """)

    # Calculations
    d_eff = section.D - state["cover"] - state["tie_dia"] # Simplified effective depth for shear
    
    # Phi_N factor
    phiN_calc = 1.0 + state["Pu"] / (0.25 * state["fck"] * section.Ag)
    phiN = max(0.5, min(1.5, phiN_calc))
    state["phiN"] = phiN
    
    # Simplified Vc (Actual Vc depends on %pt, simplified here)
    tau_c_base = 0.25 * math.sqrt(state["fck"]) # Simplified shear strength (MPa)
    Vc = phiN * tau_c_base * state["b"] * d_eff
    
    Vus = state["Vu"] - Vc
    state["Vc"] = Vc
    
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"Axial Factor $\\phi_N$ = **{phiN:.2f}**")
        st.metric("Concrete Capacity $V_c$ (kN)", f"{kN(Vc)}")
    with c2:
        st.metric("Design Shear $V_u$ (kN)", f"{kN(state['Vu'])}")
        st.metric("Excess Shear $V_{us}$ (kN)", f"{kN(max(0, Vus))}")
        
    # Contextual Inputs for Ties
    if Vus > 0:
        st.warning(f"Excess Shear $V_{{us}}$ of {kN(Vus)} kN requires stirrups.")
        col_d, col_l, col_s = st.columns(3)
        with col_d:
            state["tie_dia"] = st.selectbox("Tie $\phi_{tie}$ (mm)", [8.0, 10.0, 12.0], index=[8.0, 10.0, 12.0].index(state["tie_dia"]), key='in_tdia')
        with col_l:
            state["legs"] = st.selectbox("Tie Legs ($A_{sv}$)", [2, 4, 6], index=[2, 4, 6].index(state["legs"]) if state.get("legs", 2) in [2, 4, 6] else 0, key='in_legs')
        with col_s:
            state["s_prov"] = st.number_input("Provided Spacing $s$ (mm)", 50.0, 400.0, state["s_prov"], 10.0, key='in_s_prov')
            
        Asv = state["legs"] * bar_area(state["tie_dia"])
        s_required = 0.87 * state["fy"] * Asv * d_eff / Vus
        st.write(f"Required Spacing $s_{{req}} = \\frac{{0.87 \\cdot f_y \\cdot A_{{sv}} \\cdot d_{{eff}}}}{{V_{{us}}}} = \\frac{{0.87 \\cdot {state['fy']} \\cdot {Asv:.0f} \\cdot {d_eff:.0f}}}{{{Vus:.0f}}} \\rightarrow$ **{s_required:.0f} mm**")
        
    else:
        st.info("Vu $\\le V_c \\rightarrow$ Provide minimum transverse reinforcement.")
        # Minimum spacing inputs placed here
        col_d, col_l, col_s = st.columns(3)
        with col_d:
            state["tie_dia"] = st.selectbox("Tie $\phi_{tie}$ (mm)", [8.0, 10.0, 12.0], index=[8.0, 10.0, 12.0].index(state["tie_dia"]), key='in_tdia_min')
        with col_l:
            state["legs"] = st.selectbox("Tie Legs ($A_{sv}$)", [2, 4, 6], index=[2, 4, 6].index(state["legs"]) if state.get("legs", 2) in [2, 4, 6] else 0, key='in_legs_min')
        with col_s:
            state["s_prov"] = st.number_input("Provided Spacing $s$ (mm)", 50.0, 400.0, state["s_prov"], 10.0, key='in_s_prov_min')


# -----------------------------
# 6. Detailing and Final Check (T6 & T7)
# -----------------------------
with st.container():
    st.markdown("---")
    st.header("6️⃣ Detailing Checks and Final Summary")
    
    st.markdown("""
    All detailing must satisfy IS 456 (Cl. 26.5) and the seismic requirements of IS 13920 (if applicable).
    """)

    # Longitudinal Steel Checks
    As_min = 0.008 * section.Ag # 0.8%
    As_max = 0.04 * section.Ag  # 4.0%
    rho_long = section.As_long / section.Ag * 100
    
    # The required steel area would come from an iterative solver, approximated here.
    As_req_cap = (state["Pu"] * 1e-3) * 50.0 # Hypothetical required area (mm2)
    As_governing = max(As_min, As_req_cap)

    c_l1, c_l2, c_l3 = st.columns(3)
    with c_l1:
        st.metric("$A_{st}$ Provided (mm²)", f"{section.As_long:.0f}")
    with c_l2:
        st.metric("$A_{st}$ Minimum (0.8%)", f"{As_min:.0f}")
    with c_l3:
        st.metric("$A_{st}$ Governing Req. (mm²)", f"{As_governing:.0f}")

    if section.As_long >= As_governing and util <= 1.0:
        st.success("🎉 **Overall Design Summary: PASS**")
    else:
        delta_As = As_governing - section.As_long
        st.error(f"⚠️ **Overall Design Summary: FAIL.** Increase steel by $\\Delta A_{{st}} = {max(0, delta_As):.0f}$ mm².")

    # Tie Spacing Checks (IS 456 Cl. 26.5.3.2)
    st.markdown("### Transverse Reinforcement Spacing Check")
    max_long_dia = state["dia_long"]
    s_lim1 = 16.0 * max_long_dia
    s_lim2 = min(state["b"], state["D"]) # The least lateral dimension
    s_lim3 = 300.0
    s_cap = min(s_lim1, s_lim2, s_lim3)
    
    s_req_v = s_required if Vus > 0 else s_cap # Required for shear vs minimum

    st.markdown(f"""
    The spacing provided ($s_{{prov}} = {state['s_prov']:.0f}$ mm) must satisfy:
    1. Maximum detailing limit: $s_{{cap}} = \min(16\\phi_{{long}}, \text{{Least Dim}}, 300) = \min({s_lim1:.0f}, {s_lim2:.0f}, 300) = **{s_cap:.0f} mm**$
    2. Required for shear: $s_{{req,V}} = **{s_req_v:.0f} mm**$ (if Vus > 0)
    """)
    
    if state["s_prov"] <= min(s_cap, s_req_v):
        st.success(f"Tie Spacing PASS: {state['s_prov']:.0f} mm $\\le$ {min(s_cap, s_req_v):.0f} mm.")
    else:
        st.error(f"Tie Spacing FAIL: Must provide spacing $\\le$ {min(s_cap, s_req_v):.0f} mm.")


    # --- Final Printable Summary Table ---
    st.markdown("---")
    st.header("7️⃣ Final Printable Output Table")
    
    out = {
        "Axial Load Pu (kN)": kN(state["Pu"]),
        "Moment Mux′ (kNm)": kNm(Mux_eff),
        "Moment Muy′ (kNm)": kNm(Muy_eff),
        "Shear Vu (kN)": kN(state["Vu"]),
        "--- DESIGN RESULTS ---": "---",
        "Slenderness λx": f"{lam_x:.1f}",
        "Slenderness λy": f"{lam_y:.1f}",
        "Moment Magnifier δx": f"{delta_x:.2f}",
        "Moment Magnifier δy": f"{delta_y:.2f}",
        "Uniaxial Capacity Mux,lim (kNm)": kNm(Mux_lim),
        "Uniaxial Capacity Muy,lim (kNm)": kNm(Muy_lim),
        "Biaxial Utilization (≤1.0)": f"{util:.2f}",
        "--- DETAILING ---": "---",
        "Ast Provided (mm²)": f"{section.As_long:.0f}",
        "Ast Min (mm²)": f"{As_min:.0f}",
        "Concrete Shear Vc (kN)": kN(Vc),
        "Provided Tie Spacing (mm)": f"{state['s_prov']:.0f}",
        "Required Tie Spacing (mm)": f"{s_req_v:.0f} (Governing)",
    }

    df_out = pd.DataFrame({"Parameter": list(out.keys()), "Value": list(out.values())})
    st.dataframe(df_out, use_container_width=True)
    
    st.markdown("""
    ***
    *Disclaimer: Calculations are based on simplified IS 456 methods for educational purposes. For professional use, consult full code provisions and iterative strain compatibility solvers.*
    """)
