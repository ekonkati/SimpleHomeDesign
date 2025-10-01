import math
import json
import base64
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ------------------------- 1. CONSTANTS AND UTILITIES -------------------------
ES = 200000.0  # MPa (Modulus of Elasticity of Steel)
EPS_CU = 0.0035  # Ultimate concrete compressive strain

def bar_area(dia_mm: float) -> float:
    """Calculates the area of a single bar."""
    return math.pi * (dia_mm ** 2) / 4.0

def kN(value: float) -> str:
    """Formats force in kN."""
    return f"{value / 1e3:.1f}"

def kNm(value: float) -> str:
    """Formats moment in kNm."""
    return f"{value / 1e6:.1f}"

def effective_length_factor(restraint: str) -> float:
    """IS 456-2000 Table 28 approximation for k-factor."""
    if restraint == "Fixed-Fixed": return 0.65
    if restraint == "Fixed-Pinned": return 0.8
    if restraint == "Pinned-Pinned": return 1.0
    if restraint == "Fixed-Free (cantilever)": return 2.0
    return 1.0

def moment_magnifier(Pu: float, le_mm: float, fck: float, Ic: float, Cm: float = 0.85, sway: bool = False) -> float:
    """Calculates moment magnification factor (Delta) based on IS 456 Cl. 39.7."""
    Ec = 5000.0 * math.sqrt(max(fck, 1e-6))
    
    # Critical buckling load Pcr = (pi^2 * 0.4 * Ec * Ic) / (le^2) - IS 456 Cl. 39.7.1
    Pcr = (math.pi ** 2) * 0.4 * Ec * Ic / (le_mm ** 2 + 1e-9)
    
    if Pcr <= Pu: return 10.0  # Buckling imminent
    
    # Magnification factor Delta = 1 / (1 - Pu/Pcr)
    delta = 1.0 / max(1e-6, (1.0 - Pu / Pcr))
    
    if not sway:
        delta = max(1.0, Cm * delta)
    
    return float(np.clip(delta, 1.0, 2.5) if not sway else np.clip(delta, 1.0, 5.0))

def to_json_serializable(state: dict) -> dict:
    """Converts numpy types and tuples to basic Python types for JSON serialization."""
    safe_state = {}
    for key, value in state.items():
        if isinstance(value, float): safe_state[key] = round(value, 6)
        elif isinstance(value, np.float64): safe_state[key] = round(float(value), 6)
        elif isinstance(value, (list, tuple)) and all(isinstance(v, tuple) for v in value):
            safe_state[key] = [list(item) for item in value]
        else: safe_state[key] = value
    return safe_state

def get_json_download_link(data_dict: dict, filename: str) -> str:
    """Generates a downloadable JSON file link."""
    json_str = json.dumps(data_dict, indent=4)
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'<a href="data:file/json;base64,{b64}" download="{filename}">💾 Download Design State</a>'
    return href

# ------------------------- 2. REBAR & CAPACITY LOGIC -------------------------

def _linspace_points(a: float, c: float, n: int) -> List[float]:
    """Generates equally spaced points between a and c (inclusive) for bar layout."""
    if n <= 0: return []
    if n == 1: return [a + (c - a) / 2.0]
    return [a + i * (c - a) / (n - 1) for i in range(n)]

def _generate_bar_layout(b: float, D: float, cover: float, state: dict) -> List[Tuple[float, float, float]]:
    """Calculates the (x, y, dia) coordinates for all longitudinal bars."""
    n_top, n_bot, n_left, n_right = state["n_top"], state["n_bot"], state["n_left"], state["n_right"]
    dia_top, dia_bot, dia_side = state["dia_top"], state["dia_bot"], state["dia_side"]

    bars_list = []
    max_dia = max(dia_top, dia_bot, dia_side)
    dx_corner = cover + state["tie_dia"] + max_dia / 2.0
    
    if dx_corner > min(b, D) / 2.0: dx_corner = min(b, D) / 2.0 - 5.0

    # Top Bars (y = D - dx_corner)
    if n_top > 0:
        x_coords = _linspace_points(dx_corner, b - dx_corner, n_top)
        for x in x_coords: bars_list.append((x, D - dx_corner, dia_top))
    
    # Bottom Bars (y = dx_corner)
    if n_bot > 0:
        x_coords = _linspace_points(dx_corner, b - dx_corner, n_bot)
        for x in x_coords: bars_list.append((x, dx_corner, dia_bot))
            
    # Side Bars (Left/Right)
    y_start, y_end = dx_corner, D - dx_corner
    
    if n_left > 0:
        y_coords_l = _linspace_points(y_start, y_end, n_left)
        for y in y_coords_l: bars_list.append((dx_corner, y, dia_side))
        
    if n_right > 0:
        y_coords_r = _linspace_points(y_start, y_end, n_right)
        for y in y_coords_r: bars_list.append((b - dx_corner, y, dia_side))

    # Remove duplicate bars at corners
    unique_locations = {}
    for x, y, dia in bars_list:
        loc = (round(x, 4), round(y, 4))
        if loc not in unique_locations or dia > unique_locations[loc][2]:
            unique_locations[loc] = (x, y, dia)
            
    return sorted([v for v in unique_locations.values()], key=lambda x: (x[1], x[0]))

def _uniaxial_capacity_Mu_for_Pu(b: float, D: float, bars: List[Tuple[float, float, float]], fck: float, fy: float, Pu: float, axis: str) -> float:
    """
    Calculates the uniaxial moment capacity Mu_lim using Strain Compatibility 
    for a given axial load Pu (IS 456 Limit State).
    """
    dimension = D if axis == 'x' else b # Dimension perpendicular to the moment axis
    
    def forces_and_moment(c: float):
        """Calculates total Axial Force (N) and Moment (M) for a given Neutral Axis Depth c."""
        xu_max = min(c, dimension)
        Cc = 0.36 * fck * dimension * xu_max
        arm_Cc = 0.5 * dimension - 0.42 * xu_max
        Mc = Cc * arm_Cc
        
        Fs, Ms = 0.0, 0.0
        
        # Steel forces (Fs)
        for (x_abs, y_abs, dia) in bars:
            As = bar_area(dia)
            y = (D - y_abs) if axis == 'x' else x_abs # d' in compression zone
            
            # Strain at the steel level
            strain = EPS_CU * (1.0 - (y / max(c, 1e-6)))
            
            # Steel Stress (IS 456-2000 simplification: 0.87*fy yield)
            stress = np.clip(ES * strain, -0.87 * fy, 0.87 * fy)
            force = stress * As
            
            # Distance from geometric center to the steel layer
            z = 0.5 * dimension - y 
            
            Fs += force 
            Ms += force * z
            
        return Cc + Fs, Mc + Ms

    target = Pu 
    
    # Binary search for the neutral axis depth 'c'
    cL, cR = 0.01 * dimension, 1.50 * dimension 
    NL, ML = forces_and_moment(cL)
    NR, MR = forces_and_moment(cR)

    if target <= NL: return ML 
    if target >= NR: return MR 

    for _ in range(60): 
        cm = 0.5 * (cL + cR)
        Nm, Mm = forces_and_moment(cm)
        
        if abs(Nm - target) < 1.0: 
            return float(Mm)
        
        if (NL - target) * (Nm - target) <= 0:
            cR, NR, MR = cm, Nm, Mm
        else:
            cL, NL, ML = cm, Nm, Mm
    
    return float(0.5 * (ML + MR))

def biaxial_utilization(b: float, D: float, bars: List[Tuple[float, float, float]], fck: float, fy: float, Pu: float, Mux_eff: float, Muy_eff: float, alpha: float) -> Tuple[float, float, float]:
    """Performs the biaxial interaction check using the Power Law (IS 456 Annex G)."""
    
    Mux_lim = _uniaxial_capacity_Mu_for_Pu(b, D, bars, fck, fy, Pu, axis='x')
    Muy_lim = _uniaxial_capacity_Mu_for_Pu(b, D, bars, fck, fy, Pu, axis='y')
    
    Mux_lim = max(Mux_lim, 1e-3) if abs(Mux_eff) > 1e-3 else Mux_lim
    Muy_lim = max(Muy_lim, 1e-3) if abs(Muy_eff) > 1e-3 else Muy_lim
        
    # Power Law Interaction Equation
    Rx = (abs(Mux_eff) / Mux_lim) ** alpha
    Ry = (abs(Muy_eff) / Muy_lim) ** alpha
    util = Rx + Ry
    
    return util, Mux_lim, Muy_lim

# ------------------------- 3. PLOTLY FUNCTIONS -------------------------

def plotly_cross_section(b: float, D: float, cover: float, bars: List[Tuple[float, float, float]], tie_dia: float, tie_spacing: float) -> go.Figure:
    """Generates a Plotly figure of the column cross-section and rebar layout."""
    fig = go.Figure()
    
    # Concrete Outline
    fig.add_shape(type="rect", x0=0, y0=0, x1=b, y1=D, line=dict(color="black", width=2), fillcolor="rgba(240, 240, 240, 0.8)"))
    
    # Effective Cover Boundary
    effective_cover = cover + tie_dia
    fig.add_shape(type="rect", x0=effective_cover, y0=effective_cover, x1=b - effective_cover, y1=D - effective_cover, 
                  line=dict(color="gray", width=1, dash="dot"), fillcolor="rgba(0,0,0,0)"))
    
    # Rebar scatter plot
    bar_x, bar_y, bar_size, bar_text = [], [], [], []
    for x_geom, y_geom, dia in bars:
        bar_x.append(x_geom)
        bar_y.append(y_geom)
        bar_size.append(dia * 2) 
        bar_text.append(f"Ø{dia:.0f} mm ({bar_area(dia):.0f} mm²)")
        
    fig.add_trace(go.Scatter(x=bar_x, y=bar_y, mode='markers', name='Rebars',
        marker=dict(size=bar_size, sizemode='diameter', color='#0a66c2', line=dict(width=1, color='DarkSlateGrey')),
        hovertemplate='Size: %{text}<extra></extra>', text=bar_text))
        
    # Title and Annotation
    fig.add_annotation(x=b/2, y=D + 10, text=f"Ties: Ø{int(tie_dia)} @ {int(tie_spacing)} mm",
                       showarrow=False, yanchor="bottom")
                       
    fig.update_layout(title=f"Cross Section (b={b:.0f} x D={D:.0f} mm)", 
                      width=400, height=400 * D / b if b != 0 else 400, 
                      showlegend=False, hovermode="closest", plot_bgcolor='white', 
                      yaxis_scaleanchor="x", yaxis_scaleratio=1,
                      xaxis_title="Width b (mm)", yaxis_title="Depth D (mm)")
    return fig

def plot_pm_curve_plotly(b: float, D: float, bars: List[Tuple[float, float, float]], fck: float, fy: float, Pu: float, Mu_eff: float, axis: str):
    """Generates a Plotly P-M interaction curve for a given axis (x or y)."""
    dimension = D if axis == 'x' else b
    
    cs = np.linspace(0.01 * dimension, 1.50 * dimension, 120)
    P_list, M_list = [], []
    
    for c in cs:
        Cc = 0.36 * fck * dimension * min(c, dimension)
        xu_max = min(c, dimension)
        arm_Cc = 0.5 * dimension - 0.42 * xu_max
        Mc = Cc * arm_Cc
        Fs, Ms = 0.0, 0.0
        
        for (x_abs, y_abs, dia) in bars:
            As = bar_area(dia)
            y = (D - y_abs) if axis == 'x' else x_abs 
            strain = EPS_CU * (1.0 - (y / max(c, 1e-6)))
            stress = np.clip(ES * strain, -0.87 * fy, 0.87 * fy)
            force = stress * As
            z = 0.5 * dimension - y 
            Fs += force; Ms += force * z
            
        P_list.append(Cc + Fs)
        M_list.append(abs(Mc + Ms))

    df = pd.DataFrame({'P (kN)': np.array(P_list)/1e3, f'M_{axis} (kNm)': np.array(M_list)/1e6})
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['P (kN)'], y=df[f'M_{axis} (kNm)'], mode='lines', 
                             name='Capacity Envelope', line=dict(color='#0a66c2', width=3)))
                             
    # Demand Point
    fig.add_trace(go.Scatter(x=[Pu/1e3], y=[abs(Mu_eff)/1e6], mode='markers', 
                             name='Design Demand (Pu, Mu_eff)', 
                             marker=dict(symbol='x', size=12, color='red', line=dict(width=2, color='red'))))
                             
    fig.update_layout(title=f'P–M Capacity Envelope (Axis: {axis.upper()})', 
                      xaxis_title='Axial Load P (kN)', 
                      yaxis_title=f'Moment M_{axis} (kNm)', 
                      hovermode="x unified", 
                      margin=dict(l=20, r=20, t=40, b=20), 
                      plot_bgcolor='white')
    return fig

# ------------------------- 4. CORE CALCULATIONS -------------------------

def initialize_state():
    """Initializes default session state variables."""
    if "state" not in st.session_state:
        st.session_state.state = {
            "b": 450.0, "D": 600.0, "cover": 40.0, "fck": 30.0, "fy": 500.0,
            "Pu": 1200e3, "Mux": 120e6, "Muy": 80e6, "Vu": 150e3,
            "storey_clear": 3200.0, "restraint": "Pinned-Pinned", "sway": False,
            "n_top": 3, "n_bot": 3, "n_left": 2, "n_right": 2, 
            "dia_top": 16.0, "dia_bot": 16.0, "dia_side": 12.0, "tie_dia": 8.0, "tie_spacing": 150.0,
            "legs": 4, 
            "alpha": 1.0, "bars": [], "util": 0.0, "Vc": 0.0, "Vus": 0.0, "Ag": 0.0
        }

def recalculate_properties(state):
    """Performs all core engineering calculations and updates the state."""
    b, D, cover, fck, fy = state["b"], state["D"], state["cover"], state["fck"], state["fy"]
    Pu, Mux, Muy, Vu = state["Pu"], state["Mux"], state["Muy"], state["Vu"]
    lo = state["storey_clear"]

    # 1. Bar Layout & Area
    bars = _generate_bar_layout(b, D, cover, state)
    As_long = sum(bar_area(dia) for _, _, dia in bars)
    Ag = b * D
    Ic_x = b * D**3 / 12.0
    Ic_y = D * b**3 / 12.0
    rx = math.sqrt(Ic_x / Ag) 
    ry = math.sqrt(Ic_y / Ag)
    state.update({"As_long": As_long, "bars": bars, "Ag": Ag, "Ic_x": Ic_x, "Ic_y": Ic_y})

    # 2. Slenderness & Magnification
    k_factor = effective_length_factor(state["restraint"])
    le_x, le_y = k_factor * lo, k_factor * lo
    lam_x, lam_y = le_x / max(rx, 1e-6), le_y / max(ry, 1e-6)
    short_x = lam_x <= 12.0
    short_y = lam_y <= 12.0
    
    emin_x = max(lo/500.0 + D/30.0, 20.0)
    emin_y = max(lo/500.0 + b/30.0, 20.0)
    Mux_base = max(abs(Mux), abs(Pu * emin_x / 1000.0))
    Muy_base = max(abs(Muy), abs(Pu * emin_y / 1000.0))
    
    delta_x = moment_magnifier(Pu, le_x, fck, Ic_x, sway=state["sway"]) if not short_x else 1.0
    delta_y = moment_magnifier(Pu, le_y, fck, Ic_y, sway=state["sway"]) if not short_y else 1.0
    
    Mux_eff = Mux_base * delta_x
    Muy_eff = Muy_base * delta_y
    
    state.update(dict(le_x=le_x, le_y=le_y, lam_x=lam_x, lam_y=lam_y, short_x=short_x, short_y=short_y, 
                      delta_x=delta_x, delta_y=delta_y, Mux_base=Mux_base, Muy_base=Muy_base,
                      Mux_eff=Mux_eff, Muy_eff=Muy_eff, emin_x=emin_x, emin_y=emin_y))
    
    # 3. Biaxial Utilization
    util, Mux_lim, Muy_lim = biaxial_utilization(b, D, bars, fck, fy, Pu, Mux_eff, Muy_eff, state["alpha"])
    state.update(dict(Mux_lim=Mux_lim, Muy_lim=Muy_lim, util=util))

    # 4. Shear Design
    max_long_dia = max([bar[2] for bar in bars], default=16.0)
    d_eff = D - (cover + state["tie_dia"] + 0.5 * max_long_dia) 
    
    phiN = float(np.clip(1.0 + (Pu / max(1.0, (0.25 * fck * Ag))), 0.5, 1.5))
    tau_c_base = 0.62 * math.sqrt(fck) / 1.0 
    Vc = tau_c_base * b * d_eff * phiN 
    
    Vus = max(0.0, Vu - Vc) 
    state.update(dict(Vc=Vc, Vus=Vus, phiN=phiN, d_eff=d_eff))

    # 5. Tie Spacing Requirements 
    Asv = state["legs"] * bar_area(state["tie_dia"])
    
    s_required_shear = (0.87 * fy * Asv * d_eff) / max(Vus, 1e-6) if Vus > 0 else 300.0
    s_cap_detailing = min(16.0 * max_long_dia, min(b, D), 300.0)
    s_13920_confine = min(min(b,D) / 4.0, 100.0)
    s_governing_tie = min(s_cap_detailing, s_required_shear)
    
    state.update({"s_required_shear": s_required_shear, "s_cap_detailing": s_cap_detailing, 
                  "s_13920_confine": s_13920_confine, "s_governing_tie": s_governing_tie})
    
    return b, D, cover, bars, Ag

# ------------------------- 5. STREAMLIT APP LAYOUT -------------------------

def main():
    st.set_page_config(page_title="RCC Column (Biaxial) Designer", layout="wide")
    st.markdown("""
        <style>
        /* Styles for print and highlighting */
        .highlight {
            background: #fff8e1; /* Lighter yellow for emphasis */
            border: 2px solid #ffcc80; 
            border-radius: 12px;
            padding: 10px 15px;
            margin: 10px 0 20px 0;
        }
        .highlight .stSelectbox, .highlight .stNumberInput {
            background: #fffceb !important;
            padding: 5px;
            border-radius: 5px;
        }
        @media print {
            header, .stToolbar, .stAppDeployButton, .stActionButton, footer { display: none !important; }
            .block-container { padding: 0.6cm 1.0cm !important; max-width: 100% !important; }
            .print-break { page-break-before: always; }
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.title("🧱 RCC Column Designer — Single Canvas Output")
    st.caption("Integrated Biaxial $P-M$ and Shear Design (IS 456/13920). All calculations and inputs are below for a complete, printable report.")
    st.markdown("---")

    initialize_state()
    state = st.session_state.state
    
    # --- Data Management ---
    c_data_up, c_data_down = st.columns(2)
    with c_data_up:
        uploaded_file = st.file_uploader("Upload Saved Design State (JSON)", type="json")
        if uploaded_file is not None:
            try:
                data = json.load(uploaded_file)
                if 'bars' in data and isinstance(data['bars'], list): 
                    data['bars'] = [tuple(item) for item in data['bars']]
                st.session_state.state.update(data)
                st.rerun() 
            except Exception as e:
                st.error(f"Error loading JSON: {e}")
    with c_data_down:
        if state.get("bars"): 
            st.markdown(get_json_download_link(to_json_serializable(state), "column_design_state.json"), unsafe_allow_html=True)
            
    st.markdown("---")

    # --- INPUT SECTION: GEOMETRY, MATERIALS, LOADS (STEP 1) ---
    st.header("1️⃣ Geometry, Materials, and Factored Loads")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Section Geometry")
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        state["b"] = st.number_input("Width $b$ (mm)", 200.0, 2000.0, state["b"], 25.0, key='in_b')
        state["D"] = st.number_input("Depth $D$ (mm)", 200.0, 3000.0, state["D"], 25.0, key='in_D')
        state["cover"] = st.number_input("Clear Cover (mm)", 20.0, 75.0, state["cover"], 5.0, key='in_cover')
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.subheader("Material Properties")
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        state["fck"] = st.number_input("$f_{ck}$ (MPa, M-Grade)", 20.0, 80.0, state["fck"], 1.0, key='in_fck')
        state["fy"] = st.number_input("$f_{y}$ (MPa, Fe-Grade)", 415.0, 600.0, state["fy"], 5.0, key='in_fy')
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        st.subheader("Factored Loads")
        st.markdown("Inputs for $P_u$ (kN), $M_u$ (kNm), $V_u$ (kN)")
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        state["Pu"] = st.number_input("Axial Load $P_u$ (kN)", -3000.0, 6000.0, state["Pu"] / 1e3, 10.0, key='in_Pu') * 1e3
        state["Mux"] = st.number_input("Moment $M_{ux}$ (kNm)", -2000.0, 2000.0, state["Mux"] / 1e6, 5.0, key='in_Mux') * 1e6
        state["Muy"] = st.number_input("Moment $M_{uy}$ (kNm)", -2000.0, 2000.0, state["Muy"] / 1e6, 5.0, key='in_Muy') * 1e6
        state["Vu"] = st.number_input("Shear $V_u$ (kN)", 0.0, 5000.0, state["Vu"] / 1e3, 5.0, key='in_Vu') * 1e3
        st.markdown('</div>', unsafe_allow_html=True)
        
    b, D, cover, bars, Ag = recalculate_properties(state)
    st.markdown("---")
    st.markdown('<div class="print-break"></div>', unsafe_allow_html=True)

    # --- INPUT SECTION: LONGITUDINAL REBAR & SLENDERNESS (STEP 2 & 3) ---
    st.header("2️⃣ Reinforcement and Slenderness Setup")
    
    col_r1, col_r2, col_r3 = st.columns(3)
    bar_options = [12.0, 16.0, 20.0, 25.0, 28.0, 32.0]
    
    with col_r1:
        st.subheader("Longitudinal Bar Counts")
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        state["n_top"] = st.number_input("Top row bars ($n_{top}$)", 0, 10, state["n_top"], 1, key='in_ntop')
        state["n_bot"] = st.number_input("Bottom row bars ($n_{bot}$)", 0, 10, state["n_bot"], 1, key='in_nbot')
        state["n_left"] = st.number_input("Left column bars ($n_{left}$)", 0, 10, state["n_left"], 1, key='in_nleft')
        state["n_right"] = st.number_input("Right column bars ($n_{right}$)", 0, 10, state["n_right"], 1, key='in_nright')
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_r2:
        st.subheader("Bar Diameters & Layout")
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        state["dia_top"] = st.selectbox("Top bar $\\phi$ (mm)", bar_options, index=bar_options.index(state["dia_top"]), key='in_dtop')
        state["dia_bot"] = st.selectbox("Bottom bar $\\phi$ (mm)", bar_options, index=bar_options.index(state["dia_bot"]), key='in_dbot')
        state["dia_side"] = st.selectbox("Side bar $\\phi$ (mm)", bar_options[:-1], index=bar_options[:-1].index(state["dia_side"]), key='in_dside')
        st.markdown(f"**$A_{{st}}$ Provided:** **{state['As_long']:.0f} mm$^2$** ({state['As_long'] * 100 / Ag:.2f}% $\\rho$)")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_r3:
        st.subheader("Section Sketch")
        st.plotly_chart(plotly_cross_section(b, D, cover, bars, state['tie_dia'], state['tie_spacing']), use_container_width=True)

    st.markdown("---")
    
    # --- CALCULATION SECTION: SLENDERNESS & MAGNIFICATION (STEP 3) ---
    st.header("3️⃣ Slenderness Effects and Magnified Moments")
    
    col_sl_in, col_sl_out = st.columns(2)
    
    with col_sl_in:
        st.subheader("Slenderness Inputs (IS 456 Cl. 25.1)")
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        state["storey_clear"] = st.number_input("Clear Storey Height $l_0$ (mm)", 2000.0, 6000.0, state["storey_clear"], 50.0, key='in_l0')
        restraint_options = ["Fixed-Fixed", "Fixed-Pinned", "Pinned-Pinned", "Fixed-Free (cantilever)"]
        state["restraint"] = st.selectbox("End Restraint (for $k$ factor)", restraint_options, index=restraint_options.index(state["restraint"]), key='in_restraint')
        state["sway"] = st.checkbox("Check for Sway Frame (Moment Magnification)", value=state["sway"], key='in_sway')
        k_factor = effective_length_factor(state["restraint"])
        st.markdown(f"Calculated **$k$ Factor**: **{k_factor:.2f}**")
        st.markdown('</div>', unsafe_allow_html=True)
        
    b, D, cover, bars, Ag = recalculate_properties(state) 
    
    with col_sl_out:
        st.subheader("Slenderness and Magnification Check")
        st.markdown(f"**Slenderness Check (IS 456 Cl. 25.1.2):** Column is **{'Short' if state['short_x'] and state['short_y'] else 'Slender'}**.")
        st.metric(f"Eff. Length $l_{{e,x}}$ (mm) / $\lambda_x$", f"{state['le_x']:.0f} / {state['lam_x']:.1f}")
        st.metric(f"Eff. $M_{{ux,eff}}$ (kNm) ($\delta_x={state['delta_x']:.2f}$)", f"{kNm(state['Mux_eff'])}")
        st.metric(f"Eff. $M_{{uy,eff}}$ (kNm) ($\delta_y={state['delta_y']:.2f}$)", f"{kNm(state['Muy_eff'])}")

    st.markdown("#### Calculation Narration (Slenderness)")
    st.write(f"The column is classified as short if $\\lambda_{{x}} = l_{{e,x}}/r_{{x}} \le 12$. The slenderness $\\lambda_x$ is **{state['lam_x']:.1f}**. Since $\\lambda_x > 12$ (or $\\lambda_y > 12$), secondary moments must be accounted for by the moment magnification factor $\\delta$ (IS 456 Cl. 39.7).")
    st.latex(r"""
    M_{u,eff} = M_{u,base} \times \delta \quad \text{where } \delta = \frac{C_m}{1 - P_u/P_{cr}}
    """)
    st.markdown("---")
    st.markdown('<div class="print-break"></div>', unsafe_allow_html=True)

    # --- CALCULATION SECTION: BIAXIAL INTERACTION (STEP 4) ---
    st.header("4️⃣ Biaxial Interaction Strength Check")
    st.write("Strength is verified using the **Strain Compatibility** approach to find rigorous uniaxial capacities ($M_{u,lim}$) for the given $P_u$, followed by the **Power Law** interaction formula (IS 456 Annex G).")

    col_int_in, col_int_out = st.columns([1, 2])
    
    with col_int_in:
        st.subheader("Interaction Parameter")
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        state["alpha"] = st.slider("Interaction Exponent $\\alpha$ (IS 456 Annex G)", 0.8, 2.0, state["alpha"], 0.1, key='in_alpha_check')
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.subheader("Biaxial Check Result")
        st.latex(r"\left(\frac{M_{ux,eff}}{M_{ux,lim}}\right)^{\alpha} + \left(\frac{M_{uy,eff}}{M_{uy,lim}}\right)^{\alpha} \le 1.0")
        
        st.metric("Uniaxial $M_{ux,lim}$ (kNm)", f"{kNm(state['Mux_lim'])}")
        st.metric("Uniaxial $M_{uy,lim}$ (kNm)", f"{kNm(state['Muy_lim'])}")
        util_status = "✅ Biaxial PASS" if state['util'] <= 1.0 else "❌ Biaxial FAIL"
        st.markdown(f"**Utilization (Σ $\le 1$):** **{state['util']:.2f}** ({util_status})")
        
    with col_int_out:
        st.subheader("P–M Interaction Curves vs. Demand Point")
        fig_pmx = plot_pm_curve_plotly(b, D, bars, state["fck"], state["fy"], state["Pu"], state["Mux_eff"], 'x')
        st.plotly_chart(fig_pmx, use_container_width=True)
        fig_pmy = plot_pm_curve_plotly(b, D, bars, state["fck"], state["fy"], state["Pu"], state["Muy_eff"], 'y')
        st.plotly_chart(fig_pmy, use_container_width=True)
        
    st.markdown("---")
    st.markdown('<div class="print-break"></div>', unsafe_allow_html=True)

    # --- CALCULATION SECTION: SHEAR AND TIES (STEP 5 & 6) ---
    st.header("5️⃣ Shear and Tie Design (IS 456 Cl. 40 / IS 13920)")

    col_sh1, col_sh2, col_sh3 = st.columns(3)
    tie_options = [6.0, 8.0, 10.0, 12.0]
    
    with col_sh1:
        st.subheader("Tie Design Inputs")
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        state["tie_dia"] = st.selectbox("Tie $\\phi$ (mm)", tie_options, index=tie_options.index(state["tie_dia"]), key='in_tdia_shear')
        leg_options_map = {"2-legged": 2, "4-legged": 4, "6-legged": 6}
        current_legs_str = f"{state.get('legs', 4)}-legged"
        selected_legs_str = st.selectbox("Tie Legs ($A_{sv}$)", list(leg_options_map.keys()), index=list(leg_options_map.keys()).index(current_legs_str), key='in_legs')
        state["legs"] = leg_options_map[selected_legs_str]
        state["tie_spacing"] = st.number_input("Tie Spacing $s$ (mm) Provided", 50.0, 300.0, state["tie_spacing"], 5.0, key='in_ts_prov')
        st.markdown('</div>', unsafe_allow_html=True)
        
    b, D, cover, bars, Ag = recalculate_properties(state) 

    with col_sh2:
        st.subheader("Shear Capacity")
        st.write(f"Axial load factor $\\phi_N$ (IS 456 Cl. 40.2.2.1): **{state['phiN']:.2f}**")
        st.metric("Concrete Capacity $V_c$ (kN)", f"{kN(state['Vc'])}")
        st.metric("Required Steel Capacity $V_{us}$ (kN)", f"{kN(state['Vus'])}")
        
    with col_sh3:
        st.subheader("Tie Spacing Check")
        st.metric("Shear Required $s$ (mm)", f"{state['s_required_shear']:.0f}")
        st.metric("Detailing Cap $s$ (mm)", f"{state['s_cap_detailing']:.0f}")
        s_status = "✅ Spacing PASS" if state["tie_spacing"] <= state["s_governing_tie"] else "❌ Spacing FAIL"
        st.markdown(f"**Governing $s$ $\le$ {state['s_governing_tie']:.0f} mm:** **({s_status})**")
        
    st.markdown("#### Ductile Detailing Requirements (IS 13920)")
    lo = max(state['D'], state['b'], state['storey_clear']/6.0, 450.0)
    st.write(f"**Confinement Zone Length ($l_0$):** $l_0$ is the larger of $D$ ({D:.0f} mm), $b$ ({b:.0f} mm), $L_{{clear}}/6$ ({state['storey_clear']/6.0:.0f} mm), or $450$ mm. Governs to **{lo:.0f} mm** at each end.")
    st.write(f"**Spacing in Confinement Zone:** Ties must not exceed **{state['s_13920_confine']:.0f} mm** (min($\min(b,D)/4$, 100 mm)).")
    
    st.markdown("---")
    st.markdown('<div class="print-break"></div>', unsafe_allow_html=True)

    # --- FINAL REPORT SUMMARY (STEP 7) ---
    st.header("6️⃣ Final Submission Report Summary")
    st.caption("This entire page is designed to be printed as a complete calculation sheet.")

    # 1. Inputs & Outputs Table
    st.subheader("A. Key Design Inputs and Governing Results")
    
    As_min = 0.008 * Ag 
    As_governing_req = max(As_min, state['As_long'] * (state.get("util", 1.0) ** (1/state.get("alpha", 1.0))))

    out = {
        "Cross Section b x D (mm)": f"{b:.0f} x {D:.0f}", 
        "Material fck / fy (MPa)": f"{state['fck']:.0f} / {state['fy']:.0f}",
        "Design Pu (kN)": state["Pu"] / 1e3, 
        "Design Mux,eff / Muy,eff (kNm)": f"{kNm(state['Mux_eff'])} / {kNm(state['Muy_eff'])}",
        "Slenderness $\\lambda_x / \lambda_y$": f"{state['lam_x']:.1f} / {state['lam_y']:.1f} ({'Short' if state['short_x'] and state['short_y'] else 'Slender'})", 
        "Magnifiers $\delta_x / \delta_y$": f"{state['delta_x']:.2f} / {state['delta_y']:.2f}", 
        "Uniaxial Capacity $M_{u,lim}$ (kNm)": f"{kNm(state['Mux_lim'])} / {kNm(state['Muy_lim'])}",
        "Biaxial Utilization ($\le 1$)": f"**{state['util']:.2f}**",
        "Ast Provided (mm²)": f"{state['As_long']:.0f}",
        "Ast Governing Required (mm²)": f"{As_governing_req:.0f}",
        "Concrete Shear $V_c$ (kN)": f"{kN(state['Vc'])}",
        "Tie Spacing Provided (mm)": f"Ø{state['tie_dia']:.0f} @ {state['tie_spacing']:.0f} (Legs: {state['legs']})",
        "Tie Spacing Governing Req (mm)": f"{state['s_governing_tie']:.0f}",
        "13920 Confined Zone Limit (mm)": f"{state['s_13920_confine']:.0f}"
    }

    df_out = pd.DataFrame({"Parameter": list(out.keys()), "Value": list(out.values())})
    st.dataframe(df_out, use_container_width=True)

    st.markdown("---")
    st.subheader("B. Detailed Calculation Figures")
    st.write("Cross-section and P-M interaction curves are plotted below to verify capacity.")
    
    fig_sec_report = plotly_cross_section(b, D, cover, bars, state['tie_dia'], state['tie_spacing'])
    st.plotly_chart(fig_sec_report, use_container_width=True)
    st.markdown('<div class="print-break"></div>', unsafe_allow_html=True)
    
    col_rep_fig1, col_rep_fig2 = st.columns(2)
    with col_rep_fig1:
        fig_pmx_report = plot_pm_curve_plotly(b, D, bars, state["fck"], state["fy"], state["Pu"], state["Mux_eff"], 'x')
        st.plotly_chart(fig_pmx_report, use_container_width=True)
    with col_rep_fig2:
        fig_pmy_report = plot_pm_curve_plotly(b, D, bars, state["fck"], state["fy"], state["Pu"], state["Muy_eff"], 'y')
        st.plotly_chart(fig_pmy_report, use_container_width=True)
        
    st.markdown("---")
    st.caption("Final Note: Calculations follow the Limit State Method of IS 456 (2000) with ductile detailing checks from IS 13920 (2016).")


if __name__ == "__main__":
    main()
