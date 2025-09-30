import math
import json
import base64
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# -----------------------------
# 1. CONSTANTS AND UTILITIES
# -----------------------------
ES = 200000.0  # MPa (Modulus of Elasticity of Steel)
EPS_CU = 0.0035  # Ultimate concrete compressive strain

def bar_area(dia_mm: float) -> float:
    return math.pi * (dia_mm ** 2) / 4.0

def kN(value: float) -> str:
    # Format force in kN
    return f"{value / 1e3:.1f}"

def kNm(value: float) -> str:
    # Format moment in kNm
    return f"{value / 1e6:.1f}"

def effective_length_factor(restraint: str) -> float:
    # IS 456-2000 Table 28 approximation
    if restraint == "Fixed-Fixed": return 0.65
    if restraint == "Fixed-Pinned": return 0.8
    if restraint == "Pinned-Pinned": return 1.0
    if restraint == "Fixed-Free (cantilever)": return 2.0
    return 1.0

def moment_magnifier(Pu: float, le_mm: float, fck: float, Ic: float, Cm: float = 0.85, sway: bool = False) -> float:
    """Calculates moment magnification factor (Delta_b or Delta_s) based on IS 456."""
    Ec = 5000.0 * math.sqrt(max(fck, 1e-6))
    
    # Critical buckling load Pcr = (pi^2 * 0.4 * Ec * Ic) / (le^2)
    Pcr = (math.pi ** 2) * 0.4 * Ec * Ic / (le_mm ** 2 + 1e-9)
    
    if Pcr <= Pu: return 10.0
    
    delta = 1.0 / max(1e-6, (1.0 - Pu / Pcr))
    
    if not sway:
        # Non-sway (braced) moment magnifier Delta_b
        delta = max(1.0, Cm * delta)
    
    # Clip delta to reasonable maximums
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
    href = f'<a href="data:file/json;base64,{b64}" download="{filename}">💾 Download State as JSON</a>'
    return href

# -----------------------------
# 2. CORE ENGINEERING LOGIC
# -----------------------------

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
    # Effective cover to the center of the bar
    max_dia = max(dia_top, dia_bot, dia_side)
    dx_corner = cover + state["tie_dia"] + max_dia / 2.0
    
    # Check for minimal section size
    if dx_corner > min(b, D) / 2.0: dx_corner = min(b, D) / 2.0 - 5.0

    # Top Bars (y = D - dx_corner)
    if n_top > 0:
        x_coords = _linspace_points(dx_corner, b - dx_corner, n_top)
        for x in x_coords: bars_list.append((x, D - dx_corner, dia_top))
    
    # Bottom Bars (y = dx_corner)
    if n_bot > 0:
        x_coords = _linspace_points(dx_corner, b - dx_corner, n_bot)
        for x in x_coords: bars_list.append((x, dx_corner, dia_bot))
            
    # Side Bars (Left/Right) - Ensure corners are covered by Top/Bottom
    y_start, y_end = dx_corner, D - dx_corner
    
    # Left Bars (x = dx_corner)
    if n_left > 0:
        y_coords_l = _linspace_points(y_start, y_end, n_left)
        for y in y_coords_l: bars_list.append((dx_corner, y, dia_side))
        
    # Right Bars (x = b - dx_corner)
    if n_right > 0:
        y_coords_r = _linspace_points(y_start, y_end, n_right)
        for y in y_coords_r: bars_list.append((b - dx_corner, y, dia_side))

    # Remove duplicates, keep max diameter at corners
    unique_locations = {}
    for x, y, dia in bars_list:
        loc = (round(x, 4), round(y, 4))
        if loc not in unique_locations or dia > unique_locations[loc][2]:
            unique_locations[loc] = (x, y, dia)
            
    return sorted([v for v in unique_locations.values()], key=lambda x: (x[1], x[0]))

def _uniaxial_capacity_Mu_for_Pu(b: float, D: float, bars: List[Tuple[float, float, float]], fck: float, fy: float, Pu: float, axis: str) -> float:
    """Calculates the uniaxial moment capacity Mu_lim for a given axial load Pu."""
    dimension = D if axis == 'x' else b 
    
    def forces_and_moment(c: float):
        """Calculates total Axial Force (N) and Moment (M) about the geometric center."""
        # Concrete compression force (Cc) and moment contribution (Mc)
        Cc = 0.36 * fck * dimension * min(c, dimension)
        arm_Cc = 0.5 * dimension - 0.42 * min(c, dimension)
        Mc = Cc * arm_Cc
        
        Fs, Ms = 0.0, 0.0
        
        for (x_abs, y_abs, dia) in bars:
            As = bar_area(dia)
            y = (D - y_abs) if axis == 'x' else x_abs # Distance from compression face
            
            # Strain at the steel level
            strain = EPS_CU * (1.0 - (y / max(c, 1e-6)))
            
            # Steel Stress (IS 456-2000 curve, clipped)
            stress = np.clip(ES * strain, -0.87 * fy, 0.87 * fy)
            force = stress * As
            
            # Distance from geometric center
            z = 0.5 * dimension - y 
            
            Fs += force 
            Ms += force * z
            
        return Cc + Fs, Mc + Ms

    target = Pu 
    
    # Binary search for 'c'
    cL, cR = 0.01 * dimension, 1.50 * dimension 
    NL, ML = forces_and_moment(cL)
    NR, MR = forces_and_moment(cR)

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

def biaxial_utilization(b: float, D: float, bars: List[Tuple[float, float, float]], fck: float, fy: float, Pu: float, Mux_eff: float, Muy_eff: float, alpha: float) -> Tuple[float, float, float]:
    """Performs the biaxial interaction check using the Power Law (IS 456 Annex G)."""
    
    Mux_lim = _uniaxial_capacity_Mu_for_Pu(b, D, bars, fck, fy, Pu, axis='x')
    Muy_lim = _uniaxial_capacity_Mu_for_Pu(b, D, bars, fck, fy, Pu, axis='y')
    
    # Handle near-zero moments/capacities
    Mux_lim = max(Mux_lim, 1e-3) if abs(Mux_eff) > 1e-3 else Mux_lim
    Muy_lim = max(Muy_lim, 1e-3) if abs(Muy_eff) > 1e-3 else Muy_lim
        
    # (Mux_eff / Mux_lim)^alpha + (Muy_eff / Muy_lim)^alpha
    Rx = (abs(Mux_eff) / Mux_lim) ** alpha
    Ry = (abs(Muy_eff) / Muy_lim) ** alpha
    util = Rx + Ry
    
    return util, Mux_lim, Muy_lim

# ----------------------------
# 3. PLOTLY FUNCTIONS 
# ----------------------------

def plotly_cross_section(b: float, D: float, cover: float, bars: List[Tuple[float, float, float]]) -> go.Figure:
    """Generates a Plotly figure of the column cross-section and rebar layout."""
    fig = go.Figure()
    
    # Concrete Outline
    fig.add_shape(type="rect", x0=0, y0=0, x1=b, y1=D, line=dict(color="black", width=2), fillcolor="rgba(240, 240, 240, 0.8)"))
    
    # Effective Cover Boundary
    effective_cover = cover + st.session_state.state.get("tie_dia", 8.0)
    fig.add_shape(type="rect", x0=effective_cover, y0=effective_cover, x1=b - effective_cover, y1=D - effective_cover, 
                  line=dict(color="gray", width=1, dash="dot"), fillcolor="rgba(0,0,0,0)"))
    
    # Rebar scatter plot
    bar_x, bar_y, bar_size, bar_text = [], [], [], []
    for x_geom, y_geom, dia in bars:
        bar_x.append(x_geom); bar_y.append(y_geom); bar_size.append(dia * 2) 
        bar_text.append(f"Ø{dia:.0f} mm ({bar_area(dia):.0f} mm²)")
        
    fig.add_trace(go.Scatter(x=bar_x, y=bar_y, mode='markers', name='Rebars',
        marker=dict(size=bar_size, sizemode='diameter', color='#0a66c2', line=dict(width=1, color='DarkSlateGrey')),
        hovertemplate='Size: %{text}<extra></extra>', text=bar_text))
        
    fig.update_layout(title=f"Cross Section (b={b:.0f} x D={D:.0f} mm)", width=400, 
                      height=400 * D / b if b != 0 else 400, 
                      showlegend=False, hovermode="closest", plot_bgcolor='white', 
                      yaxis_scaleanchor="x", yaxis_scaleratio=1)
    return fig

def plot_pm_curve_plotly(b: float, D: float, bars: List[Tuple[float, float, float]], fck: float, fy: float, Pu: float, Mu_eff: float, axis: str):
    """Generates a Plotly P-M interaction curve for a given axis (x or y)."""
    dimension = D if axis == 'x' else b
    
    def forces_and_moment_for_curve(c: float):
        Cc = 0.36 * fck * dimension * min(c, dimension)
        arm_Cc = 0.5 * dimension - 0.42 * min(c, dimension)
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
        return Cc + Fs, Mc + Ms

    cs = np.linspace(0.01 * dimension, 1.50 * dimension, 120)
    P_list, M_list = [], []
    for c in cs:
        N, M = forces_and_moment_for_curve(c)
        P_list.append(N); M_list.append(abs(M))

    df = pd.DataFrame({'P (kN)': np.array(P_list)/1e3, f'M_{axis} (kNm)': np.array(M_list)/1e6})
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['P (kN)'], y=df[f'M_{axis} (kNm)'], mode='lines', 
                             name='Capacity Envelope', line=dict(color='#0a66c2', width=3)))
                             
    fig.add_trace(go.Scatter(x=[Pu/1e3], y=[abs(Mu_eff)/1e6], mode='markers', 
                             name='Design Demand', 
                             marker=dict(symbol='x', size=12, color='red', line=dict(width=2, color='red'))))
                             
    fig.update_layout(title=f'P–M Capacity Envelope (Axis: {axis.upper()})', 
                      xaxis_title='Axial Load P (kN)', 
                      yaxis_title=f'Moment M_{axis} (kNm)', 
                      hovermode="x unified", 
                      margin=dict(l=20, r=20, t=40, b=20), 
                      plot_bgcolor='white')
    return fig

# ----------------------------
# 4. MAIN APPLICATION LOGIC
# ----------------------------

def initialize_state():
    """Initializes default session state variables."""
    if "state" not in st.session_state:
        st.session_state.state = {
            "b": 450.0, "D": 600.0, "cover": 40.0, "fck": 30.0, "fy": 500.0,
            "Pu": 1200e3, "Mux": 120e6, "Muy": 80e6, "Vu": 150e3,
            "storey_clear": 3200.0, "kx": 1.0, "ky": 1.0, "restraint": "Pinned-Pinned", "sway": False,
            "n_top": 3, "n_bot": 3, "n_left": 2, "n_right": 2, 
            "dia_top": 16.0, "dia_bot": 16.0, "dia_side": 12.0, "tie_dia": 8.0, "tie_spacing": 150.0,
            "alpha": 1.0, "bars": [], "legs": 2
        }

def recalculate_properties(state):
    """Performs all core engineering calculations and updates the state."""
    b, D, cover, fck, fy = state["b"], state["D"], state["cover"], state["fck"], state["fy"]
    Pu, Mux, Muy, Vu = state["Pu"], state["Mux"], state["Muy"], state["Vu"]
    lo = state["storey_clear"]

    # 1. Bar Layout & Area
    bars = _generate_bar_layout(b, D, cover, state)
    As_long = sum(bar_area(dia) for _, _, dia in bars)
    state.update({"As_long": As_long, "bars": bars})
    Ag = b * D
    Ic_x = b * D**3 / 12.0
    Ic_y = D * b**3 / 12.0
    rx = math.sqrt(Ic_x / Ag) # Radius of gyration
    ry = math.sqrt(Ic_y / Ag)

    # 2. Slenderness & Magnification
    k_factor = effective_length_factor(state["restraint"])
    le_x, le_y = k_factor * lo, k_factor * lo
    lam_x, lam_y = le_x / max(rx, 1e-6), le_y / max(ry, 1e-6)
    
    # IS 456 Clause 25.1.2: Short if le/D (or le/b) <= 12.0
    short_x = lam_x <= 12.0
    short_y = lam_y <= 12.0
    state.update(dict(le_x=le_x, le_y=le_y, lam_x=lam_x, lam_y=lam_y, short_x=short_x, short_y=short_y, kx=k_factor))

    # Calculate moment magnifiers
    delta_x = moment_magnifier(Pu, le_x, fck, Ic_x, sway=state["sway"]) if not short_x else 1.0
    delta_y = moment_magnifier(Pu, le_y, fck, Ic_y, sway=state["sway"]) if not short_y else 1.0
    Mux_eff, Muy_eff = Mux * delta_x, Muy * delta_y
    state.update(dict(Mux_eff=Mux_eff, Muy_eff=Muy_eff, delta_x=delta_x, delta_y=delta_y))
    
    # 3. Biaxial Utilization
    util, Mux_lim, Muy_lim = biaxial_utilization(b, D, bars, fck, fy, Pu, Mux_eff, Muy_eff, state["alpha"])
    state.update(dict(Mux_lim=Mux_lim, Muy_lim=Muy_lim, util=util))

    # 4. Shear Design (V_c and V_us)
    max_long_dia = max([bar[2] for bar in bars], default=16.0)
    d_eff = D - (cover + state["tie_dia"] + 0.5 * max_long_dia) # Effective depth
    
    # Axial Load Factor (phi_N) - IS 456, Cl. 40.2.2.1
    phiN = float(np.clip(1.0 + (Pu / max(1.0, (0.25 * fck * Ag))), 0.5, 1.5))
    
    # Nominal shear strength of concrete (tau_c)
    tau_c_base = 0.62 * math.sqrt(fck) / 1.0 # 0.62 * sqrt(fck)
    Vc = tau_c_base * b * d_eff * phiN
    
    Vus = max(0.0, Vu - Vc) # Required shear reinforcement capacity
    state.update(dict(Vc=Vc, Vus=Vus, phiN=phiN, d_eff=d_eff))

    # 5. Tie Spacing Requirements (IS 456 Cl. 26.5.3.2 and IS 13920 Cl. 7.4.7 for ductile detailing)
    Asv = state["legs"] * bar_area(state["tie_dia"])
    
    # Spacing required for shear 
    s_required_shear = (0.87 * fy * Asv * d_eff) / max(Vus, 1e-6) if Vus > 0 else 300.0
    
    # Detailing caps
    s_cap_detailing = min(16.0 * max_long_dia, min(b, D), 300.0)
    
    s_governing_tie = min(s_cap_detailing, s_required_shear)
    state.update({"s_required_shear": s_required_shear, "s_cap_detailing": s_cap_detailing, "s_governing_tie": s_governing_tie})
    
    return b, D, cover, bars, Ag

def main():
    st.set_page_config(page_title="RCC Column (Biaxial) Designer", layout="wide")
    st.title("🧱 RCC Column Design — Biaxial Moments ± Axial ± Shear (IS 456/13920)")
    st.markdown("---")

    initialize_state()
    state = st.session_state.state

    # --- 0. Data Management ---
    st.header("🗄️ Data Management (Import / Export)")
    c_up, c_down = st.columns(2)
    with c_up:
        uploaded_file = st.file_uploader("Upload State (JSON)", type="json")
        if uploaded_file is not None:
            try:
                data = json.load(uploaded_file)
                if 'bars' in data and isinstance(data['bars'], list): 
                    data['bars'] = [tuple(item) for item in data['bars']]
                st.session_state.state.update(data)
                st.rerun() 
            except Exception as e:
                st.error(f"Error loading JSON: {e}")
    with c_down:
        if state.get("bars"): 
            st.markdown("**Export Current Design State**")
            st.markdown(get_json_download_link(to_json_serializable(state), "column_design_state.json"), unsafe_allow_html=True)
    st.markdown("---")

    # --- 1. Inputs (Geometry, Materials, Loads, Detailing) ---
    st.header("1️⃣ Design Inputs")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        state["b"] = st.number_input("Width $b$ (mm)", 200.0, 2000.0, state["b"], 25.0, key='in_b')
        state["D"] = st.number_input("Depth $D$ (mm)", 200.0, 3000.0, state["D"], 25.0, key='in_D')
        state["cover"] = st.number_input("Clear Cover (mm)", 20.0, 75.0, state["cover"], 5.0, key='in_cover')
    with c2:
        state["fck"] = st.number_input("$f_{ck}$ (MPa, M-Grade)", 20.0, 80.0, state["fck"], 1.0, key='in_fck')
        state["fy"] = st.number_input("$f_{y}$ (MPa, Fe-Grade)", 415.0, 600.0, state["fy"], 5.0, key='in_fy')
        state["Pu"] = st.number_input("Axial Load $P_u$ (kN, +comp)", -3000.0, 6000.0, state["Pu"] / 1e3, 10.0, key='in_Pu') * 1e3
    with c3:
        state["Mux"] = st.number_input("Moment $M_{ux}$ (kNm)", -2000.0, 2000.0, state["Mux"] / 1e6, 5.0, key='in_Mux') * 1e6
        state["Muy"] = st.number_input("Moment $M_{uy}$ (kNm)", -2000.0, 2000.0, state["Muy"] / 1e6, 5.0, key='in_Muy') * 1e6
        state["Vu"] = st.number_input("Shear $V_u$ (kN)", 0.0, 5000.0, state["Vu"] / 1e3, 5.0, key='in_Vu') * 1e3
    with c4:
        state["storey_clear"] = st.number_input("Clear Storey Height $l_0$ (mm)", 2000.0, 6000.0, state["storey_clear"], 50.0, key='in_l0')
        restraint_options = ["Fixed-Fixed", "Fixed-Pinned", "Pinned-Pinned", "Fixed-Free (cantilever)"]
        state["restraint"] = st.selectbox("End Restraint", restraint_options, index=restraint_options.index(state["restraint"]), key='in_restraint')
        state["sway"] = st.checkbox("Sway Frame? (Yes/No)", value=state["sway"], key='in_sway')
        k_factor = effective_length_factor(state["restraint"])
        st.markdown(f"Effective Length Factor **$k$**: **{k_factor:.2f}**")

    # --- 2. Longitudinal Reinforcement Detailing ---
    st.header("2️⃣ Longitudinal Rebar Arrangement")
    cL1, cL2, cL3, cL4 = st.columns(4)
    bar_options = [12.0, 16.0, 20.0, 25.0, 28.0, 32.0]
    with cL1:
        state["n_top"] = st.number_input("Top row bars ($n_{top}$)", 0, 10, state["n_top"], 1, key='in_ntop')
        state["n_bot"] = st.number_input("Bottom row bars ($n_{bot}$)", 0, 10, state["n_bot"], 1, key='in_nbot')
    with cL2:
        state["dia_top"] = st.selectbox("Top bar $\\phi$ (mm)", bar_options, index=bar_options.index(state["dia_top"]), key='in_dtop')
        state["dia_bot"] = st.selectbox("Bottom bar $\\phi$ (mm)", bar_options, index=bar_options.index(state["dia_bot"]), key='in_dbot')
    with cL3:
        state["n_left"] = st.number_input("Left column bars ($n_{left}$)", 0, 10, state["n_left"], 1, key='in_nleft')
        state["n_right"] = st.number_input("Right column bars ($n_{right}$)", 0, 10, state["n_right"], 1, key='in_nright')
    with cL4:
        state["dia_side"] = st.selectbox("Side bar $\\phi$ (mm)", bar_options[:-1], index=bar_options[:-1].index(state["dia_side"]), key='in_dside')
        leg_options_map = {"2-legged": 2, "4-legged": 4, "6-legged": 6}
        current_legs
